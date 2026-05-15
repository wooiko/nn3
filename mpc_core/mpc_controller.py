"""Full MPC control step.

Order: measurement → SVR filter → sliding window update → shared Gram matrix
       → KRR forecast + GP variance → mode & R(k) → QP → control output.

Regressor convention: xi(k) = [y(k-1), u(k-1)] so that the KRR model learns
to predict y(k) from the *previous* state — a predictive (not identity) mapping.
On the very first step y_prev = y_meas and u_prev = u_init (no data yet).
"""
import numpy as np
from kernel_models.gram_matrix import GramMatrix
from kernel_models.svr_filter import SVRFilter
from kernel_models.krr_model import KRRModel
from kernel_models.gp_supervisor import GPSupervisor
from mpc_core.predictor import Predictor
from mpc_core.r_adaptation import RAdaptation
from mpc_core.qp_solver import QPSolver


class MPCController:
    """Adaptive MPC with SVR filtering, KRR prediction, GP variance, and 3-level R.

    Parameters from cfg (merged global + object config):
        kernel_length_scale, lambda_reg, window_size
        svr_epsilon, svr_C, svr_gamma
        Np, Nc, delta_u_max
        R0, alpha_r, gamma_r, theta1, theta2, hysteresis
        B/rho/Q bounds
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._Np = int(cfg.get("Np") or 5)

        self._gram = GramMatrix(cfg)
        self._svr_beta = SVRFilter(cfg)   # SVR for βFe channel
        self._svr_eps = SVRFilter(cfg)    # SVR for ε channel
        self._krr = KRRModel(cfg, self._gram)
        self._gp = GPSupervisor(cfg, self._gram)
        self._predictor = Predictor(cfg)
        self._r_adapt = RAdaptation(cfg)
        self._qp = QPSolver(cfg)

        self._n_u = 3
        self._n_y = 2
        u_min = np.array([cfg["B_min_dom"], cfg["rho_min_dom"], cfg["Q_min_dom"]], dtype=float)
        u_max = np.array([cfg["B_max_dom"],  cfg["rho_max_dom"],  cfg["Q_max_dom"]],  dtype=float)
        self._u_prev = (u_min + u_max) / 2.0
        self._mode = "nominal"

        # Delayed state: xi(k) = [y(k-1), u(k-1)] for predictive regression
        self._y_prev: np.ndarray | None = None

        self._svr_fitted = False
        self._step_count = 0
        self._log: list[dict] = []

    def reset(self, u_init: np.ndarray | None = None) -> None:
        """Reset controller state (call before a new episode)."""
        u_min = np.array([self._cfg["B_min_dom"], self._cfg["rho_min_dom"], self._cfg["Q_min_dom"]], dtype=float)
        u_max = np.array([self._cfg["B_max_dom"],  self._cfg["rho_max_dom"],  self._cfg["Q_max_dom"]],  dtype=float)
        if u_init is not None:
            self._u_prev = u_init.copy()
        else:
            self._u_prev = (u_min + u_max) / 2.0
        self._mode = "nominal"
        self._krr = KRRModel(self._cfg, self._gram)
        self._gp = GPSupervisor(self._cfg, self._gram)
        self._svr_fitted = False
        self._y_prev = None
        self._step_count = 0
        self._log = []

    def step(self, y_meas: np.ndarray, y_ref: np.ndarray,
             u_meas: np.ndarray | None = None) -> np.ndarray:
        """Execute one control step and return u(k).

        Args:
            y_meas : current output measurement y(k) (n_y,)
            y_ref  : reference setpoint (n_y,)
            u_meas : control input u(k-1) that produced y_meas (n_u,);
                     if None, u_prev is assumed

        Returns:
            u_opt : optimal control u(k) (n_u,)
        """
        u_context = u_meas if u_meas is not None else self._u_prev.copy()

        # Regressor xi(k) = [y(k-1), u(k-1)] — past state predicts current output.
        # On the very first step, y_prev is unavailable; use y_meas as fallback
        # (window will not be used for training until step 2).
        y_prev = self._y_prev if self._y_prev is not None else y_meas.copy()
        xi = np.concatenate([y_prev, u_context])  # (n_y + n_u,)

        # 1. SVR filter: classify each output channel independently
        is_anomaly = False
        if self._svr_fitted:
            r_beta = self._svr_beta.predict_with_label(xi[: self._n_y], float(y_meas[0]))
            r_eps = self._svr_eps.predict_with_label(xi[: self._n_y], float(y_meas[1]))
            is_anomaly = r_beta["is_anomaly"] or r_eps["is_anomaly"]

        # 2. Update sliding window with (xi(k), y(k)) — skip anomalies
        if not is_anomaly:
            self._krr.update_window(xi, y_meas.copy())

        # 3. Fit SVR filters once window is large enough
        if not self._svr_fitted and len(self._krr._X) >= 10:
            X_win = np.array(self._krr._X)           # (n, n_y + n_u)
            Y_win = np.array(self._krr._Y)           # (n, n_y)
            X_feat = X_win[:, : self._n_y]           # use output part as features
            self._svr_beta.fit(X_feat, Y_win[:, 0])  # βFe channel
            self._svr_eps.fit(X_feat, Y_win[:, 1])   # ε   channel
            self._svr_fitted = True

        # 4. Shared Gram matrix + KRR forecast + GP variance
        if self._krr.is_ready():
            X_win = np.array(self._krr._X)
            K_inv = self._gram.compute(X_win)
            self._gp.update(X_win)
            y_pred_horizon = self._predictor.predict(xi, self._krr, self._Np)
            sigma2_horizon = self._gp.variance_horizon(xi, self._Np)
            sigma2 = float(sigma2_horizon.mean())
            y_pred = y_pred_horizon[0]  # first-step prediction for QP
        else:
            y_pred_horizon = None
            y_pred = y_meas.copy()
            sigma2 = 1.0  # prior: high uncertainty → conservative mode

        # 5. Three-level R adaptation
        R, self._mode = self._r_adapt.compute(sigma2, self._mode)

        # 6. Solve QP — pass full horizon so tracking term depends on Δu
        result = self._qp.solve(y_pred, y_ref, self._u_prev, R,
                                y_pred_horizon=y_pred_horizon)
        u_opt = result["u_opt"]

        # Advance delayed state
        self._y_prev = y_meas.copy()
        self._u_prev = u_opt.copy()

        entry = {
            "step": self._step_count,
            "is_anomaly": is_anomaly,
            "sigma2": sigma2,
            "mode": self._mode,
            "u_opt": u_opt.copy(),
            "y_pred": y_pred.copy(),
            "qp_status": result["status"],
        }
        self._log.append(entry)
        self._step_count += 1
        return u_opt


class MPCControllerNoSVR(MPCController):
    """Ablation variant: SVR filter disabled — all observations accepted as valid."""

    def step(self, y_meas: np.ndarray, y_ref: np.ndarray,
             u_meas: np.ndarray | None = None) -> np.ndarray:
        u_context = u_meas if u_meas is not None else self._u_prev.copy()

        y_prev = self._y_prev if self._y_prev is not None else y_meas.copy()
        xi = np.concatenate([y_prev, u_context])

        # No SVR filtering — always accept
        self._krr.update_window(xi, y_meas.copy())

        # Fit SVR if window ready (kept for API consistency, but never queried)
        if not self._svr_fitted and len(self._krr._X) >= 10:
            self._svr_fitted = True

        if self._krr.is_ready():
            X_win = np.array(self._krr._X)
            self._gram.compute(X_win)
            self._gp.update(X_win)
            y_pred_horizon = self._predictor.predict(xi, self._krr, self._Np)
            sigma2_horizon = self._gp.variance_horizon(xi, self._Np)
            sigma2 = float(sigma2_horizon.mean())
            y_pred = y_pred_horizon[0]
        else:
            y_pred_horizon = None
            y_pred = y_meas.copy()
            sigma2 = 1.0

        R, self._mode = self._r_adapt.compute(sigma2, self._mode)
        result = self._qp.solve(y_pred, y_ref, self._u_prev, R,
                                y_pred_horizon=y_pred_horizon)
        u_opt = result["u_opt"]

        self._y_prev = y_meas.copy()
        self._u_prev = u_opt.copy()

        self._log.append({
            "step": self._step_count,
            "is_anomaly": False,
            "sigma2": sigma2,
            "mode": self._mode,
            "u_opt": u_opt.copy(),
            "y_pred": y_pred.copy(),
            "qp_status": result["status"],
        })
        self._step_count += 1
        return u_opt
