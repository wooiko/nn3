"""Full MPC control step.

Order: measurement → SVR filter → sliding window update → shared Gram matrix
       → KRR forecast + GP variance → mode & R(k) → QP → control output.
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
        self._svr = SVRFilter(cfg)
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

        # SVR requires initial fit — will be done on first window fill
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
        self._step_count = 0
        self._log = []

    def step(self, y_meas: np.ndarray, y_ref: np.ndarray,
             u_meas: np.ndarray | None = None) -> np.ndarray:
        """Execute one control step and return u(k).

        Args:
            y_meas : current output measurement (n_y,)
            y_ref  : reference setpoint (n_y,)
            u_meas : current control input used to produce y_meas (n_u,);
                     if None, u_prev is assumed

        Returns:
            u_opt : optimal control (n_u,)
        """
        u_context = u_meas if u_meas is not None else self._u_prev.copy()

        # Build regressor: [y_meas, u_context]
        xi = np.concatenate([y_meas, u_context])  # (n_y + n_u,)

        # 1. SVR filter: classify observation
        is_anomaly = False
        if self._svr_fitted:
            result = self._svr.predict_with_label(xi[: self._n_y], float(np.mean(y_meas)))
            is_anomaly = result["is_anomaly"]

        # 2. Update sliding window (skip anomalies)
        if not is_anomaly:
            self._krr.update_window(xi, y_meas.copy())

        # 3. Fit SVR once window is large enough
        if not self._svr_fitted and len(self._krr._X) >= 10:
            X_win = np.array(self._krr._X)
            Y_win = np.array(self._krr._Y)
            y_scalar = Y_win[:, 0]  # fit on βFe channel
            self._svr.fit(X_win[:, : self._n_y], y_scalar)
            self._svr_fitted = True

        # 4. Shared Gram matrix + KRR forecast + GP variance
        if self._krr.is_ready():
            X_win = np.array(self._krr._X)
            K_inv = self._gram.compute(X_win)
            self._gp.update(X_win)
            y_pred_horizon = self._predictor.predict(xi, self._krr, self._Np)
            sigma2_horizon = self._gp.variance_horizon(xi, self._Np)
            sigma2 = float(sigma2_horizon.mean())
            y_pred = y_pred_horizon[0]  # use first-step prediction for QP
        else:
            y_pred = y_meas.copy()
            sigma2 = 1.0  # prior: high uncertainty → conservative mode

        # 5. Three-level R adaptation
        R, self._mode = self._r_adapt.compute(sigma2, self._mode)

        # 6. Solve QP
        result = self._qp.solve(y_pred, y_ref, self._u_prev, R)
        u_opt = result["u_opt"]
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
        xi = np.concatenate([y_meas, u_context])

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
            y_pred = y_meas.copy()
            sigma2 = 1.0

        R, self._mode = self._r_adapt.compute(sigma2, self._mode)
        result = self._qp.solve(y_pred, y_ref, self._u_prev, R)
        u_opt = result["u_opt"]
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
