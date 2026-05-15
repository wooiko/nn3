"""Baseline: MPC with R adaptation driven by prediction-error autocorrelation.

R(k) is scaled by a factor derived from the autocorrelation of the
one-step-ahead prediction error e(k) = y_meas(k) - y_pred(k-1):

    rho_corr = |autocorr(e, lag=1)| / autocorr(e, lag=0)
    R(k) = R0 * (1 + alpha_r * rho_corr)

where alpha_r is a configurable gain.  High autocorrelation (systematic
unmodelled dynamics) → larger R → more conservative control action.

The tracking cost at each horizon step is:
    e(i) = y_meas + G @ Σ_{j=0}^{i} Δu_j - y_ref
so the QP variable Δu actually influences the predicted output.

Parameters from cfg:
    R0          : float — base penalty diagonal
    alpha_r     : float — autocorrelation gain
    window_size : int   — window for autocorrelation estimate (reused)
    io_gain     : float — incremental output-input gain scale (default 1.0)
    Np, Nc, delta_u_max : same as ClassicMPC
    B/rho/Q bounds
"""
import time
import numpy as np
import cvxpy as cp


class PrototypeMPC:
    """MPC with autocorrelation-based R adaptation.

    Interface mirrors ClassicMPC:
        step(y_meas, y_ref) → u_opt (n_u,)
    """

    def __init__(self, cfg: dict):
        self._R0 = float(cfg.get("R0") or 1.0)
        self._alpha_r = float(cfg.get("alpha_r") or 1.0)
        self._Np = int(cfg.get("Np") or 5)
        self._Nc = int(cfg.get("Nc") or self._Np)
        self._du_max = float(cfg.get("delta_u_max") or 0.01)
        self._win = int(cfg.get("window_size") or 20)

        self._u_min = np.array([
            cfg["B_min_dom"],
            cfg["rho_min_dom"],
            cfg["Q_min_dom"],
        ], dtype=float)
        self._u_max = np.array([
            cfg["B_max_dom"],
            cfg["rho_max_dom"],
            cfg["Q_max_dom"],
        ], dtype=float)

        self._n_u = 3
        self._n_y = 2
        self._u_prev = (self._u_min + self._u_max) / 2.0
        self._Q_w = np.eye(self._n_y)

        # Incremental gain G (n_y × n_u): ∂y/∂u approximation at operating point.
        io_gain = float(cfg.get("io_gain") or 1.0)
        self._G = np.zeros((self._n_y, self._n_u))
        self._G[0, 0] = io_gain  # βFe sensitive to B
        self._G[1, 0] = io_gain  # ε   sensitive to B

        # Error buffer for autocorrelation estimation
        self._error_buf: list[np.ndarray] = []
        self._y_pred_prev: np.ndarray | None = None
        self._log: list[dict] = []

    def reset(self, u_init: np.ndarray | None = None) -> None:
        if u_init is not None:
            self._u_prev = u_init.copy()
        else:
            self._u_prev = (self._u_min + self._u_max) / 2.0
        self._error_buf = []
        self._y_pred_prev = None

    def _autocorr_scale(self) -> float:
        """Return R scaling factor from lag-1 autocorrelation of error buffer."""
        if len(self._error_buf) < 3:
            return 1.0
        errors = np.array(self._error_buf[-self._win:])  # (T, n_y)
        e_flat = errors.ravel()
        n = len(e_flat)
        mean = e_flat.mean()
        c0 = float(np.sum((e_flat - mean) ** 2)) / n
        if c0 < 1e-12:
            return 1.0
        c1 = float(np.sum((e_flat[:-1] - mean) * (e_flat[1:] - mean))) / n
        rho = abs(c1 / c0)
        return 1.0 + self._alpha_r * rho

    def step(self, y_meas: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        """Compute optimal control action with autocorrelation-adapted R.

        Args:
            y_meas : current measurement (n_y,)
            y_ref  : reference setpoint (n_y,)

        Returns:
            u_opt : optimal control (n_u,)
        """
        # Update error buffer
        if self._y_pred_prev is not None:
            e = y_meas - self._y_pred_prev
            self._error_buf.append(e.copy())

        scale = self._autocorr_scale()
        R = self._R0 * scale * np.eye(self._n_u)
        G = self._G  # (n_y, n_u)

        t0 = time.perf_counter()
        Nc = self._Nc

        delta_u = cp.Variable((Nc, self._n_u))
        cost = 0.0
        constraints = []
        u_cur = self._u_prev.copy()
        for i in range(Nc):
            du_i = delta_u[i]
            u_cur = u_cur + du_i
            # Predicted output at step i: y_meas + G @ (Σ_{j=0}^{i} Δu_j)
            delta_u_cum_i = cp.sum(delta_u[:i + 1], axis=0)
            y_hat_i = y_meas + G @ delta_u_cum_i
            e_track = y_hat_i - y_ref
            cost += cp.quad_form(e_track, self._Q_w) + cp.quad_form(du_i, R)
            constraints += [
                u_cur >= self._u_min,
                u_cur <= self._u_max,
                cp.abs(du_i) <= self._du_max,
            ]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.CLARABEL, warm_start=True)

        elapsed = time.perf_counter() - t0
        status = prob.status

        if status not in ("optimal", "optimal_inaccurate"):
            entry = {
                "status": status,
                "R_scale": scale,
                "objective": float("nan"),
                "elapsed": elapsed,
                "u_opt": self._u_prev.copy(),
            }
            self._log.append(entry)
            raise RuntimeError(
                f"PrototypeMPC QP infeasible at step {len(self._log)}: status={status}"
            )

        du_opt = delta_u.value[0]
        u_opt = np.clip(self._u_prev + du_opt, self._u_min, self._u_max)
        self._u_prev = u_opt.copy()

        # Store naive prediction for next error estimate
        self._y_pred_prev = y_meas.copy()

        entry = {
            "status": status,
            "R_scale": scale,
            "objective": float(prob.value),
            "elapsed": elapsed,
            "u_opt": u_opt.copy(),
        }
        self._log.append(entry)
        return u_opt
