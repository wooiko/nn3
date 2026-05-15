"""Baseline: MPC with fixed penalty matrix R₀ (no adaptation).

Uses a local cvxpy QP with:
  - quadratic output tracking term  Q_w * ||y_hat(i) - y_ref||²
  - quadratic input penalty term    R₀ * ||Δu||²
  - box constraints on u and Δu from object_config / config_global

The predicted output at step i along the horizon is:
    y_hat(i) = y_meas + G @ Σ_{j=0}^{i} Δu_j
where G (n_y × n_u) is an incremental gain matrix (cfg: io_gain scalar).
This makes the tracking cost depend on Δu so the QP actually drives the
output toward the setpoint.

Parameters from cfg (config_global.yaml merged with object_config.yaml):
    R0          : float — diagonal element of fixed penalty matrix R₀
    Np          : int   — prediction horizon
    Nc          : int   — control horizon (≤ Np)
    delta_u_max : float — max per-step control increment
    io_gain     : float — incremental output-input gain scale (default 1.0)
    B_min_dom, B_max_dom, Q_min_dom, Q_max_dom, rho_min_dom, rho_max_dom : control bounds
"""
import time
import numpy as np
import cvxpy as cp


class ClassicMPC:
    """MPC with fixed penalty matrix R₀.

    Interface mirrors the adaptive controller:
        step(y_meas, y_ref) → u_opt (n_u,)
    """

    def __init__(self, cfg: dict):
        self._R0 = float(cfg.get("R0") or 1.0)
        self._Np = int(cfg.get("Np") or 5)
        self._Nc = int(cfg.get("Nc") or self._Np)
        self._du_max = float(cfg.get("delta_u_max") or 0.01)

        # Control bounds: [B, rho, Q] — order matches u vector
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
        self._Q_w = np.eye(self._n_y)  # output weight (identity)

        # Incremental gain G (n_y × n_u): ∂y/∂u approximation at operating point.
        io_gain = float(cfg.get("io_gain") or 1.0)
        self._G = np.zeros((self._n_y, self._n_u))
        self._G[0, 0] = io_gain  # βFe sensitive to B
        self._G[1, 0] = io_gain  # ε   sensitive to B

        self._log: list[dict] = []

    def reset(self, u_init: np.ndarray | None = None) -> None:
        """Reset internal state (call before a new episode)."""
        if u_init is not None:
            self._u_prev = u_init.copy()
        else:
            self._u_prev = (self._u_min + self._u_max) / 2.0

    def step(self, y_meas: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
        """Compute optimal control action.

        Args:
            y_meas : current output measurement (n_y,) — used as base prediction
            y_ref  : reference setpoint (n_y,)

        Returns:
            u_opt : optimal control (n_u,)
        """
        t0 = time.perf_counter()
        n_u, Nc = self._n_u, self._Nc
        R = self._R0 * np.eye(n_u)
        G = self._G  # (n_y, n_u)

        delta_u = cp.Variable((Nc, n_u))
        cost = 0.0
        constraints = []
        u_cur = self._u_prev.copy()
        for i in range(Nc):
            du_i = delta_u[i]
            u_cur = u_cur + du_i
            # y(k+i) ≈ y_meas + G @ (Σ_{j=0}^{i} Δu_j); u_cur - u_prev = Σ Δu_j
            y_hat_i = y_meas + G @ (u_cur - self._u_prev)
            e_i = y_hat_i - y_ref
            cost += cp.quad_form(e_i, self._Q_w) + cp.quad_form(du_i, R)
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
            # QP infeasible — this is an error per spec (CLAUDE.md)
            entry = {
                "status": status,
                "objective": float("nan"),
                "elapsed": elapsed,
                "u_opt": self._u_prev.copy(),
            }
            self._log.append(entry)
            raise RuntimeError(
                f"ClassicMPC QP infeasible at step {len(self._log)}: status={status}"
            )

        du_opt = delta_u.value[0]  # first step of optimal sequence
        u_opt = np.clip(self._u_prev + du_opt, self._u_min, self._u_max)
        self._u_prev = u_opt.copy()

        entry = {
            "status": status,
            "objective": float(prob.value),
            "elapsed": elapsed,
            "u_opt": u_opt.copy(),
        }
        self._log.append(entry)
        return u_opt
