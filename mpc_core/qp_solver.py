"""QP solver for MPC using cvxpy.

Enforces control bounds and increment bounds |Δu| ≤ Δu_max.
Checks feasibility every step; logs solve status, objective value, and wall time.

Parameters from cfg:
    R0, Np, Nc, delta_u_max : see ClassicMPC
    B/rho/Q bounds
    io_gain : float or None — incremental output-input gain scalar (dy/du scale).
              Used to build the linear prediction model y(k+i) ≈ y_pred(k+i) + G @ Σ Δu.
              Default 1.0 if not provided.
"""
import time
import numpy as np
import cvxpy as cp


class QPSolver:
    """cvxpy-based QP solver for one MPC step.

    Minimises:
        sum_{i=0}^{Nc-1} [ e(i)'Q_w e(i) + Δu_i' R Δu_i ]
    where
        e(i) = y_pred_horizon[i] + G @ cumsum(Δu)[i] - y_ref
        G    : (n_y × n_u) incremental gain matrix (from cfg or identity*io_gain)

    This ensures the tracking term depends on the optimisation variable Δu,
    so the solver can drive the predicted output toward the setpoint.

    subject to:
        u_min ≤ u_k ≤ u_max   for each step
        |Δu_i| ≤ Δu_max       element-wise

    A QP with no feasible solution raises RuntimeError (not silently ignored).
    """

    def __init__(self, cfg: dict):
        self._Nc = int(cfg.get("Nc") or cfg.get("Np") or 3)
        self._du_max = float(cfg.get("delta_u_max") or 0.01)
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
        self._Q_w = np.eye(self._n_y)  # output tracking weight

        # Incremental gain G (n_y × n_u): approximates ∂y/∂u at operating point.
        # Populated by set_gain(); defaults to scaled identity until first update.
        io_gain = float(cfg.get("io_gain") or 1.0)
        self._G = np.zeros((self._n_y, self._n_u))
        # Map each output to its most influential input (B→βFe, B→ε)
        self._G[0, 0] = io_gain  # βFe sensitive to B
        self._G[1, 0] = io_gain  # ε   sensitive to B

        self._log: list[dict] = []

    def set_gain(self, G: np.ndarray) -> None:
        """Update the incremental gain matrix G (n_y × n_u) from outside (e.g. numerical Jacobian)."""
        self._G = np.array(G, dtype=float)

    def solve(self, y_pred: np.ndarray, y_ref: np.ndarray, u_prev: np.ndarray,
              R: np.ndarray,
              y_pred_horizon: np.ndarray | None = None) -> dict:
        """Solve QP and return optimal first control increment applied.

        Args:
            y_pred          : predicted output at step k (n_y,) — used as y_pred_horizon[0]
                              when y_pred_horizon is not supplied.
            y_ref           : reference setpoint (n_y,)
            u_prev          : control at previous step (n_u,)
            R               : penalty matrix (n_u × n_u)
            y_pred_horizon  : optional (Nc, n_y) open-loop prediction over horizon.
                              If None, y_pred is broadcast across all Nc steps.

        Returns:
            {
              'u_opt'    : np.ndarray (n_u,) — optimal control (first step applied),
              'status'   : str — cvxpy solve status,
              'objective': float — optimal objective value,
              'elapsed'  : float — wall time [s],
            }

        Raises:
            RuntimeError if the problem is infeasible (spec: infeasibility is an error).
        """
        t0 = time.perf_counter()
        Nc = self._Nc
        G = self._G  # (n_y, n_u)

        if y_pred_horizon is None:
            # Broadcast single-step prediction across horizon
            y_pred_horizon = np.tile(y_pred, (Nc, 1))  # (Nc, n_y)
        else:
            # Use only first Nc steps; pad with last value if shorter
            if y_pred_horizon.shape[0] < Nc:
                pad = np.tile(y_pred_horizon[-1], (Nc - y_pred_horizon.shape[0], 1))
                y_pred_horizon = np.vstack([y_pred_horizon, pad])
            y_pred_horizon = y_pred_horizon[:Nc]

        delta_u = cp.Variable((Nc, self._n_u))
        cost = 0.0
        constraints = []
        u_cur = u_prev.copy()

        for i in range(Nc):
            du_i = delta_u[i]
            u_cur = u_cur + du_i
            # y(k+i) ≈ y_pred_horizon[i] + G @ (Σ_{j=0}^{i} Δu_j)
            # u_cur accumulates Δu incrementally, so u_cur - u_prev = Σ Δu_j
            y_hat_i = y_pred_horizon[i] + G @ (u_cur - u_prev)  # (n_y,) — cvxpy expression
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
            entry = {
                "u_opt": u_prev.copy(),
                "status": status,
                "objective": float("nan"),
                "elapsed": elapsed,
            }
            self._log.append(entry)
            raise RuntimeError(
                f"QPSolver: infeasible QP at step {len(self._log)}, status={status}"
            )

        du_opt = delta_u.value[0]
        u_opt = np.clip(u_prev + du_opt, self._u_min, self._u_max)
        entry = {
            "u_opt": u_opt.copy(),
            "status": status,
            "objective": float(prob.value),
            "elapsed": elapsed,
        }
        self._log.append(entry)
        return entry
