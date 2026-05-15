"""QP solver for MPC using cvxpy.

Enforces control bounds and increment bounds |Δu| ≤ Δu_max.
Checks feasibility every step; logs solve status, objective value, and wall time.

Parameters from cfg:
    R0, Np, Nc, delta_u_max : see ClassicMPC
    B/rho/Q bounds
"""
import time
import numpy as np
import cvxpy as cp


class QPSolver:
    """cvxpy-based QP solver for one MPC step.

    Minimises:
        sum_{i=0}^{Nc-1} [ e'Q_w e + Δu_i' R Δu_i ]
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
        self._log: list[dict] = []

    def solve(self, y_pred: np.ndarray, y_ref: np.ndarray, u_prev: np.ndarray,
              R: np.ndarray) -> dict:
        """Solve QP and return optimal first control increment applied.

        Args:
            y_pred : predicted output at current step (n_y,)
            y_ref  : reference setpoint (n_y,)
            u_prev : control at previous step (n_u,)
            R      : penalty matrix (n_u × n_u)

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

        delta_u = cp.Variable((Nc, self._n_u))
        cost = 0.0
        constraints = []
        u_cur = u_prev.copy()
        e = y_pred - y_ref  # tracking error (constant over simplified horizon)
        for i in range(Nc):
            du_i = delta_u[i]
            u_cur = u_cur + du_i
            cost += cp.quad_form(e, self._Q_w) + cp.quad_form(du_i, R)
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
