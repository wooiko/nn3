"""KPI calculations: RMSE, constraint violations, total variation, mode fractions."""
import numpy as np


def rmse(y_true: np.ndarray, y_ref: np.ndarray) -> float:
    """Root mean squared error between trajectory y_true and reference y_ref."""
    return float(np.sqrt(np.mean((y_true - y_ref) ** 2)))


def constraint_violations(y: np.ndarray, y_min: float, y_max: float) -> dict:
    """Count steps where y is outside [y_min, y_max].

    Returns:
        {'count': int, 'total_steps': int, 'fraction': float}
    """
    n = len(y)
    viol = int(np.sum((y < y_min) | (y > y_max)))
    return {"count": viol, "total_steps": n, "fraction": viol / n if n > 0 else 0.0}


def total_variation(u: np.ndarray) -> float:
    """Total variation of control trajectory u (T × n_u or T,)."""
    return float(np.sum(np.abs(np.diff(u, axis=0))))


def mode_fractions(modes: list[str]) -> dict:
    """Fraction of time in each R-adaptation mode.

    Returns:
        {'nominal': float, 'cautious': float, 'conservative': float}
    """
    n = len(modes)
    if n == 0:
        return {"nominal": 0.0, "cautious": 0.0, "conservative": 0.0}
    counts = {"nominal": 0, "cautious": 0, "conservative": 0}
    for m in modes:
        if m in counts:
            counts[m] += 1
    return {k: v / n for k, v in counts.items()}


def integral_criterion(y_true: np.ndarray, y_ref: np.ndarray,
                        u: np.ndarray, Q_w: np.ndarray, R_w: np.ndarray) -> float:
    """Integral quadratic criterion J = Σ [e'Q_w e + Δu'R_w Δu].

    Args:
        y_true : (T, n_y) trajectory
        y_ref  : (n_y,) or (T, n_y) reference
        u      : (T, n_u) control trajectory
        Q_w    : (n_y, n_y) output weight
        R_w    : (n_u, n_u) control weight
    """
    if y_ref.ndim == 1:
        y_ref = np.tile(y_ref, (len(y_true), 1))
    e = y_true - y_ref  # (T, n_y)
    du = np.diff(u, axis=0)  # (T-1, n_u)
    J_e = float(sum(e[t] @ Q_w @ e[t] for t in range(len(e))))
    J_u = float(sum(du[t] @ R_w @ du[t] for t in range(len(du))))
    return J_e + J_u


def compute_all(trace: dict, y_ref: np.ndarray,
                y_limits: dict | None = None) -> dict:
    """Compute all KPIs from a single-run trace dict.

    Args:
        trace    : dict with keys 'y' (T, n_y), 'u' (T, n_u), 'modes' (list)
        y_ref    : reference setpoint (n_y,)
        y_limits : {'beta_fe': (min, max), 'epsilon': (min, max)} — optional

    Returns:
        dict of all KPI values.
    """
    y = np.array(trace["y"])
    u = np.array(trace["u"])
    modes = trace.get("modes", [])
    n_y = y.shape[1] if y.ndim == 2 else 1
    n_u = u.shape[1] if u.ndim == 2 else 1

    kpis: dict = {}
    kpis["rmse_beta_fe"] = rmse(y[:, 0], y_ref[0])
    kpis["rmse_epsilon"] = rmse(y[:, 1], y_ref[1])

    if y_limits:
        bf_min, bf_max = y_limits.get("beta_fe", (0.0, 1.0))
        ep_min, ep_max = y_limits.get("epsilon", (0.0, 1.0))
        kpis["violations_beta_fe"] = constraint_violations(y[:, 0], bf_min, bf_max)
        kpis["violations_epsilon"] = constraint_violations(y[:, 1], ep_min, ep_max)

    kpis["tv_total"] = total_variation(u)
    kpis["tv_B"] = total_variation(u[:, 0]) if n_u >= 1 else 0.0
    kpis["tv_rho"] = total_variation(u[:, 1]) if n_u >= 2 else 0.0
    kpis["tv_Q"] = total_variation(u[:, 2]) if n_u >= 3 else 0.0

    kpis["mode_fractions"] = mode_fractions(modes)
    kpis["n_filtered"] = int(trace.get("n_filtered", 0))

    n_y2 = y.shape[1]
    n_u2 = u.shape[1]
    Q_w = np.eye(n_y2)
    R_w = np.eye(n_u2)
    kpis["J_integral"] = integral_criterion(y, y_ref, u, Q_w, R_w)

    return kpis
