"""Unit tests: QP solver on problems with known solutions."""
import numpy as np
import pytest
from mpc_core.qp_solver import QPSolver


@pytest.fixture
def cfg():
    return {
        "B_min_dom": 0.065, "B_max_dom": 0.160,
        "rho_min_dom": 1500.0, "rho_max_dom": 1550.0,
        "Q_min_dom": 105.0, "Q_max_dom": 160.0,
        "Nc": 3,
        "delta_u_max": 0.005,
    }


def test_unconstrained_solution(cfg):
    """With a very loose delta_u_max, QP returns a feasible u_opt."""
    cfg2 = dict(cfg)
    cfg2["delta_u_max"] = 50.0  # very loose — effectively unconstrained
    solver = QPSolver(cfg2)

    u_prev = np.array([0.1125, 1525.0, 132.5])
    y_pred = np.array([0.60, 0.88])
    y_ref  = np.array([0.65, 0.92])
    R = np.eye(3)

    result = solver.solve(y_pred, y_ref, u_prev, R)
    assert result["status"] in ("optimal", "optimal_inaccurate")
    u_opt = result["u_opt"]
    assert u_opt.shape == (3,)
    assert np.all(u_opt >= np.array([cfg2["B_min_dom"], cfg2["rho_min_dom"], cfg2["Q_min_dom"]]))
    assert np.all(u_opt <= np.array([cfg2["B_max_dom"], cfg2["rho_max_dom"], cfg2["Q_max_dom"]]))


def test_constraint_satisfaction(cfg):
    """u_opt must satisfy box constraints and delta_u_max on every call."""
    solver = QPSolver(cfg)
    rng = np.random.default_rng(0)

    u_min = np.array([cfg["B_min_dom"], cfg["rho_min_dom"], cfg["Q_min_dom"]])
    u_max = np.array([cfg["B_max_dom"], cfg["rho_max_dom"], cfg["Q_max_dom"]])
    u_prev = (u_min + u_max) / 2.0

    for _ in range(10):
        y_pred = rng.uniform(0.58, 0.70, 2)
        y_ref  = rng.uniform(0.60, 0.68, 2)
        R = np.diag(rng.uniform(0.5, 2.0, 3))
        result = solver.solve(y_pred, y_ref, u_prev, R)
        u_opt = result["u_opt"]
        assert np.all(u_opt >= u_min - 1e-6)
        assert np.all(u_opt <= u_max + 1e-6)
        du = np.abs(u_opt - u_prev)
        assert np.all(du <= cfg["delta_u_max"] + 1e-6), f"Δu={du} exceeds delta_u_max={cfg['delta_u_max']}"
        u_prev = u_opt


def test_infeasibility_detected(cfg):
    """Deliberately infeasible QP raises RuntimeError (spec: infeasibility is error)."""
    cfg2 = dict(cfg)
    # Impossible: u_min = u_max + 1 (no feasible u)
    cfg2["B_min_dom"] = 0.20   # above B_max_dom=0.160 → infeasible box
    solver = QPSolver(cfg2)

    u_prev = np.array([0.1125, 1525.0, 132.5])
    y_pred = np.array([0.618, 0.891])
    y_ref  = np.array([0.650, 0.920])
    R = np.eye(3)

    with pytest.raises(RuntimeError, match="infeasible"):
        solver.solve(y_pred, y_ref, u_prev, R)
