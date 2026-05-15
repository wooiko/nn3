"""Integration tests: full MPC step on a simplified plant."""
import numpy as np
import pytest
import yaml
from pathlib import Path

from kernel_models.gram_matrix import GramMatrix
from kernel_models.krr_model import KRRModel
from kernel_models.gp_supervisor import GPSupervisor
from mpc_core.mpc_controller import MPCController

OBJ_CFG_PATH = Path(__file__).parents[2] / "object_model" / "object_config.yaml"


@pytest.fixture(scope="module")
def cfg():
    with open(OBJ_CFG_PATH, encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    # Fill in global params with defaults for tests
    obj.update({
        "kernel_length_scale": 1.0,
        "lambda_reg": 1e-3,
        "window_size": 20,
        "svr_epsilon": 0.1,
        "svr_C": 1.0,
        "svr_gamma": 1.0,
        "Np": 3,
        "Nc": 2,
        "delta_u_max": 0.005,
        "R0": 1.0,
        "alpha_r": 1.0,
        "gamma_r": 5.0,
        "theta1": 0.05,
        "theta2": 0.20,
        "hysteresis": False,
    })
    return obj


def test_full_step_returns_feasible_control(cfg):
    """MPCController.step() returns u within GOST bounds on every call."""
    ctrl = MPCController(cfg)
    u_min = np.array([cfg["B_min_dom"], cfg["rho_min_dom"], cfg["Q_min_dom"]])
    u_max = np.array([cfg["B_max_dom"],  cfg["rho_max_dom"],  cfg["Q_max_dom"]])

    rng = np.random.default_rng(0)
    y_ref = np.array([0.650, 0.920])

    for _ in range(30):
        y_meas = np.array([0.618 + rng.normal(0, 0.002),
                           0.891 + rng.normal(0, 0.003)])
        u = ctrl.step(y_meas, y_ref)
        assert u.shape == (3,)
        assert np.all(u >= u_min - 1e-6), f"u={u} below min {u_min}"
        assert np.all(u <= u_max + 1e-6), f"u={u} above max {u_max}"


def test_shared_gram_matches_separate_inversions(cfg):
    """Verify (K+λI)⁻¹ from GramMatrix equals a manual inversion up to rounding."""
    gram = GramMatrix(cfg)
    rng = np.random.default_rng(42)
    n, d = 15, 4
    X = rng.standard_normal((n, d))

    K_inv_shared = gram.compute(X)

    # Manual inversion: same RBF kernel + λI
    l = cfg["kernel_length_scale"]
    lam = cfg["lambda_reg"]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    sq = np.sum(diff ** 2, axis=-1)
    K = np.exp(-sq / (2 * l ** 2))
    K_inv_manual = np.linalg.inv(K + lam * np.eye(n))

    assert np.allclose(K_inv_shared, K_inv_manual, atol=1e-10), (
        f"Max diff: {np.abs(K_inv_shared - K_inv_manual).max():.2e}"
    )


def test_mode_transitions_logged(cfg):
    """After warm-up, mode log contains valid mode strings."""
    ctrl = MPCController(cfg)
    rng = np.random.default_rng(1)
    y_ref = np.array([0.650, 0.920])
    for _ in range(35):
        y_meas = np.array([0.618 + rng.normal(0, 0.002),
                           0.891 + rng.normal(0, 0.003)])
        ctrl.step(y_meas, y_ref)

    modes = [e["mode"] for e in ctrl._log]
    valid = {"nominal", "cautious", "conservative"}
    assert all(m in valid for m in modes), f"Unknown mode in log: {set(modes) - valid}"
