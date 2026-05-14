"""Unit tests: GP posterior variance consistency with analytic values."""
import numpy as np
import pytest
from kernel_models.gram_matrix import GramMatrix
from kernel_models.gp_supervisor import GPSupervisor


@pytest.fixture
def cfg():
    return {
        "kernel_length_scale": 1.0,
        "lambda_reg": 1e-4,
        "window_size": 20,
    }


@pytest.fixture
def gram(cfg):
    return GramMatrix(cfg)


def test_variance_nonnegative(cfg, gram):
    """GP posterior variance must be ≥ 0 everywhere."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((15, 2))
    gram.compute(X)
    gp = GPSupervisor(cfg, gram)
    gp.update(X)

    for _ in range(20):
        xi = rng.standard_normal(2)
        var = gp.variance_horizon(xi, Np=5)
        assert np.all(var >= 0.0), f"Negative variance: {var}"


def test_variance_increases_outside_training_domain(cfg, gram):
    """Variance at far-out-of-domain point ≥ variance at in-domain point."""
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (20, 2))
    gram.compute(X)
    gp = GPSupervisor(cfg, gram)
    gp.update(X)

    # In-domain: close to training centroid
    xi_in = np.zeros(2)
    # Out-of-domain: far from training data
    xi_out = np.array([100.0, 100.0])

    var_in = gp.variance_horizon(xi_in, Np=3)
    var_out = gp.variance_horizon(xi_out, Np=3)

    assert var_out.mean() >= var_in.mean(), (
        f"Expected higher variance out-of-domain: {var_out.mean():.4f} vs {var_in.mean():.4f}"
    )


def test_variance_shape(cfg, gram):
    """variance_horizon returns array of shape (Np,)."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((10, 3))
    gram.compute(X)
    gp = GPSupervisor(cfg, gram)
    gp.update(X)

    xi = rng.standard_normal(3)
    for Np in (1, 3, 10):
        var = gp.variance_horizon(xi, Np=Np)
        assert var.shape == (Np,), f"Expected ({Np},), got {var.shape}"


def test_prior_variance_when_no_data(cfg, gram):
    """Without compute(), variance defaults to prior (1.0)."""
    gp = GPSupervisor(cfg, gram)
    xi = np.zeros(2)
    var = gp.variance_horizon(xi, Np=3)
    assert np.allclose(var, 1.0), f"Expected prior variance 1.0, got {var}"


def test_last_variance_matches_horizon(cfg, gram):
    """last_variance() returns the mean σ² from the most recent call."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((10, 2))
    gram.compute(X)
    gp = GPSupervisor(cfg, gram)
    gp.update(X)

    xi = rng.standard_normal(2)
    var = gp.variance_horizon(xi, Np=5)
    assert abs(gp.last_variance() - var.mean()) < 1e-12
