"""Unit tests: KRR on known functions."""
import numpy as np
import pytest
from kernel_models.gram_matrix import GramMatrix
from kernel_models.krr_model import KRRModel


@pytest.fixture
def cfg():
    return {
        "kernel_length_scale": 1.0,
        "lambda_reg": 1e-4,
        "window_size": 30,
        "Np": 3,
    }


@pytest.fixture
def gram(cfg):
    return GramMatrix(cfg)


def test_constant_function(cfg, gram):
    """KRR should predict a near-constant value for constant training data."""
    model = KRRModel(cfg, gram)
    rng = np.random.default_rng(0)
    n = 20
    X = rng.standard_normal((n, 2))
    Y = np.ones((n, 1)) * 5.0  # constant target
    for x, y in zip(X, Y):
        model.update_window(x, y)

    x_test = rng.standard_normal(2)
    pred = model.predict_horizon(x_test, Np=3)
    assert pred.shape == (3, 1)
    # Should be close to 5.0 for points inside training domain
    x_in = X[0]
    pred_in = model.predict_horizon(x_in, Np=1)
    assert abs(pred_in[0, 0] - 5.0) < 0.5, f"Expected ~5.0, got {pred_in[0, 0]:.4f}"


def test_linear_function(cfg, gram):
    """KRR should extrapolate a linear function reasonably well."""
    model = KRRModel(cfg, gram)
    rng = np.random.default_rng(1)
    n = 25
    X = rng.uniform(-1, 1, (n, 1))
    Y = 2.0 * X  # y = 2x
    for x, y in zip(X, Y):
        model.update_window(x, y)

    x_test = np.array([0.0])
    pred = model.predict_horizon(x_test, Np=2)
    assert pred.shape == (2, 1)
    # At x=0, y=0
    assert abs(pred[0, 0]) < 0.3, f"Expected ~0 at x=0, got {pred[0, 0]:.4f}"


def test_gram_inversion_count_increments(cfg, gram):
    """Each predict_horizon call should increment inversion_count by 1."""
    model = KRRModel(cfg, gram)
    rng = np.random.default_rng(2)
    for _ in range(10):
        x = rng.standard_normal(2)
        y = rng.standard_normal(1)
        model.update_window(x, y)

    count_before = gram.inversion_count
    model.predict_horizon(rng.standard_normal(2), Np=3)
    assert gram.inversion_count == count_before + 1


def test_is_ready_after_two_samples(cfg, gram):
    """KRR reports not ready before 2 samples, ready after."""
    model = KRRModel(cfg, gram)
    assert not model.is_ready()
    model.update_window(np.zeros(2), np.zeros(1))
    assert not model.is_ready()
    model.update_window(np.ones(2), np.ones(1))
    assert model.is_ready()


def test_window_size_respected(cfg, gram):
    """Sliding window does not grow beyond window_size."""
    model = KRRModel(cfg, gram)
    rng = np.random.default_rng(3)
    for _ in range(50):
        model.update_window(rng.standard_normal(2), rng.standard_normal(1))
    assert len(model._X) <= cfg["window_size"]
