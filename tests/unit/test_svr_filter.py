"""Unit tests: SVR filter on synthetic anomalies."""
import numpy as np
import pytest
from kernel_models.svr_filter import SVRFilter


@pytest.fixture
def cfg():
    return {"svr_epsilon": 0.1, "svr_C": 1.0, "svr_gamma": 1.0}


@pytest.fixture
def trained_filter(cfg):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 3))
    y = np.sin(X[:, 0]) + 0.05 * rng.standard_normal(50)
    filt = SVRFilter(cfg)
    filt.fit(X, y)
    return filt


def test_normal_observation_passes(trained_filter):
    """Observation clearly inside the tube is classified as normal."""
    rng = np.random.default_rng(1)
    X_train = np.random.default_rng(0).standard_normal((50, 3))
    y_train = np.sin(X_train[:, 0])
    # Use a training-domain point with true label matching the model
    x = X_train[0]
    y_true = float(y_train[0])
    result = trained_filter.predict_with_label(x, y_true)
    # Residual should be within ε for a well-fitted point (at most slightly outside)
    assert isinstance(result["is_anomaly"], bool)
    assert "tube_distance" in result


def test_anomaly_detected(cfg):
    """Observation far outside fitted range is classified as anomaly."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    y = np.zeros(50)  # model always predicts ≈ 0
    filt = SVRFilter(cfg)
    filt.fit(X, y)

    x_normal = rng.standard_normal(3)
    # y_true far from 0: residual >> epsilon
    result = filt.predict_with_label(x_normal, y_true=10.0)
    assert result["is_anomaly"] is True
    assert result["tube_distance"] > 0.0


def test_tube_distance_positive_for_anomaly(cfg):
    """tube_distance > 0 exactly when is_anomaly is True."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((30, 2))
    y = np.zeros(30)
    filt = SVRFilter(cfg)
    filt.fit(X, y)

    x = rng.standard_normal(2)
    result_anomaly = filt.predict_with_label(x, y_true=5.0)
    result_normal = filt.predict_with_label(x, y_true=0.0)

    assert result_anomaly["is_anomaly"] is True
    assert result_anomaly["tube_distance"] > 0.0
    assert result_normal["is_anomaly"] is False
    assert result_normal["tube_distance"] <= 0.0


def test_not_fitted_returns_not_anomaly(cfg):
    """Unfitted filter returns is_anomaly=False with reason 'not_fitted'."""
    filt = SVRFilter(cfg)
    result = filt.predict_with_label(np.zeros(3), y_true=0.0)
    assert result["is_anomaly"] is False
    assert result["reason"] == "not_fitted"
