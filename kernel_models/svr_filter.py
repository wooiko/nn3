"""SVR ε-tube anomaly filter.

Classifies each observation as normal/anomaly based on whether it falls
outside the ε-insensitive tube of a fitted SVR (RBF kernel).
"""
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


class SVRFilter:
    """SVR-based anomaly filter using an ε-insensitive tube.

    Parameters from cfg:
        svr_epsilon : float — ε-tube half-width
        svr_C       : float — regularization parameter C
        svr_gamma   : float — RBF bandwidth γ (kernel coefficient)

    Usage:
        fit(X, y)               — train on clean historical data
        predict(x)              — classify one observation
    """

    def __init__(self, cfg: dict):
        self._epsilon = float(cfg.get("svr_epsilon") or 0.1)
        self._C = float(cfg.get("svr_C") or 1.0)
        self._gamma = float(cfg.get("svr_gamma") or 1.0)
        self._svr: SVR | None = None
        self._scaler = StandardScaler()
        self._log: list[dict] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit SVR on training data X (n × d), y (n,).

        Parameters come from cfg; cross-validation can be invoked separately
        via fit_with_cv() to tune ε, C, γ on the provided data.
        """
        X_sc = self._scaler.fit_transform(X)
        self._svr = SVR(
            kernel="rbf",
            epsilon=self._epsilon,
            C=self._C,
            gamma=self._gamma,
        )
        self._svr.fit(X_sc, y)

    def fit_with_cv(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
        """Fit SVR with grid-search cross-validation over C and γ.

        Returns the best parameters found. Updates internal model.
        cv folds are used; gamma values span 3 orders around cfg value.
        """
        X_sc = self._scaler.fit_transform(X)
        param_grid = {
            "C": [self._C * f for f in (0.1, 1.0, 10.0)],
            "gamma": [self._gamma * f for f in (0.1, 1.0, 10.0)],
            "epsilon": [self._epsilon],
        }
        base = SVR(kernel="rbf")
        gs = GridSearchCV(base, param_grid, cv=cv, scoring="neg_mean_squared_error")
        gs.fit(X_sc, y)
        best = gs.best_params_
        self._C = best["C"]
        self._gamma = best["gamma"]
        self._svr = gs.best_estimator_
        return best

    def predict(self, x: np.ndarray, y_true: float | None = None) -> dict:
        """Classify one observation x (d,).

        When y_true is provided the residual |y_true - SVR(x)| is used to
        determine whether the point falls outside the ε-tube — this is the
        correct anomaly criterion.  When y_true is None the method falls back
        to predict_with_label using the SVR's own prediction as a stand-in
        (i.e. tube_distance = 0, always normal), which is only useful as a
        quick sanity check; callers that need real anomaly detection should
        always supply y_true or call predict_with_label directly.

        Returns:
            {'is_anomaly': bool, 'tube_distance': float, 'reason': str}

        tube_distance > 0 means x is outside the ε-tube (anomaly).
        tube_distance ≤ 0 means x is within the tube (normal).
        """
        if self._svr is None:
            return {"is_anomaly": False, "tube_distance": 0.0, "reason": "not_fitted"}
        x_sc = self._scaler.transform(x.reshape(1, -1))
        y_hat = float(self._svr.predict(x_sc)[0])
        if y_true is not None:
            residual = abs(y_true - y_hat)
        else:
            # No label available: residual is zero by definition (prediction == itself)
            residual = 0.0
        tube_dist = residual - self._epsilon
        is_anomaly = tube_dist > 0.0
        reason = "outside_tube" if is_anomaly else "inside_tube"
        entry = {"is_anomaly": is_anomaly, "tube_distance": float(tube_dist), "reason": reason}
        self._log.append(entry)
        return entry

    def predict_with_label(self, x: np.ndarray, y_true: float) -> dict:
        """Classify observation x with known target y_true.

        Computes residual = |y_true - SVR(x)| and checks against ε-tube.
        This is the main anomaly-detection interface for online filtering.
        """
        if self._svr is None:
            return {"is_anomaly": False, "tube_distance": 0.0, "reason": "not_fitted"}
        x_sc = self._scaler.transform(x.reshape(1, -1))
        y_pred = float(self._svr.predict(x_sc)[0])
        residual = abs(y_true - y_pred)
        tube_dist = residual - self._epsilon
        is_anomaly = tube_dist > 0.0
        reason = "outside_tube" if is_anomaly else "inside_tube"
        entry = {"is_anomaly": is_anomaly, "tube_distance": float(tube_dist), "reason": reason}
        self._log.append(entry)
        return entry
