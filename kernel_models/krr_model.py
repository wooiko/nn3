"""KRR nominal prediction model.

Trained on a sliding window; uses the shared Gram matrix from gram_matrix.py.
Regularizer λ is shared with GP module.
"""
import numpy as np
from .gram_matrix import GramMatrix


class KRRModel:
    """Kernel Ridge Regression over a sliding window.

    Parameters from cfg:
        window_size : int — sliding window length [steps]

    Depends on GramMatrix for (K+λI)⁻¹ — does NOT invert independently.
    """

    def __init__(self, cfg: dict, gram: GramMatrix):
        self._gram = gram
        self._window = int(cfg.get("window_size") or 30)
        self._X: list[np.ndarray] = []  # regressor vectors
        self._Y: list[np.ndarray] = []  # output vectors
        self._alpha: np.ndarray | None = None  # dual weights (n × n_out)

    def update_window(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        """Append (x_new, y_new) to the sliding window (drops oldest if full)."""
        self._X.append(x_new.copy())
        self._Y.append(y_new.copy())
        if len(self._X) > self._window:
            self._X.pop(0)
            self._Y.pop(0)
        self._alpha = None  # invalidate weights

    def _fit(self) -> None:
        """Compute dual weights α = (K+λI)⁻¹ Y using the shared Gram matrix."""
        if len(self._X) < 2:
            self._alpha = None
            return
        X = np.array(self._X)   # (n, d)
        Y = np.array(self._Y)   # (n, n_out)
        K_inv = self._gram.compute(X)  # also updates inversion_count
        self._alpha = K_inv @ Y        # (n, n_out)

    def predict_horizon(self, xi: np.ndarray, Np: int) -> np.ndarray:
        """Return nominal output forecast over horizon Np.

        Naïve multi-step: uses the same regressor xi for each step
        (proper recurrent roll-out requires the predictor module).

        Returns shape (Np, n_out).
        """
        if len(self._X) < 2:
            n_out = self._Y[0].shape[0] if self._Y else 1
            return np.zeros((Np, n_out))
        self._fit()
        if self._alpha is None:
            n_out = self._Y[0].shape[0]
            return np.zeros((Np, n_out))
        k_vec = self._gram.kernel_vector(xi)   # (n,)
        y_hat = k_vec @ self._alpha            # (n_out,)
        return np.tile(y_hat, (Np, 1))

    def is_ready(self) -> bool:
        return len(self._X) >= 2
