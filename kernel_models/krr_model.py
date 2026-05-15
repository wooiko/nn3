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
        """Return nominal output forecast over horizon Np via recurrent roll-out.

        Regressor convention: xi = [y_prev (n_out,), u_prev (n_u,)].
        Roll-out: at each step the KRR output becomes y_prev for the next step
        while u_prev is held constant (open-loop prediction — no future control
        increments are assumed here; QPSolver applies the gain correction on top).

        Returns shape (Np, n_out).
        """
        if len(self._X) < 2:
            n_out = self._Y[0].shape[0] if self._Y else 1
            return np.zeros((Np, n_out))
        self._fit()
        if self._alpha is None:
            n_out = self._Y[0].shape[0]
            return np.zeros((Np, n_out))

        n_out = self._Y[0].shape[0]
        # Split xi into output and input parts
        u_part = xi[n_out:]   # control part held constant during roll-out
        xi_cur = xi.copy()
        horizon = np.empty((Np, n_out))
        for i in range(Np):
            k_vec = self._gram.kernel_vector(xi_cur)  # (n,)
            y_hat = k_vec @ self._alpha               # (n_out,)
            horizon[i] = y_hat
            # Advance regressor: replace output part with predicted output
            xi_cur = np.concatenate([y_hat, u_part])
        return horizon

    def is_ready(self) -> bool:
        return len(self._X) >= 2
