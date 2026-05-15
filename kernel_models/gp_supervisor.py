"""GP supervisor: posterior variance σ²_GP(ξ(k)) over prediction horizon Np.

Uses the same (K+λI)⁻¹ as KRR — no extra inversion.
GP posterior variance: σ²(x*) = k(x*,x*) - k(x*,X)(K+λI)⁻¹k(X,x*)
"""
import numpy as np
from .gram_matrix import GramMatrix


class GPSupervisor:
    """GP posterior variance estimator sharing the GramMatrix with KRR.

    Parameters from cfg:
        kernel_length_scale : RBF length scale (via gram)
    """

    def __init__(self, cfg: dict, gram: GramMatrix):
        self._gram = gram
        self._X: np.ndarray | None = None
        self._log: list[dict] = []

    def update(self, X_train: np.ndarray) -> None:
        """Set the current training data (called after GramMatrix.compute)."""
        self._X = X_train.copy()

    def _posterior_var(self, x_star: np.ndarray) -> float:
        """GP posterior variance at a single query point x_star."""
        if self._X is None or self._gram._K_inv is None:
            return 1.0  # prior variance when no data available
        k_ss = 1.0  # k(x*,x*) = 1 for normalized RBF kernel
        k_vec = self._gram.kernel_vector(x_star)  # (n,)
        sigma2 = float(k_ss - k_vec @ self._gram.K_inv @ k_vec)
        return max(sigma2, 0.0)  # clip numerical negatives

    def variance_horizon(self, xi: np.ndarray, Np: int) -> np.ndarray:
        """Return σ²_GP for each step in [0, Np).

        Uses the same regressor xi for all steps (constant-input assumption
        for variance estimate); returns shape (Np,).
        Logs the mean variance for this call.
        """
        sigma2 = self._posterior_var(xi)
        result = np.full(Np, sigma2)
        self._log.append({"xi": xi.copy(), "sigma2_mean": sigma2, "Np": Np})
        return result

    def last_variance(self) -> float:
        """Return σ²_GP from the most recent variance_horizon call."""
        return self._log[-1]["sigma2_mean"] if self._log else 1.0
