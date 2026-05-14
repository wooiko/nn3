"""Shared Gram matrix: computes (K + λI)⁻¹ once per step k.

Both KRR and GP modules consume the result; inversion count and timing are logged
to benchmark against the two-separate-inversions scheme (EXP-6).
"""
import time
import numpy as np


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, length_scale: float) -> np.ndarray:
    """RBF kernel matrix K[i,j] = exp(-||X[i]-Y[j]||² / (2*l²))."""
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]  # (n, m, d)
    sq_dist = np.sum(diff ** 2, axis=-1)
    return np.exp(-sq_dist / (2.0 * length_scale ** 2))


class GramMatrix:
    """Computes and caches (K + λI)⁻¹ for the current training window.

    Parameters from cfg:
        kernel_length_scale : RBF length scale
        lambda_reg          : regularization λ (shared with KRR and GP)
    """

    def __init__(self, cfg: dict):
        self._l = float(cfg.get("kernel_length_scale") or 1.0)
        self._lam = float(cfg.get("lambda_reg") or 1e-3)
        self.inversion_count: int = 0
        self.last_elapsed: float = 0.0
        self._K_inv: np.ndarray | None = None
        self._X_train: np.ndarray | None = None

    def compute(self, X: np.ndarray) -> np.ndarray:
        """Compute and return (K + λI)⁻¹ for training data X (n × d).

        Updates inversion_count and last_elapsed.
        """
        t0 = time.perf_counter()
        K = _rbf_kernel(X, X, self._l)
        n = K.shape[0]
        K_reg = K + self._lam * np.eye(n)
        K_inv = np.linalg.inv(K_reg)
        self.last_elapsed = time.perf_counter() - t0
        self.inversion_count += 1
        self._K_inv = K_inv
        self._X_train = X.copy()
        return K_inv

    def kernel_vector(self, x: np.ndarray) -> np.ndarray:
        """Return k(x, X_train): kernel vector (n,) for a new point x (d,)."""
        if self._X_train is None:
            raise RuntimeError("GramMatrix.compute() must be called before kernel_vector()")
        return _rbf_kernel(x[np.newaxis, :], self._X_train, self._l)[0]

    @property
    def K_inv(self) -> np.ndarray:
        if self._K_inv is None:
            raise RuntimeError("GramMatrix.compute() must be called first")
        return self._K_inv

    @property
    def X_train(self) -> np.ndarray:
        if self._X_train is None:
            raise RuntimeError("GramMatrix.compute() must be called first")
        return self._X_train

    @property
    def length_scale(self) -> float:
        return self._l
