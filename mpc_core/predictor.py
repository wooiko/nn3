"""Output predictor over horizon Np using KRR model."""
import numpy as np


class Predictor:
    """Wraps KRRModel.predict_horizon for use inside MPCController.

    The predictor delegates to the KRR model's multi-step forecast.
    A proper recurrent roll-out (feeding predicted outputs as regressor inputs)
    can be added here without changing the MPCController interface.
    """

    def __init__(self, cfg: dict):
        self._Np = int(cfg.get("Np") or 5)

    def predict(self, xi: np.ndarray, krr, Np: int | None = None) -> np.ndarray:
        """Return output forecast array of shape (Np, n_out).

        Args:
            xi  : current regressor vector (d,)
            krr : KRRModel instance (must be ready)
            Np  : prediction horizon (defaults to cfg value)
        """
        if Np is None:
            Np = self._Np
        return krr.predict_horizon(xi, Np)
