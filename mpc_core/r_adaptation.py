"""Three-level R(k) adaptation logic driven by σ²_GP.

Levels:
  nominal      σ²_GP < θ₁       →  R(k) = R₀
  cautious     θ₁ ≤ σ²_GP < θ₂  →  R(k) = R₀ + α·σ²_GP·I
  conservative σ²_GP ≥ θ₂       →  R(k) = γ·R₀

Hysteresis between levels is a configurable option.

Parameters from cfg:
    theta1     : float — nominal→cautious threshold
    theta2     : float — cautious→conservative threshold
    R0         : float — diagonal element of base penalty matrix
    alpha_r    : float — cautious-mode scaling coefficient
    gamma_r    : float — conservative-mode scaling factor
    hysteresis : bool  — enable transition hysteresis (default False)
"""
import numpy as np


_MODES = ("nominal", "cautious", "conservative")


class RAdaptation:
    """Three-level penalty matrix adaptation."""

    def __init__(self, cfg: dict):
        self._theta1 = float(cfg.get("theta1") or 0.05)
        self._theta2 = float(cfg.get("theta2") or 0.20)
        self._R0 = float(cfg.get("R0") or 1.0)
        self._alpha = float(cfg.get("alpha_r") or 1.0)
        self._gamma = float(cfg.get("gamma_r") or 5.0)
        self._hysteresis = bool(cfg.get("hysteresis", False))
        self._n_u = 3  # [B, rho, Q]
        self._log: list[dict] = []

    def _target_mode(self, sigma2: float) -> str:
        if sigma2 < self._theta1:
            return "nominal"
        if sigma2 < self._theta2:
            return "cautious"
        return "conservative"

    def _apply_hysteresis(self, target: str, prev_mode: str) -> str:
        """Allow transition only if it is an upgrade or a significant step up."""
        if not self._hysteresis:
            return target
        mode_idx = {m: i for i, m in enumerate(_MODES)}
        prev_i = mode_idx[prev_mode]
        tgt_i = mode_idx[target]
        # Downgrade (higher risk → lower risk): require crossing by 2 steps or θ/2 margin
        # Simple implementation: block single-step downgrade for one call
        if tgt_i < prev_i:
            return prev_mode  # hold current level for one step (one-step hysteresis)
        return target

    def compute(self, sigma2: float, prev_mode: str) -> tuple[np.ndarray, str]:
        """Return (R_matrix (n_u × n_u), mode_name).

        Args:
            sigma2    : current GP posterior variance (scalar)
            prev_mode : mode from previous step ('nominal'|'cautious'|'conservative')
        """
        target = self._target_mode(sigma2)
        mode = self._apply_hysteresis(target, prev_mode)

        R0_mat = self._R0 * np.eye(self._n_u)
        if mode == "nominal":
            R = R0_mat.copy()
        elif mode == "cautious":
            R = R0_mat + self._alpha * sigma2 * np.eye(self._n_u)
        else:  # conservative
            R = self._gamma * R0_mat

        entry = {"sigma2": sigma2, "mode": mode, "R_diag": float(R[0, 0])}
        self._log.append(entry)
        return R, mode
