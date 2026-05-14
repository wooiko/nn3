"""Disturbance generators for αFe feed grade (spec 2.2.4).

All generators return an additive offset on αFe at discrete step t.
Combine via: alpha_fe_disturbed = alpha_fe_nominal + drift(t, cfg) + step_change(t, cfg) + impulse_spike(t, cfg)
Measurement noise is applied separately after model evaluation.
"""
import numpy as np


def drift(t: int, cfg: dict) -> float:
    """Linear drift of αFe, bounded by drift_amplitude."""
    raw = cfg["drift_rate"] * t
    return float(np.clip(raw, -cfg["drift_amplitude"], cfg["drift_amplitude"]))


def step_change(t: int, cfg: dict) -> float:
    """Single step change in αFe at step_time."""
    return float(cfg["step_magnitude"]) if t >= cfg["step_time"] else 0.0


def impulse_spike(t: int, cfg: dict) -> float:
    """Rectangular impulse spikes of impulse_duration steps at each impulse_times entry."""
    impulse_times = cfg.get("impulse_times", [])
    for t_start in impulse_times:
        if t_start <= t < t_start + cfg["impulse_duration"]:
            return float(cfg["impulse_magnitude"])
    return 0.0


def measurement_noise(shape: tuple, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Additive Gaussian measurement noise for [beta_fe, epsilon].

    Args:
        shape: output shape, typically (2,) or (N, 2).
        rng:   seeded numpy Generator for reproducibility.
        cfg:   config dict with noise_std_beta_fe and noise_std_epsilon.

    Returns:
        noise array matching shape; last dim must be 2 (channels: beta_fe, epsilon).
    """
    std = np.array([cfg["noise_std_beta_fe"], cfg["noise_std_epsilon"]])
    noise = rng.standard_normal(shape)
    # broadcast std over all but last dimension
    noise = noise * std
    return noise
