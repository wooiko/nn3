"""Dynamic model: first-order lag with transport delay for each output.

Wraps the static model: y_static = static_model(u) → first-order lag → delay buffer → y.

State: internal lag state + circular delay buffer for each output channel.
Two outputs: [beta_fe, epsilon].  Two inputs: [B, Q, rho, alpha_fe] (passed through to static model).

All time constants and delays are in discrete steps (integers).
spec 2.2.2: aperiodic lag 1st–2nd order + transport delay.
We implement 1st-order here (configurable via tau_*); a 2nd-order variant
can be composed by chaining two DynamicModel instances if needed.
"""
import numpy as np
from object_model.static_model import predict_outputs


class DynamicModel:
    """Discrete first-order lag + transport delay for both outputs.

    The discrete-time transfer function for each channel:
        y_lag[k] = a * y_lag[k-1] + (1-a) * y_static[k]
        y[k]     = y_lag[k - delay]

    where a = exp(-1 / tau) (matched z-transform of continuous 1st-order lag).
    """

    N_OUTPUTS = 2  # [beta_fe, epsilon]

    def __init__(self, cfg: dict):
        self._cfg = cfg

        tau_bf = float(cfg["tau_beta_fe"])
        tau_ep = float(cfg["tau_epsilon"])
        self._a = np.array([
            np.exp(-1.0 / tau_bf) if tau_bf > 0 else 0.0,
            np.exp(-1.0 / tau_ep) if tau_ep > 0 else 0.0,
        ])

        self._delay = np.array([
            int(cfg["delay_beta_fe"]),
            int(cfg["delay_epsilon"]),
        ])
        max_delay = int(max(self._delay)) + 1

        # Circular delay buffers: shape (max_delay, N_OUTPUTS)
        self._buf = np.zeros((max_delay, self.N_OUTPUTS))
        self._buf_ptr = 0
        self._buf_len = max_delay

        # Lag state initialised to zero; will be warm-started on first step
        self._y_lag = np.zeros(self.N_OUTPUTS)
        self._initialised = False

    def reset(self, B: float, rho: float, Q: float, alpha_fe: float) -> None:
        """Warm-start all internal states from a steady-state operating point."""
        y_ss = np.array(predict_outputs(B, rho, Q, alpha_fe, self._cfg))
        self._y_lag[:] = y_ss
        self._buf[:] = y_ss
        self._buf_ptr = 0
        self._initialised = True

    def step(self, B: float, rho: float, Q: float, alpha_fe: float, disturbance: float = 0.0) -> np.ndarray:
        """Advance one discrete step and return [beta_fe, epsilon] after lag + delay.

        Args:
            B, rho, Q, alpha_fe: current control inputs and disturbance level.
            disturbance: additive disturbance on alpha_fe (from disturbances module).

        Returns:
            y: np.ndarray shape (2,) — [beta_fe, epsilon] with dynamics applied.
        """
        alpha_fe_total = np.clip(
            alpha_fe + disturbance,
            self._cfg["alpha_fe_min_dom"],
            self._cfg["alpha_fe_max_dom"],
        )

        y_static = np.array(predict_outputs(B, rho, Q, alpha_fe_total, self._cfg))

        if not self._initialised:
            self._y_lag[:] = y_static
            self._buf[:] = y_static
            self._initialised = True

        # First-order lag update
        self._y_lag = self._a * self._y_lag + (1.0 - self._a) * y_static

        # Write into circular buffer
        self._buf[self._buf_ptr, :] = self._y_lag

        # Read delayed values
        y_out = np.empty(self.N_OUTPUTS)
        for i in range(self.N_OUTPUTS):
            read_ptr = (self._buf_ptr - self._delay[i]) % self._buf_len
            y_out[i] = self._buf[read_ptr, i]

        self._buf_ptr = (self._buf_ptr + 1) % self._buf_len
        return y_out
