import numpy as np


class SecondaryPathEstimator:
    """Estimates secondary path S(z) using LMS system identification."""

    def __init__(self, order=128, step_size=0.01):
        self.order = order
        self.mu = step_size
        self.w = np.zeros(order)
        self._buf = np.zeros(order)

    def offline_estimate(self, probe_signal, measured_response, n_iterations=3):
        """Estimate S(z) from probe signal and measured response.

        Uses standard LMS to minimize |measured - probe * w|^2.

        Args:
            probe_signal: excitation signal (N,)
            measured_response: system output (N,)
            n_iterations: number of passes over the data

        Returns:
            Estimated FIR coefficients (order,)
        """
        self.w = np.zeros(self.order)
        self._buf = np.zeros(self.order)
        N = len(probe_signal)
        for _ in range(n_iterations):
            for n in range(N):
                self._buf = np.roll(self._buf, 1)
                self._buf[0] = probe_signal[n]
                y_hat = float(np.dot(self.w, self._buf))
                e = measured_response[n] - y_hat
                self.w += self.mu * e * self._buf
        return self.w.copy()

    def online_update(self, x_n, y_n, e_n):
        """Single-step online update for secondary path estimation."""
        self._buf = np.roll(self._buf, 1)
        self._buf[0] = x_n
        self.w += self.mu * e_n * self._buf
