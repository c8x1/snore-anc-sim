"""Standard FxLMS (Filtered-x Least Mean Squares) adaptive filter for ANC.

Signal flow::

    x(n) -> [W(z)] -> y(n) -> [S(z)] -> speaker output at error mic
    e(n) = d(n) + (S * y)(n)   [residual error]

Weight update::

    W(n+1) = W(n) - mu * e(n) * X'(n)
    where X'(n) = S_hat(z) * X(n) is the filtered reference vector.
"""

from __future__ import annotations

import numpy as np


class FxLMS:
    """Filtered-x LMS adaptive filter for active noise control.

    Parameters
    ----------
    filter_length : int
        Number of adaptive filter taps (L).
    step_size : float
        Convergence step size (mu).
    secondary_path_est : array_like or None
        FIR coefficients of the secondary path estimate S_hat(z).
        Defaults to ``[1.0]`` (unity passthrough).
    """

    def __init__(
        self,
        filter_length: int = 256,
        step_size: float = 0.01,
        secondary_path_est: list | np.ndarray | None = None,
    ) -> None:
        self.L = filter_length
        self.mu = step_size
        self.w = np.zeros(self.L)

        # Secondary path estimate S_hat(z)
        if secondary_path_est is not None:
            self.s_hat = np.array(secondary_path_est, dtype=float)
        else:
            self.s_hat = np.array([1.0])
        self.s_hat_len = len(self.s_hat)

        # Circular-ish buffers (rolled each sample)
        self._x_buf = np.zeros(self.L)   # reference history
        self._xf_buf = np.zeros(self.L)  # filtered-x history

    def process_sample(self, x_n: float) -> float:
        """Generate anti-noise sample y(n) = W^T . X(n).

        Parameters
        ----------
        x_n : float
            Current reference signal sample.

        Returns
        -------
        float
            Anti-noise output y(n).
        """
        self._x_buf = np.roll(self._x_buf, 1)
        self._x_buf[0] = x_n
        return float(np.dot(self.w, self._x_buf))

    def update(self, e_n: float) -> None:
        """Update adaptive filter weights.

        W(n+1) = W(n) - mu * e(n) * X'(n)

        Parameters
        ----------
        e_n : float
            Current error signal sample.
        """
        # Filtered-x: x'(n) = S_hat^T . [x(n), x(n-1), ..., x(n-M+1)]
        xf_n = float(np.dot(self.s_hat, self._x_buf[: self.s_hat_len]))
        self._xf_buf = np.roll(self._xf_buf, 1)
        self._xf_buf[0] = xf_n
        self.w -= self.mu * e_n * self._xf_buf

    def process_block(
        self,
        x_block: np.ndarray,
        d_block: np.ndarray,
        secondary_path: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run full ANC simulation on a block of samples.

        Parameters
        ----------
        x_block : ndarray, shape (N,)
            Reference signal.
        d_block : ndarray, shape (N,)
            Desired signal -- primary path output at error mic.
        secondary_path : ndarray
            FIR coefficients for S(z) simulation.

        Returns
        -------
        y_out : ndarray, shape (N,)
            Anti-noise signal.
        e_out : ndarray, shape (N,)
            Residual error.
        d_out : ndarray, shape (N,)
            Copy of *d_block*.
        """
        N = len(x_block)
        y_out = np.zeros(N)
        e_out = np.zeros(N)
        sp_buf = np.zeros(len(secondary_path))

        for n in range(N):
            # 1. Generate anti-noise
            y_n = self.process_sample(x_block[n])
            y_out[n] = y_n

            # 2. Simulate secondary path: y_s(n) = S^T . [y(n), y(n-1), ...]
            sp_buf = np.roll(sp_buf, 1)
            sp_buf[0] = y_n
            y_s = float(np.dot(secondary_path, sp_buf))

            # 3. Error: e(n) = d(n) + y_s(n)
            e_n = d_block[n] + y_s
            e_out[n] = e_n

            # 4. Update weights
            self.update(e_n)

        return y_out, e_out, d_block.copy()
