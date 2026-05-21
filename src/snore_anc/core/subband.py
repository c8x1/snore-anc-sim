"""Delayless Subband FxLMS using Morgan & Thi (1995) weight transformation.

Architecture
------------
1. Analysis: Partition fullband filtered-x into L_sub blocks of M samples,
   FFT each block -> M subband vectors of length L_sub
2. Per-subband: Short LMS filters (length L_sub) with independent updates
3. Weight transform: Stack subband weights, IFFT each tap -> fullband weights
4. Fullband output: y(n) = W_full^T . X(n) -- no synthesis filterbank delay

The Morgan-Thi delayless architecture transforms subband adaptive weights
to fullband weights after each update, avoiding synthesis filterbank delay.
Each subband sees decorrelated (whitened) signal content, which reduces
eigenvalue spread and accelerates convergence compared to fullband FxLMS.

Reference
---------
Morgan, D.R. & Thi, J.C. (1995).
"A delayless subband adaptive filter architecture."
IEEE Transactions on Signal Processing.
"""

from __future__ import annotations

import numpy as np


class DelaylessSubbandFxLMS:
    """Delayless subband adaptive filter using DFT filterbank + Morgan-Thi weight transform.

    Parameters
    ----------
    n_subbands : int
        Number of subbands (M). Must be a power of 2 for efficient FFT.
    filter_length_per_band : int
        Number of adaptive taps per subband (L_sub).
        Total fullband filter length = M * L_sub.
    step_size : float
        Convergence step size (mu) for the per-subband LMS updates.
        Should be scaled by 1/M relative to fullband step size due to
        the IDFT normalization in the weight transform.
    secondary_path_est : array_like or None
        FIR coefficients of the secondary path estimate S_hat(z).
        Defaults to ``[1.0]`` (unity passthrough).
    """

    def __init__(
        self,
        n_subbands: int = 64,
        filter_length_per_band: int = 4,
        step_size: float = 0.01,
        secondary_path_est: list | np.ndarray | None = None,
    ) -> None:
        self.M = n_subbands
        self.L_sub = filter_length_per_band
        self.L_full = self.M * self.L_sub
        self.mu = step_size

        # Subband adaptive weight matrix: M subbands x L_sub taps
        self.W_sub = np.zeros((self.M, self.L_sub))

        # Fullband weights (transformed from subband via IDFT)
        self.W_full = np.zeros(self.L_full)

        # Fullband reference buffer
        self._x_buf = np.zeros(self.L_full)

        # Fullband filtered-x buffer
        self._xf_buf = np.zeros(self.L_full)

        # Secondary path estimate
        if secondary_path_est is not None:
            self._s_hat = np.array(secondary_path_est, dtype=float)
        else:
            self._s_hat = np.array([1.0])
        self._s_hat_len = len(self._s_hat)

    def _get_subband_reference(self) -> np.ndarray:
        """Decompose fullband filtered-x into subband reference vectors.

        Partitions _xf_buf into L_sub blocks of M samples, FFTs each block,
        and collects the k-th element from each FFT to form the reference
        vector for subband k.

        Returns
        -------
        X_sub : ndarray, shape (M, L_sub)
            Subband reference matrix. X_sub[k, :] is the reference vector
            for subband k, containing L_sub complex FFT-bin values.
        """
        # Reshape into L_sub blocks of M samples
        # _xf_buf[0] is newest, _xf_buf[M-1] is oldest in first block
        X_blocks = self._xf_buf[: self.L_full].reshape(self.L_sub, self.M)

        # FFT each block (axis=1): shape (L_sub, M)
        X_fft = np.fft.fft(X_blocks, n=self.M, axis=1)

        # Transpose to (M, L_sub): X_sub[k, j] = FFT of block j at bin k
        return X_fft.T

    def _transform_weights(self) -> None:
        """Transform subband weights -> fullband weights via IDFT.

        For each tap position j (0..L_sub-1), applies inverse FFT across
        the M subband weights, then interleaves the result into a single
        fullband weight vector.
        """
        # W_time shape: (M, L_sub) -- IDFT along subband axis (axis=0)
        W_time = np.fft.ifft(self.W_sub, n=self.M, axis=0).real
        # Interleave: W_full[j*M + m] = W_time[m, j]
        self.W_full = W_time.T.flatten()

    def process_sample(self, x_n: float) -> float:
        """Generate anti-noise sample y(n) = W_full^T . X(n).

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
        return float(np.dot(self.W_full, self._x_buf))

    def update(self, e_n: float) -> None:
        """Update subband filters based on error, then transform to fullband.

        Parameters
        ----------
        e_n : float
            Current error signal sample.
        """
        # 1. Compute filtered-x in fullband
        xf_n = float(np.dot(self._s_hat, self._x_buf[: self._s_hat_len]))
        self._xf_buf = np.roll(self._xf_buf, 1)
        self._xf_buf[0] = xf_n

        # 2. Subband decomposition: partition filtered-x into blocks and FFT
        X_sub = self._get_subband_reference()  # (M, L_sub)

        # 3. Per-subband LMS update: W_sub[k] -= mu * e * Re{X_sub[k]}
        # Using real part ensures Hermitian symmetry (real fullband weights)
        self.W_sub -= self.mu * e_n * X_sub.real

        # 4. Transform subband weights to fullband (delayless)
        self._transform_weights()

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
            y_n = self.process_sample(x_block[n])
            y_out[n] = y_n

            sp_buf = np.roll(sp_buf, 1)
            sp_buf[0] = y_n
            y_s = float(np.dot(secondary_path, sp_buf))

            e_n = d_block[n] + y_s
            e_out[n] = e_n

            self.update(e_n)

        return y_out, e_out, d_block.copy()
