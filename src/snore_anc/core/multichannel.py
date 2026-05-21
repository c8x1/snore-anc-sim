import numpy as np
from snore_anc.core.fxlms import FxLMS

class MultiChannelANC:
    """Multi-channel ANC: 1 reference x N speakers x M error microphones.

    Architecture:
    - N_speakers independent FxLMS filters, one per speaker
    - Each filter's update incorporates errors from ALL error mics
    - Secondary path is an N_err x N_spk matrix of FIR paths

    Update rule:
        DW_j = mu * SUM_i (e_i * x'_ij)
    where x'_ij = S_hat_ij * x(n) is the filtered-x for speaker j at error mic i.
    """

    def __init__(self, n_speakers=2, n_errors=2, filter_length=256,
                 step_size=0.01, secondary_path_matrix=None):
        self.N_spk = n_speakers
        self.N_err = n_errors
        self.L = filter_length
        self.mu = step_size

        # One FxLMS filter per speaker (we manually control updates)
        self.filters = [
            FxLMS(filter_length=filter_length, step_size=step_size,
                  secondary_path_est=None)
            for _ in range(n_speakers)
        ]

        # Secondary path matrix: N_err x N_spk, each element is FIR coefficients
        if secondary_path_matrix is not None:
            self.s_matrix = secondary_path_matrix
        else:
            self.s_matrix = [
                [np.array([1.0]) if i == j else np.array([0.0])
                 for j in range(n_speakers)]
                for i in range(n_errors)
            ]

        # Secondary path simulation buffers: N_err x N_spk
        self._sp_sim_bufs = [
            [np.zeros(len(self.s_matrix[i][j])) for j in range(n_speakers)]
            for i in range(n_errors)
        ]

    def process_block(self, x_block, d_blocks, s_matrix=None):
        """Run multi-channel ANC simulation.
        Args:
            x_block: reference signal (N,)
            d_blocks: list of desired signals per error mic [d_0, d_1, ...]
            s_matrix: optional override secondary path matrix
        Returns:
            dict with 'y' (list per speaker), 'e' (list per error mic),
            'd' (list per error mic)
        """
        if s_matrix is not None:
            self.s_matrix = s_matrix
            self._sp_sim_bufs = [
                [np.zeros(len(s_matrix[i][j])) for j in range(self.N_spk)]
                for i in range(self.N_err)
            ]

        N = len(x_block)
        y_outs = [np.zeros(N) for _ in range(self.N_spk)]
        e_outs = [np.zeros(N) for _ in range(self.N_err)]

        for n in range(N):
            x_n = x_block[n]

            # 1. Each speaker generates anti-noise
            y_ns = []
            for j in range(self.N_spk):
                y_n = self.filters[j].process_sample(x_n)
                y_ns.append(y_n)
                y_outs[j][n] = y_n

            # 2. Simulate secondary paths to each error mic
            y_s = np.zeros(self.N_err)
            for i in range(self.N_err):
                for j in range(self.N_spk):
                    buf = self._sp_sim_bufs[i][j]
                    buf = np.roll(buf, 1)
                    buf[0] = y_ns[j]
                    self._sp_sim_bufs[i][j] = buf
                    y_s[i] += float(np.dot(self.s_matrix[i][j], buf))

            # 3. Compute errors
            e_ns = []
            for i in range(self.N_err):
                e_n = d_blocks[i][n] + y_s[i]
                e_outs[i][n] = e_n
                e_ns.append(e_n)

            # 4. Update each speaker's filter using accumulated gradient
            for j in range(self.N_spk):
                grad_sum = 0.0
                for i in range(self.N_err):
                    s_hat_ij = self.s_matrix[i][j]  # assume perfect estimate
                    xf_n = float(np.dot(s_hat_ij, self.filters[j]._x_buf[:len(s_hat_ij)]))
                    grad_sum += e_ns[i] * xf_n

                self.filters[j].w -= self.mu * grad_sum * self.filters[j]._x_buf

        return {
            'y': y_outs, 'e': e_outs,
            'd': [d.copy() for d in d_blocks],
        }
