import numpy as np
import pytest
from scipy import signal
from snore_anc.core.multichannel import MultiChannelANC

class TestMultiChannelANC:
    def test_dual_zone_noise_reduction(self):
        """2-spk x 2-err system should achieve > 10dB NR in both zones."""
        fs = 16000; duration = 5.0; n_samples = int(fs * duration)
        t = np.arange(n_samples) / fs
        x = np.sin(2 * np.pi * 150 * t)

        # Primary paths to each error mic (different delays)
        p1 = np.zeros(11); p1[10] = 1.0
        p2 = np.zeros(13); p2[12] = 0.9
        d1 = signal.lfilter(p1, [1.0], x)
        d2 = signal.lfilter(p2, [1.0], x)

        # Secondary path matrix (2 err x 2 spk)
        s_matrix = [
            [np.array([0.0, 1.0, 0.0]),   np.array([0.0, 0.0, 0.3])],
            [np.array([0.0, 0.3, 0.0]),   np.array([0.0, 1.0, 0.0])],
        ]

        mc = MultiChannelANC(
            n_speakers=2, n_errors=2,
            filter_length=64, step_size=0.001,
            secondary_path_matrix=s_matrix
        )
        result = mc.process_block(x, [d1, d2], s_matrix)

        check_start = int((duration - 0.5) * fs)
        for ch in range(2):
            d_tail = np.array(result['d'][ch][check_start:])
            e_tail = np.array(result['e'][ch][check_start:])
            rms_d = np.sqrt(np.mean(d_tail ** 2))
            rms_e = np.sqrt(np.mean(e_tail ** 2))
            nr = 20 * np.log10(rms_d / (rms_e + 1e-12) + 1e-12)
            assert nr > 10, f"Channel {ch}: NR = {nr:.1f} dB, expected > 10 dB"

    def test_init_shapes(self):
        """Multi-channel init should create correct filter shapes."""
        mc = MultiChannelANC(n_speakers=2, n_errors=2, filter_length=32)
        assert len(mc.filters) == 2
        assert all(len(f.w) == 32 for f in mc.filters)
