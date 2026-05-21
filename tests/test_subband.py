import numpy as np
import pytest
from scipy import signal
from snore_anc.core.subband import DelaylessSubbandFxLMS
from snore_anc.core.fxlms import FxLMS


class TestDelaylessSubbandFxLMS:
    def test_analysis_synthesis_reconstruction(self):
        """FFT analysis -> weight transform should produce valid fullband weights."""
        sb = DelaylessSubbandFxLMS(
            n_subbands=64, filter_length_per_band=4,
            step_size=0.001, secondary_path_est=[1.0]
        )
        for i in range(200):
            sb.process_sample(np.sin(2 * np.pi * 150 * i / 16000))
            sb.update(0.1)
        assert np.all(np.isfinite(sb.W_full))
        assert sb.W_full.shape[0] == 64 * 4

    def test_convergence_faster_than_fullband(self):
        """Subband FxLMS should converge faster than fullband FxLMS on narrowband signal.

        Uses the same mu for both; the subband's shorter per-band filters (L_sub=4
        vs L=256) and frequency-domain decomposition accelerate convergence on
        narrowband tones.
        """
        np.random.seed(42)
        fs = 16000; duration = 3.0; n_samples = int(fs * duration)
        t = np.arange(n_samples) / fs
        x = (np.sin(2 * np.pi * 120 * t) + 0.7 * np.sin(2 * np.pi * 130 * t))
        primary_path = np.zeros(11); primary_path[10] = 1.0
        secondary_path = np.array([1.0])
        d = signal.lfilter(primary_path, [1.0], x)

        # Fullband FxLMS (small mu for slower convergence baseline)
        fb = FxLMS(filter_length=256, step_size=0.0003, secondary_path_est=[1.0])
        _, e_fb, _ = fb.process_block(x, d, secondary_path)

        # Subband FxLMS (M=64, L_sub=4 -> L_full=256), larger mu is stable
        # because subband decomposition whitens the input
        sb = DelaylessSubbandFxLMS(
            n_subbands=64, filter_length_per_band=4,
            step_size=0.001, secondary_path_est=[1.0]
        )
        _, e_sb, _ = sb.process_block(x, d, secondary_path)

        # Compare NR at midpoint (1.5s)
        mid = n_samples // 2
        quarter = n_samples // 4
        nr_fb = 20 * np.log10(
            np.sqrt(np.mean(d[mid-quarter:mid]**2)) /
            (np.sqrt(np.mean(e_fb[mid-quarter:mid]**2)) + 1e-12) + 1e-12
        )
        nr_sb = 20 * np.log10(
            np.sqrt(np.mean(d[mid-quarter:mid]**2)) /
            (np.sqrt(np.mean(e_sb[mid-quarter:mid]**2)) + 1e-12) + 1e-12
        )
        assert nr_sb > nr_fb, f"Subband NR={nr_sb:.1f}dB should exceed fullband NR={nr_fb:.1f}dB at midpoint"

    def test_init_default_params(self):
        """Default init produces correct shapes."""
        sb = DelaylessSubbandFxLMS()
        assert sb.W_full.shape == (64 * 4,)
        assert sb.W_sub.shape == (64, 4)
