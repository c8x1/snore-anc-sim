"""Unit tests for the FxLMS adaptive filter."""

import numpy as np
import scipy.signal
import pytest

from snore_anc.core.fxlms import FxLMS


class TestFxLMS:
    def test_single_tone_cancellation(self):
        """150Hz sine through pure-delay path: NR > 20dB in last 0.5s."""
        fs = 16000
        duration = 5.0
        n_samples = int(fs * duration)
        freq = 150.0
        t = np.arange(n_samples) / fs
        x = np.sin(2 * np.pi * freq * t)

        primary_path = np.zeros(11)
        primary_path[10] = 1.0
        secondary_path = np.array([1.0])
        s_hat = np.array([1.0])

        d = scipy.signal.lfilter(primary_path, [1.0], x)

        fxlms = FxLMS(filter_length=128, step_size=0.005, secondary_path_est=s_hat)
        _, e, _ = fxlms.process_block(x, d, secondary_path)

        check_start = int((duration - 0.5) * fs)
        d_tail = d[check_start:]
        e_tail = e[check_start:]
        nr_db = 20 * np.log10(
            np.sqrt(np.mean(d_tail**2)) / (np.sqrt(np.mean(e_tail**2)) + 1e-12)
        )
        assert nr_db > 20.0

    def test_weights_converge_for_known_path(self):
        """White noise through known path: W converges to -P."""
        np.random.seed(42)
        n_samples = 80000
        x = np.random.randn(n_samples) * 0.5

        primary_path = np.zeros(5)
        primary_path[3] = 1.0
        d = scipy.signal.lfilter(primary_path, [1.0], x)

        fxlms = FxLMS(filter_length=16, step_size=0.01, secondary_path_est=[1.0])
        fxlms.process_block(x, d, np.array([1.0]))

        expected = np.zeros(16)
        expected[3] = -1.0
        error_db = 20 * np.log10(
            np.linalg.norm(fxlms.w - expected) / (np.linalg.norm(expected) + 1e-12)
            + 1e-12
        )
        assert error_db < -20

    def test_init_default_params(self):
        """Default init: L=256, mu=0.01, w=zeros."""
        fxlms = FxLMS()
        assert len(fxlms.w) == 256
        assert fxlms.mu == 0.01
        assert np.all(fxlms.w == 0)

    def test_process_sample_returns_float(self):
        """process_sample returns float."""
        fxlms = FxLMS(filter_length=8, step_size=0.01, secondary_path_est=[1.0])
        y = fxlms.process_sample(0.5)
        assert isinstance(y, float)

    def test_error_decreases_over_time(self):
        """Running average error should decrease."""
        fs = 16000
        n_samples = 32000
        t = np.arange(n_samples) / fs
        x = np.sin(2 * np.pi * 100 * t)

        primary_path = np.array([0.0, 0.0, 0.0, 1.0])
        d = scipy.signal.lfilter(primary_path, [1.0], x)

        fxlms = FxLMS(filter_length=64, step_size=0.005, secondary_path_est=[1.0])
        _, e, _ = fxlms.process_block(x, d, np.array([1.0]))

        q = n_samples // 4
        rms_first = np.sqrt(np.mean(e[:q] ** 2))
        rms_last = np.sqrt(np.mean(e[3 * q :] ** 2))
        assert rms_last < rms_first
