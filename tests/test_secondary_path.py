import numpy as np
import pytest
from snore_anc.core.secondary_path import SecondaryPathEstimator

class TestSecondaryPathEstimator:
    def test_offline_estimation_known_fir(self):
        """Offline LMS should recover a known FIR path with error < -30dB."""
        np.random.seed(42)
        true_path = np.array([0.1, 0.3, 0.5, 0.3, 0.1])
        n_samples = 8000
        probe = np.random.randn(n_samples) * 0.5
        response = np.convolve(probe, true_path, mode='full')[:n_samples]
        estimator = SecondaryPathEstimator(order=10, step_size=0.01)
        s_hat = estimator.offline_estimate(probe, response)
        true_padded = np.zeros(10)
        true_padded[:len(true_path)] = true_path
        error_db = 20 * np.log10(
            np.linalg.norm(s_hat - true_padded) / (np.linalg.norm(true_padded) + 1e-12) + 1e-12
        )
        assert error_db < -30, f"Estimation error = {error_db:.1f} dB, expected < -30 dB"

    def test_offline_estimation_produces_correct_length(self):
        """Estimated path should have the requested order."""
        estimator = SecondaryPathEstimator(order=64)
        probe = np.random.randn(1000)
        response = np.convolve(probe, [1.0, 0.5, 0.1], mode='full')[:1000]
        s_hat = estimator.offline_estimate(probe, response)
        assert len(s_hat) == 64
