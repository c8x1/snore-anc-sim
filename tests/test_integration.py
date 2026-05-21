"""Integration tests: full pipeline from snoring source through ANC to evaluation."""
import numpy as np
import pytest
from scipy import signal

from snore_anc.config import ANCConfig
from snore_anc.core.fxlms import FxLMS
from snore_anc.core.subband import DelaylessSubbandFxLMS
from snore_anc.simulation.pillow_model import PillowScenario
from snore_anc.evaluation.metrics import ANCEvaluator


class TestFullPipeline:

    def test_fxlms_with_synthetic_snoring(self):
        """Full pipeline: synthetic snoring -> FxLMS -> NR > 8dB."""
        cfg = ANCConfig(fs=16000)
        scenario = PillowScenario(cfg)
        sig, fs = scenario.generate_snoring(duration_s=5.0)

        # Use simple paths for stable convergence
        primary_path = np.zeros(11)
        primary_path[10] = 1.0
        secondary_path = np.array([1.0])

        from scipy.signal import lfilter
        d = lfilter(primary_path, [1.0], sig)

        fxlms = FxLMS(filter_length=256, step_size=0.005,
                       secondary_path_est=secondary_path)
        y, e, _ = fxlms.process_block(sig, d, secondary_path)

        eval_check = int(4.0 * fs)  # check last 1 second
        nr = ANCEvaluator.noise_reduction_db(
            d[eval_check:], e[eval_check:]
        )
        assert nr > 8, f"Integration NR = {nr:.1f} dB, expected > 8 dB"

    def test_subband_with_synthetic_snoring(self):
        """Full pipeline: synthetic snoring -> Subband FxLMS -> converges."""
        cfg = ANCConfig(fs=16000)
        scenario = PillowScenario(cfg)
        sig, fs = scenario.generate_snoring(duration_s=5.0)

        primary_path = np.zeros(11)
        primary_path[10] = 1.0
        secondary_path = np.array([1.0])

        from scipy.signal import lfilter
        d = lfilter(primary_path, [1.0], sig)

        sb = DelaylessSubbandFxLMS(
            n_subbands=64, filter_length_per_band=4,
            step_size=0.005, secondary_path_est=secondary_path
        )
        y, e, _ = sb.process_block(sig, d, secondary_path)

        eval_check = int(4.0 * fs)
        nr = ANCEvaluator.noise_reduction_db(
            d[eval_check:], e[eval_check:]
        )
        assert nr > 0, f"Subband NR = {nr:.1f} dB, expected > 0 dB"
