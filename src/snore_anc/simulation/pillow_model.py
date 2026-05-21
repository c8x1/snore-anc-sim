import numpy as np
from scipy import signal as sig

from snore_anc.config import ANCConfig
from snore_anc.simulation.acoustic_path import generate_primary_path, generate_secondary_path
from snore_anc.simulation.snoring_source import SnoringSynthesizer


class PillowScenario:
    """Full pillow ANC simulation scenario."""

    def __init__(self, config=None):
        self.cfg = config or ANCConfig()

    def build_paths(self):
        """Generate primary and secondary paths from config geometry.

        Returns:
            (primary_path, secondary_path) tuple of FIR coefficient arrays
        """
        primary = generate_primary_path(
            fs=self.cfg.fs,
            distance=self.cfg.husband_wife_distance,
            room_dims=self.cfg.room_dims,
        )
        secondary = generate_secondary_path(
            fs=self.cfg.fs,
            distance=self.cfg.pillow_speaker_mic_distance,
        )
        return primary, secondary

    def generate_snoring(self, duration_s=30.0, pattern='regular'):
        """Generate a snoring source signal. Returns (signal, fs)."""
        synth = SnoringSynthesizer(
            fs=self.cfg.fs,
            f0=self.cfg.snore_f0,
            f0_variation=self.cfg.snore_f0_variation,
        )
        return synth.generate(duration_s=duration_s, pattern=pattern)

    def run_simulation(self, source_signal, anc_algorithm, secondary_path=None):
        """Run full ANC simulation on a source signal.

        Args:
            source_signal: input snoring signal (N,)
            anc_algorithm: object with process_block(x, d, S) method
            secondary_path: optional override for S(z)

        Returns:
            dict with keys: y, e, d, fs, primary_path, secondary_path
        """
        primary_path, sec_path = self.build_paths()
        if secondary_path is not None:
            sec_path = np.array(secondary_path, dtype=float)

        d = sig.lfilter(primary_path, [1.0], source_signal)
        y, e, _ = anc_algorithm.process_block(source_signal, d, sec_path)

        return {
            'y': y, 'e': e, 'd': d, 'fs': self.cfg.fs,
            'primary_path': primary_path, 'secondary_path': sec_path,
        }
