import os
import numpy as np

class SnoringSynthesizer:
    """Synthetic snoring signal generator.
    Generates realistic snoring waveforms with:
    - Fundamental frequency (100-200Hz) with vibrato
    - Harmonics decaying as 1/n
    - Bandpass-colored noise (50-500Hz)
    - Periodic ON/OFF envelope (inhale/exhale cycle)
    """
    def __init__(self, fs=16000, f0=150.0, f0_variation=5.0,
                 inhale_duration=0.4, exhale_duration=0.3):
        self.fs = fs
        self.f0 = f0
        self.f0_var = f0_variation
        self.inhale_dur = inhale_duration
        self.exhale_dur = exhale_duration

    def generate(self, duration_s=30.0, pattern='regular'):
        """Generate synthetic snoring signal.
        Args:
            duration_s: total duration in seconds
            pattern: 'regular' | 'irregular' | 'apnea'
        Returns:
            (signal, fs) tuple
        """
        N = int(duration_s * self.fs)
        sig = np.zeros(N)
        cycle_duration = self.inhale_dur + self.exhale_dur
        samples_per_cycle = int(cycle_duration * self.fs)

        for i in range(N):
            t = i / self.fs
            cycle_pos = (i % samples_per_cycle) / self.fs
            if cycle_pos < self.inhale_dur:
                phase = cycle_pos / self.inhale_dur
                envelope = np.sin(np.pi * phase)
            else:
                if pattern == 'regular':
                    envelope = 0.0
                elif pattern == 'irregular':
                    envelope = 0.1 * np.random.random()
                else:
                    envelope = 0.0

            vibrato = self.f0_var * np.sin(2 * np.pi * 3.0 * t)
            f0_inst = self.f0 + vibrato
            theta = 2 * np.pi * f0_inst * t
            fundamental = np.sin(theta)
            harmonics = sum((1.0 / n) * np.sin(n * theta) for n in range(2, 9))
            noise = np.random.randn() * 0.15
            sig[i] = envelope * (0.5 * fundamental + 0.3 * harmonics + noise)

        peak = np.max(np.abs(sig))
        if peak > 0:
            sig = sig / peak * 0.8
        return sig, self.fs

def load_dataset(name, segment=None, data_dir=None):
    """Load a public snoring dataset.
    Supports: 'synthetic' (auto-generated), 'mpssc' (local files required).
    Returns: (signal, fs, annotation) tuple
    """
    if name == 'synthetic':
        synth = SnoringSynthesizer()
        sig, fs = synth.generate(duration_s=30.0, pattern='regular')
        return sig, fs, {'pattern': 'regular', 'source': 'synthetic'}
    if name == 'mpssc':
        if data_dir is None:
            raise FileNotFoundError("MPSSC dataset requires local files and data_dir parameter.")
        import soundfile as sf
        wav_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.wav')])
        if not wav_files:
            raise FileNotFoundError(f"No .wav files in {data_dir}")
        idx = segment if segment is not None else 0
        filepath = os.path.join(data_dir, wav_files[idx])
        sig, fs = sf.read(filepath)
        if sig.ndim > 1:
            sig = sig[:, 0]
        return sig, fs, {'file': wav_files[idx], 'source': 'mpssc'}
    raise ValueError(f"Unknown dataset: {name}")
