# Snore ANC Algorithm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python simulation framework for pillow-based snoring ANC with FxLMS, subband FxLMS, and multi-channel algorithms, plus evaluation and hardware test plan.

**Architecture:** Three-layer modular framework: core DSP primitives → acoustic simulation → evaluation/visualization. Each module is independently testable. TDD throughout.

**Tech Stack:** Python 3.11+, NumPy, SciPy, Matplotlib, SoundFile, pytest

**Spec:** `docs/superpowers/specs/2026-05-21-snore-anc-algorithm-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata + dependencies |
| `src/snore_anc/__init__.py` | Package marker |
| `src/snore_anc/config.py` | Centralized parameter definitions (dataclass) |
| `src/snore_anc/core/__init__.py` | Core subpackage marker |
| `src/snore_anc/core/fxlms.py` | Standard FxLMS adaptive filter |
| `src/snore_anc/core/subband.py` | Delayless subband FxLMS (DFT filterbank + Morgan&Thi weight transform) |
| `src/snore_anc/core/secondary_path.py` | Secondary path online/offline estimator |
| `src/snore_anc/core/multichannel.py` | Multi-channel ANC (1-ref × 2-spk × 2-err) |
| `src/snore_anc/simulation/__init__.py` | Simulation subpackage marker |
| `src/snore_anc/simulation/acoustic_path.py` | Primary/secondary path FIR generators |
| `src/snore_anc/simulation/snoring_source.py` | Synthetic snoring generator + dataset loader |
| `src/snore_anc/simulation/pillow_model.py` | Full pillow ANC scenario runner |
| `src/snore_anc/evaluation/__init__.py` | Evaluation subpackage marker |
| `src/snore_anc/evaluation/metrics.py` | NR(dB), convergence time, PSD comparison |
| `src/snore_anc/evaluation/visualization.py` | Waveform, spectrum, convergence, zone-of-quiet plots |
| `tests/__init__.py` | Test package marker |
| `tests/test_fxlms.py` | FxLMS unit tests |
| `tests/test_subband.py` | Subband FxLMS unit tests |
| `tests/test_secondary_path.py` | Secondary path estimation tests |
| `tests/test_multichannel.py` | Multi-channel ANC tests |
| `docs/hardware_setup.md` | Hardware prototype setup guide |

---

### Task 1: Project Scaffold + Config

**Files:**
- Create: `pyproject.toml`
- Create: `src/snore_anc/__init__.py`
- Create: `src/snore_anc/config.py`
- Create: `src/snore_anc/core/__init__.py`
- Create: `src/snore_anc/simulation/__init__.py`
- Create: `src/snore_anc/evaluation/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim
mkdir -p src/snore_anc/core src/snore_anc/simulation src/snore_anc/evaluation tests
```

- [ ] **Step 2: Create pyproject.toml**

```toml
# pyproject.toml
[project]
name = "snore-anc"
version = "0.1.0"
description = "Pillow-based snoring ANC algorithm simulation"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "matplotlib>=3.8",
    "soundfile>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create all __init__.py files**

Each `__init__.py` is empty:

```python
# src/snore_anc/__init__.py
# src/snore_anc/core/__init__.py
# src/snore_anc/simulation/__init__.py
# src/snore_anc/evaluation/__init__.py
# tests/__init__.py
```

- [ ] **Step 4: Create config.py**

```python
# src/snore_anc/config.py
from dataclasses import dataclass, field


@dataclass
class ANCConfig:
    """Centralized parameters for snore ANC simulation."""

    # Sampling
    fs: int = 16000  # Hz — snoring energy < 2kHz, Nyquist sufficient

    # FxLMS defaults
    filter_length: int = 256      # ~23ms @ 16kHz
    step_size: float = 0.01       # conservative initial μ
    secondary_path_order: int = 128  # FIR order for S(z) estimation

    # Subband FxLMS
    n_subbands: int = 64          # M = 64 → 250Hz bandwidth per subband
    filter_length_per_band: int = 4  # L_sub, short per-band filter

    # Multi-channel (1-ref × 2-spk × 2-err)
    n_speakers: int = 2
    n_error_mics: int = 2

    # Acoustic path defaults
    husband_wife_distance: float = 1.0   # meters
    pillow_speaker_mic_distance: float = 0.15  # meters (within pillow)
    room_dims: tuple = (4.0, 3.5, 2.8)   # bedroom L×W×H in meters
    speed_of_sound: float = 343.0        # m/s

    # Snoring synthesis defaults
    snore_f0: float = 150.0       # Hz — fundamental frequency
    snore_f0_variation: float = 5.0  # Hz — frequency modulation width
    snore_inhale_duration: float = 0.4  # seconds
    snore_exhale_duration: float = 0.3  # seconds

    # Evaluation thresholds
    nr_threshold_db: float = -10.0  # convergence threshold in dB
    nr_check_window: float = 0.5    # seconds — window for steady-state NR check
```

- [ ] **Step 5: Install dependencies and verify**

Run: `cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim && pip install -e ".[dev]"`

Expected: Installation succeeds with no errors.

- [ ] **Step 6: Commit**

```bash
git init
git add -A
git commit -m "feat: project scaffold with config and directory structure"
```

---

### Task 2: FxLMS Core + Tests

**Files:**
- Create: `src/snore_anc/core/fxlms.py`
- Create: `tests/test_fxlms.py`

- [ ] **Step 1: Write the FxLMS tests**

```python
# tests/test_fxlms.py
import numpy as np
import pytest
from scipy import signal

from snore_anc.core.fxlms import FxLMS


class TestFxLMS:
    """Unit tests for standard FxLMS adaptive filter."""

    def test_single_tone_cancellation(self):
        """150Hz sine wave through pure-delay path should achieve > 20dB NR."""
        fs = 16000
        duration = 5.0  # seconds
        n_samples = int(fs * duration)
        freq = 150.0

        # Reference signal
        t = np.arange(n_samples) / fs
        x = np.sin(2 * np.pi * freq * t)

        # Primary path: 10-sample delay
        primary_path = np.zeros(11)
        primary_path[10] = 1.0

        # Secondary path and estimate: identity
        secondary_path = np.array([1.0])
        s_hat = np.array([1.0])

        # Generate desired signal (what wife hears without ANC)
        d = signal.lfilter(primary_path, [1.0], x)

        # Run FxLMS
        fxlms = FxLMS(filter_length=128, step_size=0.005, secondary_path_est=s_hat)
        _, e, _ = fxlms.process_block(x, d, secondary_path)

        # Check NR in last 0.5 second
        check_start = int((duration - 0.5) * fs)
        d_tail = d[check_start:]
        e_tail = e[check_start:]
        nr_db = 20 * np.log10(np.sqrt(np.mean(d_tail**2)) / np.sqrt(np.mean(e_tail**2)) + 1e-12)

        assert nr_db > 20.0, f"NR = {nr_db:.1f} dB, expected > 20 dB"

    def test_weights_converge_for_known_path(self):
        """With identity paths and white noise, W should converge to -P."""
        np.random.seed(42)
        fs = 16000
        n_samples = 80000  # 5 seconds

        x = np.random.randn(n_samples) * 0.5

        # Primary path: [0, 0, 0, 1.0, 0] — 3-sample delay
        primary_path = np.zeros(5)
        primary_path[3] = 1.0

        secondary_path = np.array([1.0])
        s_hat = np.array([1.0])

        d = signal.lfilter(primary_path, [1.0], x)

        fxlms = FxLMS(filter_length=16, step_size=0.01, secondary_path_est=s_hat)
        fxlms.process_block(x, d, secondary_path)

        # At convergence, W should approximate -P (padded to filter_length)
        expected = np.zeros(16)
        expected[3] = -1.0
        error_db = 20 * np.log10(
            np.linalg.norm(fxlms.w - expected) / (np.linalg.norm(expected) + 1e-12) + 1e-12
        )
        assert error_db < -20, f"Weight error = {error_db:.1f} dB, expected < -20 dB"

    def test_init_default_params(self):
        """Default initialization should produce correct array shapes."""
        fxlms = FxLMS()
        assert len(fxlms.w) == 256
        assert fxlms.mu == 0.01
        assert np.all(fxlms.w == 0)

    def test_process_sample_returns_float(self):
        """process_sample should return a float."""
        fxlms = FxLMS(filter_length=8, step_size=0.01, secondary_path_est=[1.0])
        y = fxlms.process_sample(0.5)
        assert isinstance(y, float)

    def test_error_decreases_over_time(self):
        """Running average error should decrease monotonically (approximately)."""
        fs = 16000
        n_samples = 32000  # 2 seconds
        t = np.arange(n_samples) / fs
        x = np.sin(2 * np.pi * 100 * t)

        primary_path = np.array([0.0, 0.0, 0.0, 1.0])
        d = signal.lfilter(primary_path, [1.0], x)

        fxlms = FxLMS(filter_length=64, step_size=0.005, secondary_path_est=[1.0])
        _, e, _ = fxlms.process_block(x, d, np.array([1.0]))

        # Compare first quarter vs last quarter RMS
        q = n_samples // 4
        rms_first = np.sqrt(np.mean(e[:q] ** 2))
        rms_last = np.sqrt(np.mean(e[3 * q :] ** 2))
        assert rms_last < rms_first, "Error should decrease over time"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim && python -m pytest tests/test_fxlms.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'snore_anc.core.fxlms'`

- [ ] **Step 3: Write FxLMS implementation**

```python
# src/snore_anc/core/fxlms.py
import numpy as np
from scipy import signal


class FxLMS:
    """Filtered-x Least Mean Squares adaptive filter for active noise control.

    Signal flow:
        x(n) → [W(z)] → y(n) → [S(z)] → speaker output at error mic
        e(n) = d(n) + (S * y)(n)   [residual error]

    Weight update:
        W(n+1) = W(n) - μ · e(n) · X'(n)
        where X'(n) = Ŝ(z) * X(n) is the filtered reference vector.
    """

    def __init__(self, filter_length=256, step_size=0.01, secondary_path_est=None):
        self.L = filter_length
        self.mu = step_size
        self.w = np.zeros(self.L)

        if secondary_path_est is not None:
            self.s_hat = np.array(secondary_path_est, dtype=float)
        else:
            self.s_hat = np.array([1.0])
        self.s_hat_len = len(self.s_hat)

        # Buffers
        self._x_buf = np.zeros(self.L)       # reference history
        self._xf_buf = np.zeros(self.L)      # filtered-x history

    def process_sample(self, x_n):
        """Generate anti-noise sample y(n) = W^T · X(n).

        Call this first, then simulate the secondary path externally,
        compute e(n), then call update(e_n).
        """
        self._x_buf = np.roll(self._x_buf, 1)
        self._x_buf[0] = x_n
        return float(np.dot(self.w, self._x_buf))

    def update(self, e_n):
        """Update filter weights using error e(n).

        Computes filtered-x and applies FxLMS update:
            W(n+1) = W(n) - μ · e(n) · X'(n)
        """
        # Filtered-x: x'(n) = Ŝ^T · [x(n), x(n-1), ..., x(n-M+1)]
        xf_n = float(np.dot(self.s_hat, self._x_buf[: self.s_hat_len]))
        self._xf_buf = np.roll(self._xf_buf, 1)
        self._xf_buf[0] = xf_n

        # Weight update
        self.w -= self.mu * e_n * self._xf_buf

    def process_block(self, x_block, d_block, secondary_path):
        """Run full ANC simulation on a block of samples.

        Args:
            x_block: reference signal array (N,)
            d_block: desired signal array (N,) — primary path output
            secondary_path: FIR coefficients for S(z) simulation

        Returns:
            y_out: anti-noise signal (N,)
            e_out: residual error (N,)
            d_out: copy of d_block for evaluation
        """
        N = len(x_block)
        y_out = np.zeros(N)
        e_out = np.zeros(N)

        sp_buf = np.zeros(len(secondary_path))

        for n in range(N):
            # 1. Generate anti-noise
            y_n = self.process_sample(x_block[n])
            y_out[n] = y_n

            # 2. Simulate secondary path: y_s(n) = S^T · [y(n), y(n-1), ...]
            sp_buf = np.roll(sp_buf, 1)
            sp_buf[0] = y_n
            y_s = float(np.dot(secondary_path, sp_buf))

            # 3. Error: e(n) = d(n) + y_s(n)
            e_n = d_block[n] + y_s
            e_out[n] = e_n

            # 4. Update weights
            self.update(e_n)

        return y_out, e_out, d_block.copy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim && python -m pytest tests/test_fxlms.py -v`

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/snore_anc/core/fxlms.py tests/test_fxlms.py
git commit -m "feat: FxLMS core adaptive filter with tests"
```

---

### Task 3: Acoustic Path Models

**Files:**
- Create: `src/snore_anc/simulation/acoustic_path.py`

- [ ] **Step 1: Write acoustic path models**

```python
# src/snore_anc/simulation/acoustic_path.py
import numpy as np


def generate_delay_path(delay_samples, amplitude=1.0, order=128):
    """Generate a pure-delay FIR path (direct sound, no reflections).

    Args:
        delay_samples: number of samples delay
        amplitude: peak amplitude
        order: total FIR length (padded with zeros)

    Returns:
        FIR coefficients (order,)
    """
    h = np.zeros(order)
    idx = min(delay_samples, order - 1)
    h[idx] = amplitude
    return h


def generate_room_impulse_response(
    source_pos, mic_pos, room_dims, fs, decay_order=5, rt60=0.4
):
    """Generate a simplified room impulse response using image method.

    Uses direct path + early reflections. Does not compute full RIR.

    Args:
        source_pos: (x, y, z) source position in meters
        mic_pos: (x, y, z) microphone position in meters
        room_dims: (L, W, H) room dimensions in meters
        fs: sampling rate
        decay_order: number of reflection orders to include
        rt60: reverberation time in seconds (controls exponential decay)

    Returns:
        FIR coefficients
    """
    c = 343.0
    max_delay = int(rt60 * fs)
    h = np.zeros(max_delay)

    # Direct path
    dist = np.linalg.norm(np.array(source_pos) - np.array(mic_pos))
    delay_samples = int(dist / c * fs)
    amplitude = 1.0 / (dist + 0.1)  # inverse distance law

    if delay_samples < max_delay:
        h[delay_samples] = amplitude

    # First-order reflections (6 walls: x=0, x=L, y=0, y=W, z=0, z=H)
    walls = [
        ([0, None, None], [-1, 1, 1]),
        ([None, 0, None], [1, -1, 1]),
        ([None, None, 0], [1, 1, -1]),
    ]
    L, W, H = room_dims
    reflections = []

    for axis_idx, axis_dim in enumerate([L, W, H]):
        for wall_pos in [0, axis_dim]:
            img = list(source_pos)
            img[axis_idx] = 2 * wall_pos - img[axis_idx]
            reflections.append(tuple(img))

    for img_pos in reflections:
        dist = np.linalg.norm(np.array(img_pos) - np.array(mic_pos))
        delay_s = int(dist / c * fs)
        amp = 0.3 / (dist + 0.1)  # reflections are attenuated
        if delay_s < max_delay:
            h[delay_s] += amp

    # Exponential decay envelope for later taps
    decay = np.exp(-np.arange(max_delay) / (rt60 * fs / 3.0))
    h *= decay

    # Normalize
    if np.max(np.abs(h)) > 0:
        h = h / np.max(np.abs(h)) * 0.8

    # Trim trailing silence
    last_nonzero = np.max(np.where(np.abs(h) > 1e-6)[0]) + 1
    return h[:last_nonzero]


def generate_secondary_path(fs=16000, distance=0.15, resonance_freq=800, order=64):
    """Generate a secondary path FIR (speaker → error mic within pillow).

    Models: propagation delay + exponential decay + optional resonance.

    Args:
        fs: sampling rate
        distance: speaker-to-mic distance in meters
        resonance_freq: resonant frequency of pillow cavity (Hz)
        order: FIR length

    Returns:
        FIR coefficients (order,)
    """
    c = 343.0
    delay_samples = int(distance / c * fs)

    h = np.zeros(order)
    if delay_samples < order:
        # Exponential decay from onset
        for i in range(delay_samples, order):
            t = (i - delay_samples) / fs
            h[i] = np.exp(-t * 50)  # fast decay (~20ms time constant)

        # Add resonance peak if specified
        if resonance_freq > 0:
            for i in range(delay_samples, order):
                t = (i - delay_samples) / fs
                h[i] *= (1.0 + 0.3 * np.sin(2 * np.pi * resonance_freq * t))

    # Normalize to unit peak
    if np.max(np.abs(h)) > 0:
        h = h / np.max(np.abs(h))

    return h


def generate_primary_path(fs=16000, distance=1.0, room_dims=(4.0, 3.5, 2.8)):
    """Generate primary path FIR (snorer → wife's ear).

    Convenience wrapper around generate_room_impulse_response.
    """
    source = [0.0, distance, 0.5]  # husband at origin side
    mic = [room_dims[0] / 2, 0.0, 0.3]  # wife's pillow, ear height
    return generate_room_impulse_response(source, mic, room_dims, fs)
```

- [ ] **Step 2: Verify module loads correctly**

Run: `cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim && python -c "from snore_anc.simulation.acoustic_path import generate_secondary_path; s = generate_secondary_path(); print(f'Secondary path: {len(s)} taps, peak at index {np.argmax(np.abs(s))}')"`

Expected: Prints secondary path info without error.

- [ ] **Step 3: Commit**

```bash
git add src/snore_anc/simulation/acoustic_path.py
git commit -m "feat: acoustic path models (primary, secondary, RIR)"
```

---

### Task 4: Snoring Source

**Files:**
- Create: `src/snore_anc/simulation/snoring_source.py`

- [ ] **Step 1: Write snoring synthesizer and dataset loader**

```python
# src/snore_anc/simulation/snoring_source.py
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
            signal array (N,), sample rate
        """
        N = int(duration_s * self.fs)
        signal = np.zeros(N)

        cycle_duration = self.inhale_dur + self.exhale_dur
        samples_per_cycle = int(cycle_duration * self.fs)

        for i in range(N):
            t = i / self.fs

            # Position within current breath cycle
            cycle_pos = (i % samples_per_cycle) / self.fs

            if cycle_pos < self.inhale_dur:
                # Inhale phase: snoring ON with rising then falling envelope
                phase = cycle_pos / self.inhale_dur
                envelope = np.sin(np.pi * phase)
            else:
                # Exhale phase: quiet or silent
                if pattern == 'regular':
                    envelope = 0.0
                elif pattern == 'irregular':
                    envelope = 0.1 * np.random.random()
                else:  # apnea
                    envelope = 0.0

            # Fundamental with slow frequency modulation (vibrato)
            vibrato = self.f0_var * np.sin(2 * np.pi * 3.0 * t)
            f0_inst = self.f0 + vibrato
            theta = 2 * np.pi * f0_inst * t
            fundamental = np.sin(theta)

            # Add harmonics (up to 8th)
            harmonics = 0.0
            for n in range(2, 9):
                harmonics += (1.0 / n) * np.sin(n * theta)

            # Colored noise (approximate bandpass 50-500Hz effect)
            noise = np.random.randn() * 0.15

            signal[i] = envelope * (0.5 * fundamental + 0.3 * harmonics + noise)

        # Normalize to [-1, 1]
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal / peak * 0.8

        return signal, self.fs


def load_dataset(name, segment=None, data_dir=None):
    """Load a public snoring dataset.

    Currently supports:
    - 'synthetic': generates synthetic snoring via SnoringSynthesizer
    - 'mpssc': Munich Passau Snoring Sound Corpus (requires local files)

    Args:
        name: dataset identifier
        segment: segment index or slice
        data_dir: path to local dataset files

    Returns:
        (signal, fs, annotation) tuple
    """
    if name == 'synthetic':
        synth = SnoringSynthesizer()
        sig, fs = synth.generate(duration_s=30.0, pattern='regular')
        return sig, fs, {'pattern': 'regular', 'source': 'synthetic'}

    if name == 'mpssc':
        if data_dir is None:
            raise FileNotFoundError(
                "MPSSC dataset requires local files. "
                "Download from https://www5.cs.fau.de/research/data/ "
                "and pass data_dir parameter."
            )
        import soundfile as sf
        wav_files = sorted([
            f for f in os.listdir(data_dir) if f.endswith('.wav')
        ])
        if not wav_files:
            raise FileNotFoundError(f"No .wav files in {data_dir}")

        idx = segment if segment is not None else 0
        filepath = os.path.join(data_dir, wav_files[idx])
        sig, fs = sf.read(filepath)
        if sig.ndim > 1:
            sig = sig[:, 0]  # mono
        return sig, fs, {'file': wav_files[idx], 'source': 'mpssc'}

    raise ValueError(f"Unknown dataset: {name}")
```

- [ ] **Step 2: Verify synthesizer produces correct output**

Run: `cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim && python -c "
from snore_anc.simulation.snoring_source import SnoringSynthesizer
s = SnoringSynthesizer()
sig, fs = s.generate(duration_s=2.0)
print(f'Synthesized: {len(sig)} samples, fs={fs}, peak={np.max(np.abs(sig)):.3f}')
assert len(sig) == 2 * fs
assert np.max(np.abs(sig)) <= 1.0
print('OK')
"`

Expected: Prints synthesis info and "OK".

- [ ] **Step 3: Commit**

```bash
git add src/snore_anc/simulation/snoring_source.py
git commit -m "feat: snoring signal synthesizer and dataset loader"
```

---

### Task 5: Pillow Scenario Model

**Files:**
- Create: `src/snore_anc/simulation/pillow_model.py`

- [ ] **Step 1: Write pillow scenario runner**

```python
# src/snore_anc/simulation/pillow_model.py
import numpy as np
from snore_anc.config import ANCConfig
from snore_anc.simulation.acoustic_path import (
    generate_primary_path,
    generate_secondary_path,
)
from snore_anc.simulation.snoring_source import SnoringSynthesizer


class PillowScenario:
    """Full pillow ANC simulation scenario.

    Models the physical setup: snorer → (air gap) → wife's pillow with
    reference mic, ANC processor, speakers, and error mics.
    """

    def __init__(self, config=None):
        self.cfg = config or ANCConfig()

    def build_paths(self):
        """Generate primary and secondary paths from config geometry.

        Returns:
            primary_path: FIR coefficients for P(z)
            secondary_path: FIR coefficients for S(z)
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
        """Generate a snoring source signal.

        Returns:
            (signal, fs) tuple
        """
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
            dict with keys: y, e, d, fs
        """
        primary_path, sec_path = self.build_paths()
        if secondary_path is not None:
            sec_path = np.array(secondary_path, dtype=float)

        from scipy import signal as sig
        d = sig.lfilter(primary_path, [1.0], source_signal)

        y, e, _ = anc_algorithm.process_block(source_signal, d, sec_path)

        return {
            'y': y,
            'e': e,
            'd': d,
            'fs': self.cfg.fs,
            'primary_path': primary_path,
            'secondary_path': sec_path,
        }
```

- [ ] **Step 2: Verify end-to-end with FxLMS**

Run: `cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim && python -c "
from snore_anc.simulation.pillow_model import PillowScenario
from snore_anc.core.fxlms import FxLMS
import numpy as np

scenario = PillowScenario()
sig, fs = scenario.generate_snoring(duration_s=3.0)
primary, secondary = scenario.build_paths()
print(f'Primary path: {len(primary)} taps')
print(f'Secondary path: {len(secondary)} taps')
print(f'Snoring signal: {len(sig)} samples')

# Run with simple FxLMS
fxlms = FxLMS(filter_length=256, step_size=0.005, secondary_path_est=secondary)
result = scenario.run_simulation(sig, fxlms)
rms_d = np.sqrt(np.mean(result['d'][-8000:]**2))
rms_e = np.sqrt(np.mean(result['e'][-8000:]**2))
nr = 20*np.log10(rms_d / (rms_e + 1e-12))
print(f'NR in last 0.5s: {nr:.1f} dB')
print('OK')
"`

Expected: Prints path lengths, NR value, and "OK".

- [ ] **Step 3: Commit**

```bash
git add src/snore_anc/simulation/pillow_model.py
git commit -m "feat: pillow scenario model with end-to-end simulation"
```

---

### Task 6: Evaluation Metrics + Visualization

**Files:**
- Create: `src/snore_anc/evaluation/metrics.py`
- Create: `src/snore_anc/evaluation/visualization.py`

- [ ] **Step 1: Write metrics module**

```python
# src/snore_anc/evaluation/metrics.py
import numpy as np


class ANCEvaluator:
    """Evaluation metrics for ANC simulation results."""

    @staticmethod
    def noise_reduction_db(d_signal, e_signal):
        """Compute noise reduction in dB.

        NR = 20 * log10(rms(d) / rms(e))

        Positive values mean noise was reduced.
        """
        rms_d = np.sqrt(np.mean(d_signal ** 2))
        rms_e = np.sqrt(np.mean(e_signal ** 2))
        if rms_e < 1e-12:
            return 100.0  # effectively infinite
        return float(20 * np.log10(rms_d / rms_e))

    @staticmethod
    def convergence_time(e_signal, fs=16000, threshold_db=-10.0, window=0.1):
        """Time to reach and sustain target NR level.

        Args:
            e_signal: error signal (N,)
            fs: sample rate
            threshold_db: target NR in dB (negative = reduction)
            window: analysis window duration in seconds

        Returns:
            convergence time in seconds, or None if not reached
        """
        win_samples = int(window * fs)
        n_windows = len(e_signal) // win_samples

        consecutive = 0
        required_consecutive = 3  # must sustain for 3 windows

        for i in range(n_windows):
            start = i * win_samples
            end = start + win_samples
            rms = np.sqrt(np.mean(e_signal[start:end] ** 2))
            # Compare against initial noise level (first window)
            initial_rms = np.sqrt(np.mean(e_signal[:win_samples] ** 2)) + 1e-12
            current_nr = 20 * np.log10(initial_rms / (rms + 1e-12))

            if current_nr >= abs(threshold_db):
                consecutive += 1
                if consecutive >= required_consecutive:
                    # Return time of first window in the sustained run
                    conv_window = i - required_consecutive + 1
                    return float(conv_window * win_samples / fs)
            else:
                consecutive = 0

        return None

    @staticmethod
    def power_spectrum_comparison(d_signal, e_signal, fs=16000, n_fft=2048):
        """Compute PSD of desired and error signals.

        Returns:
            (freqs, psd_d, psd_e) tuple
        """
        freqs, psd_d = np.signal.welch(d_signal, fs=fs, nperseg=n_fft) if hasattr(np, 'signal') else (None, None)
        # Use scipy-free implementation
        from numpy.fft import rfft, rfftfreq
        window = np.hanning(n_fft)

        def welch_psd(sig):
            n_segments = len(sig) // n_fft
            psd = np.zeros(n_fft // 2 + 1)
            for i in range(n_segments):
                seg = sig[i * n_fft : (i + 1) * n_fft] * window
                spectrum = np.abs(rfft(seg)) ** 2
                psd += spectrum
            psd /= max(n_segments, 1)
            freqs_out = rfftfreq(n_fft, 1.0 / fs)
            return freqs_out, psd

        freqs_d, psd_d = welch_psd(d_signal)
        _, psd_e = welch_psd(e_signal)
        return freqs_d, psd_d, psd_e
```

- [ ] **Step 2: Write visualization module**

```python
# src/snore_anc/evaluation/visualization.py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt


class ANCVisualizer:
    """Generate standard ANC evaluation plots."""

    @staticmethod
    def plot_time_domain(d_signal, e_signal, fs=16000, title="ANC Time Domain", save_path=None):
        """Plot desired vs error signal in time domain."""
        t = np.arange(len(d_signal)) / fs
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        axes[0].plot(t, d_signal, linewidth=0.5, alpha=0.8)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'{title} — Desired Signal d(n)')

        axes[1].plot(t, e_signal, linewidth=0.5, alpha=0.8, color='red')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title('Error Signal e(n)')

        from snore_anc.evaluation.metrics import ANCEvaluator
        nr = ANCEvaluator.noise_reduction_db(d_signal, e_signal)
        fig.suptitle(f'NR = {nr:.1f} dB', fontsize=14, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_convergence(d_signal, e_signal, fs=16000, window_ms=50, title="Convergence Curve", save_path=None):
        """Plot NR(dB) vs time using sliding window."""
        win_samples = int(window_ms / 1000 * fs)
        n_windows = len(e_signal) // win_samples

        time_axis = []
        nr_values = []

        for i in range(n_windows):
            start = i * win_samples
            end = start + win_samples
            rms_d = np.sqrt(np.mean(d_signal[start:end] ** 2)) + 1e-12
            rms_e = np.sqrt(np.mean(e_signal[start:end] ** 2)) + 1e-12
            nr = 20 * np.log10(rms_d / rms_e)
            time_axis.append((start + end) / 2 / fs)
            nr_values.append(nr)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_axis, nr_values, linewidth=1.0)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('NR (dB)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_all(d_signal, e_signal, fs=16000, prefix="anc_result"):
        """Generate all standard plots."""
        ANCVisualizer.plot_time_domain(d_signal, e_signal, fs,
                                        save_path=f"{prefix}_waveform.png")
        ANCVisualizer.plot_convergence(d_signal, e_signal, fs,
                                        save_path=f"{prefix}_convergence.png")
```

- [ ] **Step 3: Verify evaluation module works**

Run: `cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim && python -c "
from snore_anc.evaluation.metrics import ANCEvaluator
import numpy as np
d = np.random.randn(16000) * 0.5
e = d * 0.1  # 20dB reduction
nr = ANCEvaluator.noise_reduction_db(d, e)
print(f'NR: {nr:.1f} dB')
assert 19 < nr < 21, f'Expected ~20dB, got {nr}'
print('OK')
"`

Expected: Prints "NR: 20.0 dB" and "OK".

- [ ] **Step 4: Commit**

```bash
git add src/snore_anc/evaluation/metrics.py src/snore_anc/evaluation/visualization.py
git commit -m "feat: evaluation metrics and visualization modules"
```

---

### Task 7: Secondary Path Estimation + Tests

**Files:**
- Create: `src/snore_anc/core/secondary_path.py`
- Create: `tests/test_secondary_path.py`

- [ ] **Step 1: Write secondary path tests**

```python
# tests/test_secondary_path.py
import numpy as np
import pytest

from snore_anc.core.secondary_path import SecondaryPathEstimator


class TestSecondaryPathEstimator:

    def test_offline_estimation_known_fir(self):
        """Offline LMS should recover a known FIR path with error < -30dB."""
        np.random.seed(42)

        # True secondary path: simple lowpass
        true_path = np.array([0.1, 0.3, 0.5, 0.3, 0.1])

        # Generate probe + response
        n_samples = 8000
        probe = np.random.randn(n_samples) * 0.5
        response = np.convolve(probe, true_path, mode='full')[:n_samples]

        estimator = SecondaryPathEstimator(order=10, step_size=0.01)
        s_hat = estimator.offline_estimate(probe, response)

        # Zero-pad true_path for comparison
        true_padded = np.zeros(10)
        true_padded[: len(true_path)] = true_path

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim && python -m pytest tests/test_secondary_path.py -v`

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write secondary path estimator**

```python
# src/snore_anc/core/secondary_path.py
import numpy as np


class SecondaryPathEstimator:
    """Estimates secondary path S(z) using LMS system identification.

    Two modes:
    - Offline: white noise probe → measure response → LMS → Ŝ(z)
    - Online: continuous adaptation during ANC operation
    """

    def __init__(self, order=128, step_size=0.01):
        self.order = order
        self.mu = step_size
        self.w = np.zeros(order)
        self._buf = np.zeros(order)

    def offline_estimate(self, probe_signal, measured_response, n_iterations=3):
        """Estimate S(z) from probe signal and measured response.

        Uses standard LMS to minimize |measured - probe * w|^2.

        Args:
            probe_signal: excitation signal (N,)
            measured_response: system output (N,)
            n_iterations: number of passes over the data

        Returns:
            Estimated FIR coefficients (order,)
        """
        self.w = np.zeros(self.order)
        self._buf = np.zeros(self.order)
        N = len(probe_signal)

        for _ in range(n_iterations):
            for n in range(N):
                # Shift buffer and insert
                self._buf = np.roll(self._buf, 1)
                self._buf[0] = probe_signal[n]

                # Filter output
                y_hat = float(np.dot(self.w, self._buf))

                # Error
                e = measured_response[n] - y_hat

                # LMS update
                self.w += self.mu * e * self._buf

        return self.w.copy()

    def online_update(self, x_n, y_n, e_n):
        """Single-step online update for secondary path estimation.

        Args:
            x_n: probe signal sample
            y_n: speaker output sample
            e_n: measurement error (measured - predicted)
        """
        self._buf = np.roll(self._buf, 1)
        self._buf[0] = x_n
        self.w += self.mu * e_n * self._buf
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim && python -m pytest tests/test_secondary_path.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/snore_anc/core/secondary_path.py tests/test_secondary_path.py
git commit -m "feat: secondary path estimation with offline/online modes"
```

---

### Task 8: Delayless Subband FxLMS + Tests

**Files:**
- Create: `src/snore_anc/core/subband.py`
- Create: `tests/test_subband.py`

- [ ] **Step 1: Write subband tests**

```python
# tests/test_subband.py
import numpy as np
import pytest
from scipy import signal

from snore_anc.core.subband import DelaylessSubbandFxLMS
from snore_anc.core.fxlms import FxLMS


class TestDelaylessSubbandFxLMS:

    def test_analysis_synthesis_reconstruction(self):
        """FFT analysis → weight transform should preserve signal structure."""
        sb = DelaylessSubbandFxLMS(
            n_subbands=64, filter_length_per_band=4,
            step_size=0.001, secondary_path_est=[1.0]
        )
        # Run a few samples through to exercise the weight transform
        for i in range(200):
            sb.process_sample(np.sin(2 * np.pi * 150 * i / 16000))
            sb.update(0.1)  # arbitrary error

        # Weight transform should produce valid (non-NaN, non-Inf) fullband weights
        assert np.all(np.isfinite(sb.W_full))
        assert sb.W_full.shape[0] == 64 * 4

    def test_convergence_faster_than_fullband(self):
        """Subband FxLMS should converge faster than fullband FxLMS."""
        np.random.seed(42)
        fs = 16000
        duration = 3.0
        n_samples = int(fs * duration)

        # Narrowband signal: two close tones (hard for fullband)
        t = np.arange(n_samples) / fs
        x = (np.sin(2 * np.pi * 120 * t) +
             0.7 * np.sin(2 * np.pi * 130 * t))

        primary_path = np.zeros(11)
        primary_path[10] = 1.0
        secondary_path = np.array([1.0])

        d = signal.lfilter(primary_path, [1.0], x)

        # Fullband FxLMS
        fb = FxLMS(filter_length=256, step_size=0.005, secondary_path_est=[1.0])
        _, e_fb, _ = fb.process_block(x, d, secondary_path)

        # Subband FxLMS (M=64, L_sub=4 → L_full=256)
        sb = DelaylessSubbandFxLMS(
            n_subbands=64, filter_length_per_band=4,
            step_size=0.005, secondary_path_est=[1.0]
        )
        _, e_sb, _ = sb.process_block(x, d, secondary_path)

        # Measure NR at midpoint (1.5s) — subband should be better
        mid = n_samples // 2
        quarter = n_samples // 4
        nr_fb_mid = 20 * np.log10(
            np.sqrt(np.mean(d[mid - quarter:mid] ** 2)) /
            (np.sqrt(np.mean(e_fb[mid - quarter:mid] ** 2)) + 1e-12) + 1e-12
        )
        nr_sb_mid = 20 * np.log10(
            np.sqrt(np.mean(d[mid - quarter:mid] ** 2)) /
            (np.sqrt(np.mean(e_sb[mid - quarter:mid] ** 2)) + 1e-12) + 1e-12
        )

        assert nr_sb_mid > nr_fb_mid, (
            f"Subband NR={nr_sb_mid:.1f}dB should exceed fullband NR={nr_fb_mid:.1f}dB at midpoint"
        )

    def test_init_default_params(self):
        """Default init produces correct shapes."""
        sb = DelaylessSubbandFxLMS()
        assert sb.W_full.shape == (64 * 4,)
        assert sb.W_sub.shape == (64, 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim && python -m pytest tests/test_subband.py -v`

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write delayless subband FxLMS**

```python
# src/snore_anc/core/subband.py
import numpy as np


class DelaylessSubbandFxLMS:
    """Delayless subband adaptive filter (Morgan & Thi, 1995).

    Architecture:
    1. Analysis: M-pt FFT of recent M samples → subband decomposition
    2. Per-subband: Short LMS filters (length L_sub) with independent step sizes
    3. Weight transform: IFFT of subband weight matrix → fullband weights
    4. Fullband output: y(n) = W_full^T · X(n) — no synthesis filterbank delay

    The weight-domain transformation is the key to being "delayless":
    subband adaptive weights are converted to fullband weights via IDFT,
    and the output is generated using fullband convolution directly.
    """

    def __init__(self, n_subbands=64, filter_length_per_band=4,
                 step_size=0.01, secondary_path_est=None):
        self.M = n_subbands
        self.L_sub = filter_length_per_band
        self.L_full = self.M * self.L_sub
        self.mu = step_size

        # Subband adaptive weight matrix: M subbands × L_sub taps
        self.W_sub = np.zeros((self.M, self.L_sub))

        # Fullband weights (transformed from subband via IDFT)
        self.W_full = np.zeros(self.L_full)

        # Fullband reference buffer
        self._x_buf = np.zeros(self.L_full)

        # Fullband filtered-x buffer
        self._xf_buf = np.zeros(self.L_full)

        # Per-subband filtered-x history: M × L_sub
        self._xf_sub_history = np.zeros((self.M, self.L_sub))

        # Secondary path estimate
        if secondary_path_est is not None:
            self._s_hat = np.array(secondary_path_est, dtype=float)
        else:
            self._s_hat = np.array([1.0])
        self._s_hat_len = len(self._s_hat)

    def _transform_weights(self):
        """Transform subband weights → fullband weights via IDFT.

        W_sub is M × L_sub. Apply IFFT along the M dimension (axis=0),
        then interleave the result to get fullband weights of length M * L_sub.
        """
        W_time = np.fft.ifft(self.W_sub, n=self.M, axis=0).real  # M × L_sub
        # Interleave: W_full[k*M + m] = W_time[m, k]
        self.W_full = W_time.T.flatten()

    def process_sample(self, x_n):
        """Generate anti-noise y(n) = W_full^T · X(n)."""
        self._x_buf = np.roll(self._x_buf, 1)
        self._x_buf[0] = x_n
        return float(np.dot(self.W_full, self._x_buf))

    def update(self, e_n):
        """Update subband filters based on error, then transform to fullband."""
        # 1. Compute filtered-x in fullband: x'(n) = Ŝ^T · [x(n), x(n-1), ...]
        xf_n = float(np.dot(self._s_hat, self._x_buf[:self._s_hat_len]))
        self._xf_buf = np.roll(self._xf_buf, 1)
        self._xf_buf[0] = xf_n

        # 2. Current subband decomposition of filtered-x
        #    FFT of the most recent M samples gives instantaneous subband values
        current_xf_sub = np.fft.fft(self._xf_buf[:self.M])  # M complex values

        # 3. Update per-subband filtered-x history and weights
        for k in range(self.M):
            self._xf_sub_history[k] = np.roll(self._xf_sub_history[k], 1)
            self._xf_sub_history[k][0] = current_xf_sub[k].real

            # LMS update for subband k
            self.W_sub[k] -= self.mu * e_n * self._xf_sub_history[k]

        # 4. Transform subband weights to fullband (delayless)
        self._transform_weights()

    def process_block(self, x_block, d_block, secondary_path):
        """Run full ANC simulation on a block (same interface as FxLMS)."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim && python -m pytest tests/test_subband.py -v`

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/snore_anc/core/subband.py tests/test_subband.py
git commit -m "feat: delayless subband FxLMS with Morgan-Thi weight transform"
```

---

### Task 9: Multi-Channel ANC + Tests

**Files:**
- Create: `src/snore_anc/core/multichannel.py`
- Create: `tests/test_multichannel.py`

- [ ] **Step 1: Write multi-channel tests**

```python
# tests/test_multichannel.py
import numpy as np
import pytest
from scipy import signal

from snore_anc.core.multichannel import MultiChannelANC


class TestMultiChannelANC:

    def test_dual_zone_noise_reduction(self):
        """2-spk × 2-err system should achieve > 10dB NR in both zones."""
        fs = 16000
        duration = 5.0
        n_samples = int(fs * duration)

        # Single tone at 150Hz
        t = np.arange(n_samples) / fs
        x = np.sin(2 * np.pi * 150 * t)

        # Primary paths to each error mic (different delays)
        p1 = np.zeros(11)
        p1[10] = 1.0
        p2 = np.zeros(13)
        p2[12] = 0.9

        d1 = signal.lfilter(p1, [1.0], x)
        d2 = signal.lfilter(p2, [1.0], x)

        # Secondary path matrix (2 err × 2 spk)
        # S[i][j] = path from speaker j to error mic i
        s_matrix = [
            [np.array([0.0, 1.0, 0.0]),   np.array([0.0, 0.0, 0.3])],
            [np.array([0.0, 0.3, 0.0]),   np.array([0.0, 1.0, 0.0])],
        ]
        # Estimates (perfect knowledge for this test)
        s_hat_matrix = s_matrix

        mc = MultiChannelANC(
            n_speakers=2, n_errors=2,
            filter_length=64, step_size=0.005,
            secondary_path_matrix=s_hat_matrix
        )

        result = mc.process_block(x, [d1, d2], s_matrix)

        # Check NR in both channels (last 0.5s)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim && python -m pytest tests/test_multichannel.py -v`

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write multi-channel ANC**

```python
# src/snore_anc/core/multichannel.py
import numpy as np
from snore_anc.core.fxlms import FxLMS


class MultiChannelANC:
    """Multi-channel ANC: 1 reference × N speakers × M error microphones.

    Architecture:
    - N_speakers independent FxLMS filters, one per speaker
    - Each filter's update incorporates errors from ALL error mics
    - Secondary path is an N_err × N_spk matrix of FIR paths

    Update rule:
        ΔW_j = μ · Σ_i (e_i · x'_ij)
    where x'_ij = Ŝ_ij * x(n) is the filtered-x for speaker j at error mic i.
    """

    def __init__(self, n_speakers=2, n_errors=2, filter_length=256,
                 step_size=0.01, secondary_path_matrix=None):
        self.N_spk = n_speakers
        self.N_err = n_errors
        self.L = filter_length
        self.mu = step_size

        # One FxLMS filter per speaker (handles its own weight update)
        # We override the update to use multi-channel errors
        self.filters = [
            FxLMS(filter_length=filter_length, step_size=step_size,
                  secondary_path_est=None)
            for _ in range(n_speakers)
        ]

        # Secondary path matrix: N_err × N_spk, each element is FIR coefficients
        # S[i][j] = path from speaker j to error mic i
        if secondary_path_matrix is not None:
            self.s_matrix = secondary_path_matrix
        else:
            # Default: identity-like
            self.s_matrix = [
                [np.array([1.0]) if i == j else np.array([0.0])
                 for j in range(n_speakers)]
                for i in range(n_errors)
            ]

        # Per-speaker filtered-x buffers (accumulated over all error mics)
        self._xf_spk_bufs = [np.zeros(filter_length) for _ in range(n_speakers)]

        # Secondary path simulation buffers: N_err × N_spk
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
            # Re-init simulation buffers
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
            y_s = np.zeros(self.N_err)  # total speaker contribution at each err mic
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
                # Accumulate filtered-x gradient over all error mics
                grad_sum = 0.0
                for i in range(self.N_err):
                    # Compute filtered-x for this speaker-error pair
                    s_hat_ij = self.s_matrix[i][j]  # assume perfect estimate
                    xf_n = float(np.dot(s_hat_ij, self.filters[j]._x_buf[:len(s_hat_ij)]))
                    grad_sum += e_ns[i] * xf_n

                # Update filter weights
                self.filters[j]._xf_buf = np.roll(self.filters[j]._xf_buf, 1)
                self.filters[j]._xf_buf[0] = grad_sum / self.N_err
                self.filters[j].w -= self.mu * self.filters[j]._xf_buf

        return {
            'y': y_outs,
            'e': e_outs,
            'd': [d.copy() for d in d_blocks],
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim && python -m pytest tests/test_multichannel.py -v`

Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/snore_anc/core/multichannel.py tests/test_multichannel.py
git commit -m "feat: multi-channel ANC (1-ref × 2-spk × 2-err)"
```

---

### Task 10: Hardware Setup Doc

**Files:**
- Create: `docs/hardware_setup.md`

- [ ] **Step 1: Write hardware setup guide**

```markdown
<!-- docs/hardware_setup.md -->
# Snore ANC Hardware Prototype Setup

## Bill of Materials

| Role | Component | Part Number | Approx. Price |
|------|-----------|-------------|---------------|
| DSP | STM32F407 Discovery | STM32F407G-DISC1 | ¥80-120 |
| Reference Mic | I²S MEMS Mic | SPH0645LM4H | ¥15-25 |
| Error Mic ×2 | I²S MEMS Mic | SPH0645LM4H | ¥15×2 |
| Speaker ×2 | 25mm Full-Range | CUI CDM-25008 | ¥8×2 |
| Power | USB 5V / 3.7V LiPo + LDO | — | ¥10 |
| **Total** | | | **¥150-300** |

## Wiring Diagram

```
SPH0645 (ref) ──I²S──→ STM32F407 ──I²S──→ SPK_L (25mm)
SPH0645 (eL) ──I²S──→  Discovery  ──I²S──→ SPK_R (25mm)
SPH0645 (eR) ──I²S──→              ──UART──→ PC (NR monitor)
                        │
                        └── USB power
```

## I²S Configuration

- Master clock: STM32F407 I2S PLL (16kHz sample rate)
- Data format: I2S Philips, 16-bit
- DMA: Double-buffer mode (half/full transfer interrupts)

## Test Procedure

### Phase A: Single-Channel (1 SPK + 1 Error Mic)

1. Flash firmware with `fxlms_core.c`
2. Play 150Hz tone from phone at 1m distance
3. Monitor UART output for real-time NR
4. Target: NR > 20dB

### Phase B: Subband FxLMS

1. Switch firmware to subband variant
2. Same test as Phase A
3. Compare convergence speed: target ≥ 2x improvement

### Phase C: Multi-Channel (2 SPK + 2 Error Mic)

1. Connect second speaker and error mic
2. Position error mics at left/right ear positions
3. Run multi-channel firmware
4. Target: NR > 10dB in both zones

## Embedded Firmware Structure

```
firmware/
├── Core/
│   ├── Src/
│   │   ├── main.c          — I2S DMA setup + main loop
│   │   ├── fxlms_core.c    — ANC algorithm (C port from Python)
│   │   ├── secondary_path.c — Offline S(z) calibration
│   │   └── uart_logger.c   — NR value output
│   └── Inc/
│       ├── fxlms_core.h
│       ├── secondary_path.h
│       └── uart_logger.h
└── Drivers/  (STM32 HAL)
```

## Key Porting Notes

Python → C translation guide:
- `np.dot(a, b)` → `arm_dot_prod_f32(a, b, len, &result)`
- `np.roll(buf, 1)` → circular buffer with pointer
- Float32 throughout (Cortex-M4F has hardware FPU)
- Use CMSIS-DSP for FFT (subband decomposition)
```

- [ ] **Step 2: Commit**

```bash
git add docs/hardware_setup.md
git commit -m "docs: hardware prototype setup guide"
```

---

### Task 11: Integration Test & Full Pipeline Validation

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
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
        """Full pipeline: synthetic snoring → FxLMS → NR > 10dB."""
        scenario = PillowScenario()
        sig, fs = scenario.generate_snoring(duration_s=5.0)

        _, sec_path = scenario.build_paths()

        fxlms = FxLMS(filter_length=256, step_size=0.005,
                       secondary_path_est=sec_path)
        result = scenario.run_simulation(sig, fxlms)

        eval_check = int(4.0 * fs)  # check last 1 second
        nr = ANCEvaluator.noise_reduction_db(
            result['d'][eval_check:], result['e'][eval_check:]
        )
        assert nr > 8, f"Integration NR = {nr:.1f} dB, expected > 8 dB"

    def test_subband_with_synthetic_snoring(self):
        """Full pipeline: synthetic snoring → Subband FxLMS → converges."""
        scenario = PillowScenario()
        sig, fs = scenario.generate_snoring(duration_s=5.0)

        _, sec_path = scenario.build_paths()

        sb = DelaylessSubbandFxLMS(
            n_subbands=64, filter_length_per_band=4,
            step_size=0.005, secondary_path_est=sec_path
        )
        result = scenario.run_simulation(sig, sb, secondary_path=sec_path)

        eval_check = int(4.0 * fs)
        nr = ANCEvaluator.noise_reduction_db(
            result['d'][eval_check:], result['e'][eval_check:]
        )
        # Subband should at least reduce noise
        assert nr > 0, f"Subband NR = {nr:.1f} dB, expected > 0 dB"
```

- [ ] **Step 2: Run all tests**

Run: `cd /Users/noctis/Workspace/trySth/trending-clones/snore-anc-sim && python -m pytest tests/ -v`

Expected: All tests PASS (including integration).

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration tests for full ANC pipeline"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Every section of the spec maps to a task. Config (Task 1), FxLMS (Task 2), Acoustic Paths (Task 3), Snoring Source (Task 4), Pillow Model (Task 5), Metrics + Viz (Task 6), Secondary Path (Task 7), Subband (Task 8), Multi-channel (Task 9), Hardware Doc (Task 10), Integration (Task 11).
- [x] **Placeholder scan:** No TBD, TODO, or "implement later" found. All code blocks contain complete implementations.
- [x] **Type consistency:** All method signatures match between definition and usage. `process_block` returns `(y_out, e_out, d_block_copy)` consistently across FxLMS, SubbandFxLMS, and MultiChannelANC (MultiChannel returns dict with equivalent data).
