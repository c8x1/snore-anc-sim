import numpy as np


def generate_delay_path(delay_samples, amplitude=1.0, order=128):
    """Pure-delay FIR path (direct sound, no reflections).

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


def generate_room_impulse_response(source_pos, mic_pos, room_dims, fs,
                                   decay_order=5, rt60=0.4):
    """Simplified RIR using image method (direct path + first-order
    reflections + exponential decay).

    Args:
        source_pos: (x, y, z) source position in meters
        mic_pos: (x, y, z) microphone position in meters
        room_dims: (L, W, H) room dimensions in meters
        fs: sampling rate
        decay_order: not used (kept for API compat)
        rt60: reverberation time in seconds

    Returns:
        FIR coefficients
    """
    c = 343.0
    max_delay = int(rt60 * fs)
    h = np.zeros(max_delay)

    # Direct path
    dist = np.linalg.norm(np.array(source_pos) - np.array(mic_pos))
    delay_samples = int(dist / c * fs)
    amplitude = 1.0 / (dist + 0.1)
    if delay_samples < max_delay:
        h[delay_samples] = amplitude

    # First-order reflections (6 walls)
    L, W, H = room_dims
    for axis_idx, axis_dim in enumerate([L, W, H]):
        for wall_pos in [0, axis_dim]:
            img = list(source_pos)
            img[axis_idx] = 2 * wall_pos - img[axis_idx]
            dist = np.linalg.norm(np.array(img) - np.array(mic_pos))
            delay_s = int(dist / c * fs)
            amp = 0.3 / (dist + 0.1)
            if delay_s < max_delay:
                h[delay_s] += amp

    # Exponential decay envelope
    decay = np.exp(-np.arange(max_delay) / (rt60 * fs / 3.0))
    h *= decay

    if np.max(np.abs(h)) > 0:
        h = h / np.max(np.abs(h)) * 0.8

    # Trim trailing silence
    last_nonzero = np.max(np.where(np.abs(h) > 1e-6)[0]) + 1
    return h[:last_nonzero]


def generate_secondary_path(fs=16000, distance=0.15,
                            resonance_freq=800, order=64):
    """Secondary path FIR (speaker -> error mic within pillow).

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
        for i in range(delay_samples, order):
            t = (i - delay_samples) / fs
            h[i] = np.exp(-t * 50)
            if resonance_freq > 0:
                h[i] *= (1.0 + 0.3 * np.sin(
                    2 * np.pi * resonance_freq * t))
    if np.max(np.abs(h)) > 0:
        h = h / np.max(np.abs(h))
    return h


def generate_primary_path(fs=16000, distance=1.0,
                          room_dims=(4.0, 3.5, 2.8)):
    """Primary path FIR (snorer -> wife's ear). Convenience wrapper."""
    source = [0.0, distance, 0.5]
    mic = [room_dims[0] / 2, 0.0, 0.3]
    return generate_room_impulse_response(source, mic, room_dims, fs)
