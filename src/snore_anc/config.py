from dataclasses import dataclass


@dataclass
class ANCConfig:
    """Centralized parameters for snore ANC simulation."""
    fs: int = 16000
    filter_length: int = 256
    step_size: float = 0.01
    secondary_path_order: int = 128
    n_subbands: int = 64
    filter_length_per_band: int = 4
    n_speakers: int = 2
    n_error_mics: int = 2
    husband_wife_distance: float = 1.0
    pillow_speaker_mic_distance: float = 0.15
    room_dims: tuple = (4.0, 3.5, 2.8)
    speed_of_sound: float = 343.0
    snore_f0: float = 150.0
    snore_f0_variation: float = 5.0
    snore_inhale_duration: float = 0.4
    snore_exhale_duration: float = 0.3
    nr_threshold_db: float = -10.0
    nr_check_window: float = 0.5
