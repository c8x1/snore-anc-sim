import numpy as np


class ANCEvaluator:
    @staticmethod
    def noise_reduction_db(d_signal, e_signal):
        """NR = 20 * log10(rms(d) / rms(e)). Positive = noise reduced."""
        rms_d = np.sqrt(np.mean(d_signal ** 2))
        rms_e = np.sqrt(np.mean(e_signal ** 2))
        if rms_e < 1e-12:
            return 100.0
        return float(20 * np.log10(rms_d / rms_e))

    @staticmethod
    def convergence_time(e_signal, fs=16000, threshold_db=-10.0, window=0.1):
        """Time to reach and sustain target NR level. Returns seconds or None."""
        win_samples = int(window * fs)
        n_windows = len(e_signal) // win_samples
        initial_rms = np.sqrt(np.mean(e_signal[:win_samples] ** 2)) + 1e-12
        consecutive = 0
        required = 3
        for i in range(n_windows):
            start = i * win_samples
            end = start + win_samples
            rms = np.sqrt(np.mean(e_signal[start:end] ** 2))
            current_nr = 20 * np.log10(initial_rms / (rms + 1e-12))
            if current_nr >= abs(threshold_db):
                consecutive += 1
                if consecutive >= required:
                    conv_window = i - required + 1
                    return float(conv_window * win_samples / fs)
            else:
                consecutive = 0
        return None

    @staticmethod
    def power_spectrum_comparison(d_signal, e_signal, fs=16000, n_fft=2048):
        """Compute PSD. Returns (freqs, psd_d, psd_e)."""
        from numpy.fft import rfft, rfftfreq

        window = np.hanning(n_fft)

        def welch_psd(sig):
            n_seg = len(sig) // n_fft
            psd = np.zeros(n_fft // 2 + 1)
            for i in range(n_seg):
                seg = sig[i * n_fft : (i + 1) * n_fft] * window
                psd += np.abs(rfft(seg)) ** 2
            psd /= max(n_seg, 1)
            return rfftfreq(n_fft, 1.0 / fs), psd

        freqs, psd_d = welch_psd(d_signal)
        _, psd_e = welch_psd(e_signal)
        return freqs, psd_d, psd_e
