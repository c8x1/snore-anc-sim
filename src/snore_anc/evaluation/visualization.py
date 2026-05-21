import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ANCVisualizer:
    @staticmethod
    def plot_time_domain(d_signal, e_signal, fs=16000, title="ANC Time Domain", save_path=None):
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
        win_samples = int(window_ms / 1000 * fs)
        n_windows = len(e_signal) // win_samples
        time_axis, nr_values = [], []
        for i in range(n_windows):
            start, end = i * win_samples, (i + 1) * win_samples
            rms_d = np.sqrt(np.mean(d_signal[start:end] ** 2)) + 1e-12
            rms_e = np.sqrt(np.mean(e_signal[start:end] ** 2)) + 1e-12
            time_axis.append((start + end) / 2 / fs)
            nr_values.append(20 * np.log10(rms_d / rms_e))
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
        ANCVisualizer.plot_time_domain(d_signal, e_signal, fs, save_path=f"{prefix}_waveform.png")
        ANCVisualizer.plot_convergence(d_signal, e_signal, fs, save_path=f"{prefix}_convergence.png")
