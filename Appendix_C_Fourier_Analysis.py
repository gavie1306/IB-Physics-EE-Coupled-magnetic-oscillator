"""
Fourier analysis and peak identification for the five angular trajectories.

Input:
    angular_displacement.csv
Output:
    experimental_fft_peaks.csv

The script calculates sampling frequency, Nyquist frequency, FFT-bin
spacing, one-sided amplitude spectra, and candidate peaks satisfying
a 3:1 signal-to-noise criterion.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

INPUT_FILE = "angular_displacement.csv"
OUTPUT_FILE = "experimental_fft_peaks.csv"

F_MIN = 0.10
F_MAX = 4.00
SNR_THRESHOLD = 3.0


def one_sided_fft(time, signal):
    dt = float(np.mean(np.diff(time)))
    n = len(signal)

    signal = signal - np.mean(signal)
    transform = np.fft.rfft(signal)
    frequency = np.fft.rfftfreq(n, d=dt)

    amplitude = 2.0 * np.abs(transform) / n
    amplitude[0] *= 0.5
    if n % 2 == 0:
        amplitude[-1] *= 0.5

    return frequency, amplitude


def main():
    data = pd.read_csv(INPUT_FILE)
    time = data["time"].to_numpy()

    dt = float(np.mean(np.diff(time)))
    fs = 1.0 / dt
    nyquist = fs / 2.0
    n = len(time)
    duration = n * dt
    delta_f = fs / n

    print(f"Sampling interval: {dt:.8f} s")
    print(f"Sampling frequency: {fs:.3f} Hz")
    print(f"Nyquist frequency: {nyquist:.3f} Hz")
    print(f"Recording duration: {duration:.3f} s")
    print(f"FFT-bin spacing: {delta_f:.5f} Hz")

    rows = []
    fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True)

    for i, ax in enumerate(axes, start=1):
        signal = data[f"q{i}_rad"].to_numpy()
        frequency, amplitude = one_sided_fft(time, signal)

        band = (frequency >= F_MIN) & (frequency <= F_MAX)
        f_band = frequency[band]
        a_band = amplitude[band]

        noise_floor = float(np.median(a_band))

        peaks, _ = find_peaks(a_band, prominence=noise_floor)

        for p in peaks:
            peak_amp = float(a_band[p])
            snr = peak_amp / noise_floor if noise_floor > 0 else np.inf

            if snr >= SNR_THRESHOLD:
                rows.append(
                    {
                        "Magnet": i,
                        "Frequency_Hz": float(f_band[p]),
                        "Amplitude": peak_amp,
                        "Noise_floor": noise_floor,
                        "SNR": snr,
                    }
                )

        ax.plot(f_band, a_band)
        ax.set_ylabel(f"|Q{i}(f)|")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Frequency, f (Hz)")
    fig.tight_layout()
    plt.show()

    peaks_df = pd.DataFrame(rows)
    if not peaks_df.empty:
        peaks_df = peaks_df.sort_values(["Frequency_Hz", "Magnet"])

    peaks_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
