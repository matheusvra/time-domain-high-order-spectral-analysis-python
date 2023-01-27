import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from high_order_spectra_analysis.time_domain_bispectrum.tdbs import tdbs


if __name__ == "__main__":

    dtype = np.float32
    time_step = 0.003
    fs = 1/time_step
    time = np.arange(0, 5, time_step, dtype=dtype)

    freqs = np.array([11, 17, 21, 45], dtype=dtype)
    w2, w4 = tuple(2*np.pi*np.array(freqs[1::2]))
    phases =np.array([np.pi/20, np.pi/30, np.pi/6, np.pi/15], dtype=dtype)
    phi2, phi4 = phases[1], phases[3]
    gains = np.array([0.7, 1.15, 1.05, 0.93], dtype=dtype)

    clean_signal = np.cos((w2 + w4)*time + (phi2 + phi4))

    for freq, phase, gain in zip(freqs, phases, gains):
        clean_signal += gain*np.cos(2*np.pi*freq*time + phase)

    signal = clean_signal.astype(dtype)

    frequency_array, spectrum, phase_spectrum, bispectrum, phase_bispectrum = tdbs(
        signal=signal,
        frequency_sampling=fs,
        time=None,
        fmin=None,
        fmax=None,
        freq_step=0.05,
        phase_step=0.05,
        dtype=dtype
    )

    max_freq_plot = 70

    fig = make_subplots(rows=3, cols=1)

    
    fig.append_trace(go.Scatter(
        x=frequency_array[frequency_array <= max_freq_plot],
        y=spectrum[frequency_array <= max_freq_plot],
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=frequency_array[frequency_array <= max_freq_plot],
        y=bispectrum[frequency_array <= max_freq_plot],
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=frequency_array[frequency_array <= max_freq_plot],
        y=phase_bispectrum[frequency_array <= max_freq_plot],
    ), row=3, col=1)

    fig.show()