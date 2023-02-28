import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from high_order_spectra_analysis.time_domain_trispectrum.tdts import tdts



if __name__ == "__main__":

    dtype = np.float32
    time_step = 0.003
    fs = 1/time_step
    time = np.arange(0, 5, time_step, dtype=dtype)

    freqs = np.array([29, 12, 5, 61], dtype=dtype)
    w1, w2, w3, w4 = tuple(2*np.pi*freqs)
    gains = np.array([0.7, 1.15, 1.05, 0.93], dtype=dtype)
    clean_signal = np.cos((w1 + w2)*time) + np.cos((w1 + w2 + w3)*time)

    for freq, gain in zip(freqs, gains):
        clean_signal += gain*np.cos(2*np.pi*freq*time)

    signal = clean_signal.astype(dtype)

    (
        frequency_array, 
        spectrum, 
        phase_spectrum, 
        bispectrum, 
        phase_bispectrum, 
        trispectrum, 
        phase_trispectrum
    ) = tdts(
        signal=signal,
        frequency_sampling=fs,
        time=None,
        fmin=None,
        fmax=None,
        freq_step=0.2,
        phase_step=0.01,
        dtype=dtype
    )

    max_freq_plot = 70

    fig = make_subplots(rows=4, cols=1)

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
        y=trispectrum[frequency_array <= max_freq_plot],
    ), row=3, col=1)

    fig.append_trace(go.Scatter(
        x=frequency_array[frequency_array <= max_freq_plot],
        y=phase_trispectrum[frequency_array <= max_freq_plot],
    ), row=4, col=1)

    fig.show()
