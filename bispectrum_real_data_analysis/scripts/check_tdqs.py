import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from high_order_spectra_analysis.hosa.hosa import Tdhosa


if __name__ == "__main__":

    for norm_before, norm_after in [(False, False), (True, False), (False, True), (True, True)]:

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

        signal = clean_signal.astype(dtype)*1e-5

        # adding noise
        signal += np.random.normal(0, 0.5*signal.std(), size=signal.shape).astype(dtype)

        if norm_before:
            signal = (signal - signal.min())/(signal.max() - signal.min())
            signal -= signal.mean()

        frequency_array_to_scan = np.arange(0, 100, 0.01, dtype=dtype)

        tdqs_object = Tdhosa(
            frequency_sampling=fs,
            frequency_array=frequency_array_to_scan,
            phase_step=1
        )

        (
            frequency_array, 
            spectrum, 
            phase_spectrum, 
            bispectrum, 
            phase_bispectrum, 
            trispectrum, 
            phase_trispectrum, 
            tetraspectrum, 
            phase_tetraspectrum
        ) = tdqs_object.run_tdqs(signal=signal)

        max_freq_plot = 70

        if norm_after:
            spectrum = (spectrum - spectrum.min())/(spectrum.max() - spectrum.min())
            bispectrum = (bispectrum - bispectrum.min())/(bispectrum.max() - bispectrum.min())
            trispectrum = (trispectrum - trispectrum.min())/(trispectrum.max() - trispectrum.min())
            tetraspectrum = (tetraspectrum - tetraspectrum.min())/(tetraspectrum.max() - tetraspectrum.min())

        fig = make_subplots(rows=4, cols=1)

        fig.update_layout(
            font_family="Courier New",
            font_color="blue",
            title_font_family="Times New Roman",
            title_font_color="black",
            legend_title_font_color="green",
            title=f"Normalized before: {norm_before}, Normalized after: {norm_after}"
        )

        fig.append_trace(go.Scatter(
            x=frequency_array[frequency_array <= max_freq_plot],
            y=spectrum[frequency_array <= max_freq_plot],
            name="Spectrum",
        ), row=1, col=1)

        fig.append_trace(go.Scatter(
            x=frequency_array[frequency_array <= max_freq_plot],
            y=bispectrum[frequency_array <= max_freq_plot],
            name="Bispectrum",
        ), row=2, col=1)

        fig.append_trace(go.Scatter(
            x=frequency_array[frequency_array <= max_freq_plot],
            y=trispectrum[frequency_array <= max_freq_plot],
            name="Trispectrum",
        ), row=3, col=1)

        fig.append_trace(go.Scatter(
            x=frequency_array[frequency_array <= max_freq_plot],
            y=tetraspectrum[frequency_array <= max_freq_plot],
            name="Tetraspectrum",
        ), row=4, col=1)

        fig.show()
