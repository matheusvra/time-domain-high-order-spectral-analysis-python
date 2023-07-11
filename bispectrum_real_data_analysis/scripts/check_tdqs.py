import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from high_order_spectra_analysis.hosa.hosa import Tdhosa
from matplotlib import pyplot as plt

if __name__ == "__main__":

    for norm_before, norm_after in [(False, False), (True, False), (False, True), (True, True)]:

        dtype = np.float32
        time_step = 0.003
        fs = 1/time_step
        time = np.arange(0, 5, time_step, dtype=dtype)

        freqs = np.array([5, 12, 29, 61], dtype=dtype)
        f1, f2, f3, f4 = tuple(freqs)
        w1, w2, w3, w4 = tuple(2*np.pi*freqs)
        gains = np.array([1.05, 1.15, 0.7, 0.93], dtype=dtype)
        clean_signal = np.cos((w2 + w3)*time) # add frequency coupling
        clean_signal += np.cos((w1 + w2 + w3)*time) # add frequency coupling

        for freq, gain in zip(freqs, gains):
            clean_signal += gain*np.cos(2*np.pi*freq*time)

        signal = clean_signal.astype(dtype)*1e-5

        # adding noise
        # signal += np.random.normal(0, 1*signal.std(), size=signal.shape).astype(dtype)

        if norm_before:
            signal = (signal - signal.min())/(signal.max() - signal.min())
            signal -= signal.mean()

        frequency_array_to_scan = np.arange(0, 70, 0.01, dtype=dtype)

        tdqs_object = Tdhosa(
            frequency_sampling=fs,
            frequency_array=frequency_array_to_scan,
            phase_step=0.5
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

        x_ticks: dict = dict(zip(np.append(freqs, [f2+f3, f1+f2+f3]), [f"w{i}" for i in range(1, len(freqs)+1)] + ["w2+w3", "w1+w2+w3"]))

        # sort dict by key
        x_ticks = dict(sorted(x_ticks.items(), key=lambda item: item[0]))

        plt.figure(figsize=(20, 15))
                
        plt.subplot(411)
        plt.plot(
            frequency_array[frequency_array <= max_freq_plot],
            spectrum[frequency_array <= max_freq_plot],
        )
        plt.ylabel("Spectrum")
        plt.xticks(list(x_ticks.keys()), list(x_ticks.values()))
        plt.xlim(0, max_freq_plot)
        
        plt.subplot(412)
        plt.plot(
            frequency_array[frequency_array <= max_freq_plot],
            bispectrum[frequency_array <= max_freq_plot],
        )
        plt.ylabel("Bispectrum")
        plt.xticks(list(x_ticks.keys()), x_ticks.values())
        plt.xlim(0, max_freq_plot)
        
        plt.subplot(413)
        plt.plot(
            frequency_array[frequency_array <= max_freq_plot],
            trispectrum[frequency_array <= max_freq_plot],
        )
        plt.ylabel("Trispectrum")
        plt.xticks(list(x_ticks.keys()), x_ticks.values())
        plt.xlim(0, max_freq_plot)
        
        plt.subplot(414)
        plt.plot(
            frequency_array[frequency_array <= max_freq_plot],
            tetraspectrum[frequency_array <= max_freq_plot],
        )
        plt.ylabel("Tetraspectrum")
        plt.xticks(list(x_ticks.keys()), list(x_ticks.keys()))
        plt.xlim(0, max_freq_plot)
        
        plt.savefig(f"tdqs_validation_clean.pdf", format="pdf")
        
        plt.show()
            
        break
    
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
        
        break
