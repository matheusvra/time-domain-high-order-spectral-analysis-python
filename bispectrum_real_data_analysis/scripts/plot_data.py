import pandas as pd
import plotly.express as px
import os
from plotly.subplots import make_subplots


"""
This function plots the data from the csv file generated by the script generate_bispectrum_of_data.py.
To filter one channel, double click one of the lines in the legend. then, select the other corresponding line (if you
clicked in the amplitude, select the phase line and vice versa). All the 16 channels are plotted in the same figure.
The plot in the first row is the amplitude and the plot in the second row is the phase.
"""

if __name__ == "__main__":

    # Select which data to plot
    data_to_plot = "bispectrum_0_to_60_by_0.1.csv"

    BASE_PATH = os.getcwd() + "/bispectrum_real_data_analysis/data"

    df = pd.read_csv(f"{BASE_PATH}/{data_to_plot}", delimiter=',', encoding="utf8")

    amplitudes = df.iloc[:, 1:17]
    phases = df.iloc[:, 17:33]
    fig = make_subplots(rows=3, cols=1)
    amplitudes = px.line(df, x=df.iloc[:, 0], y=amplitudes.columns)
    phases = px.line(df, x=df.iloc[:, 0], y=phases.columns)
    

    fig = make_subplots(rows=2, cols=1, shared_xaxes=False)
    for amplitude in amplitudes['data']:
        fig.add_trace(amplitude, row=1, col=1)
    for phase in phases['data']:
        fig.add_trace(phase, row=2, col=1)
    fig.show()