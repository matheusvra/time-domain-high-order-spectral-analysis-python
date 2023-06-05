import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from loguru import logger
import numpy as np
import plotly.graph_objects as go
import re

"""
This function plots the data from the csv file generated by the script generate_bispectrum_of_data.py.
To filter one channel, double click one of the lines in the legend. then, select the other corresponding line (if you
clicked in the amplitude, select the phase line and vice versa). All the 16 channels are plotted in the same figure.
The plot in the first row is the amplitude and the plot in the second row is the phase.
"""

if __name__ == "__main__":
        
    # If you want to filter the data, change BOTH the values of low_lim and high_lim.
    low_lim: float | None = None
    high_lim: float | None = None

    # Do not change the code below
    # ---------------------------------------------------------------------------------------------
    
    band_plotted: str = "full" if low_lim is None or high_lim is None else f"{low_lim}-{high_lim}Hz"
    
    BASE_PATH = "/".join(__file__.split("/")[:-1])

    group, number = re.findall(r'\d+', __file__.split("/")[-2])
    
    group_number, rat_number = int(group), int(number)
    
    logger.info(f"Group: {group_number}, Rat Number: {rat_number}")

    id_results: str = f"rato-{rat_number}-grupo-{group_number}"
    path_files = f'{BASE_PATH}/csv/hosa_{id_results}.csv'

    try:
        hosa_df = pd.read_csv(path_files)
    except FileNotFoundError as e:
        raise NameError(f"File hosa_{id_results}.csv not found in {BASE_PATH}. Check if the file exists.") from e


    is_amp_exp = np.vectorize(lambda x, exp: ("amp" in x) and (exp in x), excluded=["exp"])

    frequency = x = hosa_df.loc[:, "frequency"].to_numpy()

    if low_lim is not None or high_lim is not None:
        logger.info(f"Filtering from {low_lim} to {high_lim} Hz")
        df_amps_train = hosa_df.loc[(x >= low_lim)&(x<=high_lim), is_amp_exp(hosa_df.columns, exp="train")]
        df_amps_test = hosa_df.loc[(x>=low_lim)&(x<=high_lim), is_amp_exp(hosa_df.columns, exp="test")]

        frequency = x[(x>=low_lim)&(x<=high_lim)]
    else:
        logger.info(f"Plotting data from {frequency.min().round(2)} to {frequency.max().round(2)} Hz")
        df_amps_train = hosa_df.loc[:, is_amp_exp(hosa_df.columns, exp="train")]
        df_amps_test = hosa_df.loc[:, is_amp_exp(hosa_df.columns, exp="test")]
    
    df_amps_train = df_amps_train.rename(columns=lambda x: x.replace("train_", "").replace("test_", ""))
    df_amps_test = df_amps_test.rename(columns=lambda x: x.replace("train_", "").replace("test_", ""))
    
    for plot_mean in (False, True):
        for exp, amplitudes_df in zip(["train", "test"], [df_amps_train, df_amps_test]):

            fig = make_subplots(rows=4, cols=1, subplot_titles=('Spectrum',  'Bispectrum', 'Trispectrum', 'Quadrispectrum'))

            fig.update_layout(
                font_family="Courier New",
                font_color="blue",
                title_font_family="Times New Roman",
                title_font_color="black",
                legend_title_font_color="green",
                title=f"TDHOSA Amplitude - {exp} - Group {group_number} Rat {rat_number}{' - MEAN'*plot_mean}",
            )

            amplitudes = px.line(amplitudes_df, x=frequency, y=amplitudes_df.columns)
            
            if plot_mean:
                tds = np.zeros(len(amplitudes['data'][0].y))
                tdbs = tds.copy()
                tdts = tds.copy()
                tdqs = tds.copy()
                frequency = amplitudes['data'][0].x
                for amplitude in amplitudes['data']:
                    if amplitude.legendgroup.startswith("tds"):
                        tds += amplitude.y
                    elif amplitude.legendgroup.startswith("tdbs"):
                        tdbs += amplitude.y
                    elif amplitude.legendgroup.startswith("tdts"):
                        tdts += amplitude.y
                    elif amplitude.legendgroup.startswith("tdqs"):
                        tdqs += amplitude.y
                
                fig.add_trace(go.Scatter(x=frequency, y=tds/5, name="avg_spectrum"), row=1, col=1)
                fig.add_trace(go.Scatter(x=frequency, y=tdbs/5, name="avg_bispectrum"), row=2, col=1)
                fig.add_trace(go.Scatter(x=frequency, y=tdts/5, name="avg_trispectrum"), row=3, col=1)
                fig.add_trace(go.Scatter(x=frequency, y=tdqs/5, name="avg_quadrispectrum"), row=4, col=1)
                
            else:

                for amplitude in amplitudes['data']:
                    i = 1*(amplitude.legendgroup.startswith("tds")) + \
                    2 * (amplitude.legendgroup.startswith("tdbs")) + \
                    3 * (amplitude.legendgroup.startswith("tdts")) + \
                    4 * (amplitude.legendgroup.startswith("tdqs"))

                    fig.add_trace(amplitude, row=i, col=1)
                    
            fig.write_html(f"{BASE_PATH}/html/TDHOSA_{exp}_group_{group_number}_rat_{rat_number}_{band_plotted}{'_mean'*plot_mean}")
            fig.show()

    logger.success("Done!")