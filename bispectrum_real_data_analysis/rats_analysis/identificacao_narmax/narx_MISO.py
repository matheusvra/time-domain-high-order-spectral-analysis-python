from bispectrum_real_data_analysis.rats_analysis.identificacao_narmax.methods import (
    butter_bandpass,
    filter_function,
    get_events,
    get_time_given_time_sampling_and_N,
    select_event_window,
    decimate
)


from err import err
from correlation import Correlation

# Importing Python modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.signal import hilbert
from time import perf_counter, strftime, gmtime, sleep
import plotly.figure_factory as ff
from copy import deepcopy
import plotly.express as px
from loguru import logger
import os
import mat73
from bispectrum_real_data_analysis.scripts.utils import seconds_to_formatted_time
from plotly.subplots import make_subplots
from bispectrum_real_data_analysis.scripts.utils import standardize_array
import pendulum
from scipy import stats as st
import seaborn as sns
import re
import matplotlib
from scipy.linalg import block_diag
from numpy.linalg import inv


if __name__ == "__main__":
    BASE_PATH = "/home/matheus/Documents/repositories/bispectrum_real_data_analysis/bispectrum_real_data_analysis/data/"
    file_to_load: str = "F04__T_data.mat"
    logger.info(f"Loading .mat file...")

    complete_path: str = f"{BASE_PATH}/{file_to_load}"

    complete_data = mat73.loadmat(complete_path)

    logger.success("Loaded!")

    data = complete_data["data"]["data"][0]

    FrequencySampling = complete_data["parameters"]["srate"].item()
    TimeSampling = 1.0 / FrequencySampling

    columns = [
        "CS_modulating",
        "mPFC_pre_limbic_1",
        "mPFC_pre_limbic_2",
        "mPFC_infra_limbic_1",
        "mPFC_infra_limbic_2",
        "Hippocampus_CA1",
        "Hippocampus_MOL_layer",
        "Hippocampus_GD_1",
        "Hippocampus_GD_2",
        "Amygdala_lateral_1",
        "Amygdala_lateral_2",
        "Amygdala_basolateral_1",
        "Amygdala_basolateral_2",
        "Inferior_colliculus_1",
        "Inferior_colliculus_2",
        "Inferior_colliculus_3",
        "Inferior_colliculus_4",
    ]

    data = pd.DataFrame(data=data.T, columns=columns)

    time = get_time_given_time_sampling_and_N(
        time_sampling=TimeSampling, N=data.shape[0], start_in_seconds=10
    )

    data = data.assign(Time=time)

    threshold = 0.025
    window_size = 100

    data = get_events(
        data=data,
        threshold=threshold,
        window_size=window_size,
        time_sampling=TimeSampling,
        plot_events=False,
    )

    low_cut_hz = 50
    high_cut_hz = 60
    filter_order = 4
    b, a = butter_bandpass(
        low_cut_hz, high_cut_hz, fs=FrequencySampling, order=filter_order
    )

    for channel_number in range(1, 5):
        logger.info(f"Filtering the Inferior colliculus {channel_number}")
        data = data.assign(
            **{
                f"filtered_Inferior_colliculus_{channel_number}": filter_function(
                    data.loc[:, f"Inferior_colliculus_{channel_number}"].to_numpy(),
                    low_cut_hz=low_cut_hz,
                    high_cut_hz=high_cut_hz,
                    fs=FrequencySampling,
                    filter_order=4,
                )
            }
        )

    for channel_number in range(1, 3):
        logger.info(f"Filtering the Amygdala lateral {channel_number}")
        data = data.assign(
            **{
                f"filtered_Amygdala_lateral_{channel_number}": filter_function(
                    data.loc[:, f"Amygdala_lateral_{channel_number}"].to_numpy(),
                    low_cut_hz=low_cut_hz,
                    high_cut_hz=high_cut_hz,
                    fs=FrequencySampling, 
                    filter_order=4
                )
            }
        )

        logger.info(f"Filtering the Amygdala basolateral {channel_number}")
        data = data.assign(
            **{
                f"filtered_Amygdala_basolateral_{channel_number}": filter_function(
                    data.loc[:, f"Amygdala_basolateral_{channel_number}"].to_numpy(),
                    low_cut_hz=low_cut_hz,
                    high_cut_hz=high_cut_hz,
                    fs=FrequencySampling, 
                    filter_order=4
                )
            }
        )

    logger.success(f"Done filtering!")

    desired_frequency_sampling = 150

    data, TimeSampling, FrequencySampling = decimate(
        data, desired_frequency_sampling=desired_frequency_sampling
    )

    event_number = 1

    event_data = select_event_window(
        df=data, event_name=f"event_{event_number}", samples_before=0, samples_after=0
    )

    y = event_data.Inferior_colliculus_2.to_numpy()

    u0 = event_data.CS_modulating.to_numpy()
    u1 = event_data.Inferior_colliculus_1.to_numpy()
    u2 = event_data.Inferior_colliculus_3.to_numpy()
    u3 = event_data.Inferior_colliculus_4.to_numpy()
    u4 = event_data.Amygdala_lateral_1.to_numpy()
    u5 = event_data.Amygdala_lateral_2.to_numpy()
    u6 = event_data.Amygdala_basolateral_1.to_numpy()
    u7 = event_data.Amygdala_basolateral_2.to_numpy()

    u = np.vstack([u0, u6, u7])

    degree_of_non_linearity: int = 1
    max_y_delays: int = 5
    max_u_delays: int = np.array([5] * u.shape[0])

    max_delays = np.max([max_y_delays] + max_u_delays)
    err_obj = err(
        ny=max_y_delays,
        nu=max_u_delays,
        n_lin=degree_of_non_linearity,
        yid=y,
        uid=u,
        cte=True,
    )

    start_time = perf_counter()

    err_out, termos, psi = err_obj.run(print_result=True)

    end_time = perf_counter()
    
    print(termos.columns)
