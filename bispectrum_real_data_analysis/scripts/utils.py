import os
import time
import numpy as np
import pandas as pd
from loguru import logger
from scipy import signal

def get_cpus_available() -> int:
    """Function to get the number of CPUs available on the system.

    Returns:
        int: number of CPUs available on the system
    """
    return max(1, len(os.sched_getaffinity(0)))

def seconds_to_formatted_time(seconds: float) -> str:
    return "".join(
        [
            f"{value}{unit}" for value, unit in zip(
                time.strftime('%H:%M:%S', time.gmtime(seconds)
                ).split(":"), ["h", "m", "s"]) if value != "00"
        ]
    )

def standardize_array(array: np.ndarray, scale_to_unit: bool = False) -> np.ndarray:
    """Function to standardize an array.

    Args:
        array (np.ndarray): array to be standardized

    Returns:
        np.ndarray: standardized array
    """
    if scale_to_unit:
        standardized_array =  (array - array.min())/(array.max() - array.min())
    
    else:
        standardized_array = (array - array.mean()) / array.std()

    return standardized_array


def decimate(
    data: pd.DataFrame, 
    desired_frequency_sampling: float, 
    FrequencySampling: float | None = None, 
    filter_antialiasing: bool = True, 
    time: np.ndarray | None = None,
    columns_to_exclude: list | None = None,
    logger = logger
):
        
    if FrequencySampling is None:
        if time is None:
            time = data.Time.to_numpy()
        TimeSampling = round(np.mean(time[1:] - time[:-1]), 6)
        FrequencySampling = 1.0/TimeSampling
    else:
        TimeSampling = 1.0/FrequencySampling
        
    logger.info(f"The time sampling is {TimeSampling} seconds and the frequency is "
        f"{FrequencySampling/float(1000**(FrequencySampling<=1000))} {'k'*bool(FrequencySampling>=1000)}Hz")

    newTimeSampling = 1.0/desired_frequency_sampling
    decimation_rate = np.ceil(newTimeSampling/TimeSampling).astype(int)
    logger.info(f"The data will be decimated by the rate 1:{decimation_rate}")

    if filter:
        if columns_to_exclude:
            matrix = data.loc[:, ~data.columns.isin(columns_to_exclude)].to_numpy()
        else:
            matrix = data.to_numpy()
        decimated_matrix = signal.decimate(matrix, decimation_rate, axis=0, ftype='fir', zero_phase=True)
        new_data = data.copy()[::decimation_rate]
        if columns_to_exclude:
            new_data.loc[:, ~data.columns.isin(columns_to_exclude)] = decimated_matrix
        else:
            new_data.iloc[:] = decimated_matrix
    else:
        new_data = data[::decimation_rate]

    TimeSampling = TimeSampling*decimation_rate
    
    FrequencySampling = 1.0/TimeSampling
    logger.info(f"The new time sampling is {np.round(TimeSampling, 5)} s and the new frequency is "
    f"{FrequencySampling/float(1000**(FrequencySampling>=1000))} {'k'*bool(FrequencySampling>=1000)}Hz")
    
    return new_data, TimeSampling, FrequencySampling