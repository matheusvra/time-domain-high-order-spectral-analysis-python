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

fontsize = 15
matplotlib.rc('xtick', labelsize=fontsize) 
matplotlib.rc('ytick', labelsize=fontsize) 
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': fontsize})

# Method to extract events from dataframe

def select_event_window(
    df: pd.DataFrame, 
    event_name: str, 
    samples_before: int = 0, 
    samples_after: int = 0
) -> pd.DataFrame:
    """
    Method to extract the slice of the dataframe which contais the event, with some data before and after, 
    given number of samples to add to the begin and end, respectively.
    """

    window_index = np.argwhere(df.event.to_numpy() == event_name).flatten()
    begin_index = window_index[0] - samples_before
    end_index = window_index[-1] + samples_after
    return df[begin_index:end_index]


def decimate(data: pd.DataFrame, desired_frequency_sampling: float, filter: bool = False, time=None):
    if time is None:
        time = data.Time.to_numpy()
    TimeSampling = round(np.mean(time[1:] - time[:-1]), 6)
    FrequencySampling = 1.0/TimeSampling
    logger.info(
        f"The time sampling is {TimeSampling} seconds and the frequency is {FrequencySampling / float(1000**(FrequencySampling <= 1000))} {'k' * (FrequencySampling >= 1000)}Hz"
    )

    newTimeSampling = 1.0/desired_frequency_sampling
    decimation_rate = np.ceil(newTimeSampling/TimeSampling).astype(int)
    logger.info(f"The data will be decimated by the rate 1:{decimation_rate}")

    if filter:
        matrix = data.iloc[:, 1:-2].to_numpy()
        decimated_matrix = signal.decimate(matrix, decimation_rate, axis=0, ftype='fir', zero_phase=True)
        new_data = data.copy()[::decimation_rate]
        new_data.iloc[:, 1:-2] = decimated_matrix
    else:
        new_data = data[::decimation_rate]

    TimeSampling = TimeSampling*decimation_rate

    FrequencySampling = 1.0/TimeSampling
    logger.info(
        f"The new time sampling is {np.round(TimeSampling, 5)} s and the new frequency is {FrequencySampling / float(1000**(FrequencySampling >= 1000))} {'k' * (FrequencySampling >= 1000)}Hz"
    )

    return new_data, TimeSampling, FrequencySampling

def moving_average(x, N):
    return np.convolve(x, np.ones(N), 'same') / N

def get_events(data, threshold, window_size, time_sampling):
    
    x = data.CS_modulating.to_numpy()
    N = len(x)
    index = np.arange(N)
    plt.figure(figsize=(16,14))
    _extracted_from_get_events_7(321, index, x, "x")
    x = x - np.mean(x[:10])
    _extracted_from_get_events_7(322, index, x, "x - mean(x)")
    x = x**2
    _extracted_from_get_events_7(323, index, x, "(x - mean(x))^2")
    x = moving_average(x, window_size)
    plt.subplot(324)
    plt.plot(index, x)
    plt.axhline(threshold, color="red", label="threshold")
    plt.legend(loc='upper right')
    plt.ylabel("moving_average_10_(x - mean(x))^2")

    x[x>threshold] = 1
    x[x<threshold] = 0

    _extracted_from_get_events_7(
        325, index, x, "threshold(moving_average_10_(x - mean(x))^2)"
    )
    indices = index[np.append(False, x[1:] - x[:-1]) != 0]

    for event, s, e in zip(range(1, 6), indices[::2], indices[1::2]):
        print(f"\nEvent: {event}")
        print(f"start: {s}\nend: {e}") 
        print(f"time duration: {(e-s)*time_sampling}")

    print(f"\nlen(indices) = {len(indices)}")

    data = data.assign(event=np.empty(len(data), dtype=str))
    data.loc[:, "event"] = "base"

    for i, event in zip(range(0, len(indices), 2), np.arange(1, 6)):
        start = indices[i]
        end = indices[i+1]
        data.loc[start:end, "event"] = f"event_{event}"

    data.event.unique()

    plt.subplot(326)
    for event in data.event.unique():
        plt.plot(data.loc[data.event==event, "Time"], data.loc[data.event==event, "CS_modulating"], label=event)

    plt.ylabel("events")

    plt.show()
    return data


# TODO Rename this here and in `get_events`
def _extracted_from_get_events_7(arg0, index, x, arg3):
    plt.subplot(arg0)
    plt.plot(index, x)
    plt.ylabel(arg3)

def AIC(n_theta, N, var_xi):
    """
    AIC(n_theta) = N ln[var(Xi(n_theta))] + 2 n_theta
    """
    return N*np.log(var_xi) + 2*n_theta

def MQ(Psi, y):
    theta = inv(Psi.T@Psi)@Psi.T@y
    residuos = y - Psi@theta
    return theta, residuos



if __name__ == '__main__':
    
    fig = 1
    group_number, rat_number = 5, 1
    print(f"Group: {group_number}, Rat Number: {rat_number}")
    
    
    # Loading and creating the data matrix

    BASE_PATH = "/home/matheus/Documents/repositories/bispectrum_real_data_analysis/bispectrum_real_data_analysis/data/rats"    

    logger.info(f"Loading .mat files for group {group_number} rat {rat_number}...")

    data_train_filename = f"{BASE_PATH}/G{group_number}-R{rat_number}_PreTreino.mat"
    data_test_filename = f"{BASE_PATH}/G{group_number}-R{rat_number}_Salina.mat"

    raw_data_train = mat73.loadmat(data_train_filename)
    raw_data_test = mat73.loadmat(data_test_filename)

    logger.success("Creating data matrix...")

    TimeSamplingTrain = 1.0/raw_data_train["srate"]
    TimeSamplingTest = 1.0/raw_data_test["srate"]

    N = raw_data_train["data"].shape[1]
    M = raw_data_test["data"].shape[1]

    timeTrain = np.arange(0, TimeSamplingTrain*N, TimeSamplingTrain)[:N]
    timeTest = np.arange(0, TimeSamplingTest*M, TimeSamplingTest)[:M]

    data_train = pd.DataFrame(
        {
            "Time": timeTrain,
            "CS_modulating": raw_data_train["data"][0],
            "Inferior_colliculus": raw_data_train["data"][1]
        }
    ) 

    data_test = pd.DataFrame(
        {
            "Time": timeTest,
            "CS_modulating": raw_data_test["data"][0],
            "Inferior_colliculus": raw_data_test["data"][1]
        }
    ) 

    logger.success("Done!")
    
    threshold = 0.025
    window_size = 100

    data_train = get_events(
        data=data_train, 
        threshold=threshold, 
        window_size=window_size, 
        time_sampling=TimeSamplingTrain
    )
    
    desired_frequency_sampling = 150

    data_train, TimeSampling, FrequencySampling = decimate(data_train, desired_frequency_sampling=desired_frequency_sampling)
    
    plt.figure(figsize=(14,12))

    plt.subplot(211)
    for event in data_train.event.unique():
        plt.plot(data_train.loc[data_train.event==event, "Time"], data_train.loc[data_train.event==event, "CS_modulating"], label=event)
    plt.legend(loc='lower left', prop={'size': 20})
    plt.ylabel("CS modulating")

    plt.subplot(212)
    for event in data_train.event.unique():
        plt.plot(data_train.loc[data_train.event==event, "Time"], data_train.loc[data_train.event==event, "Inferior_colliculus"], label=event)
    plt.xlabel("time [s]")
    plt.ylabel("Inferior Colliculus")
    plt.legend(loc='lower left', prop={'size': 20})

    plt.show()
    
    threshold = 0.025
    window_size = 100

    data_test = get_events(
        data=data_test, 
        threshold=threshold, 
        window_size=window_size, 
        time_sampling=TimeSamplingTest
    )
    
    data_test, TimeSampling, FrequencySampling = decimate(data_test, desired_frequency_sampling=desired_frequency_sampling)
    
    plt.figure(figsize=(14,12))

    plt.subplot(211)
    for event in data_train.event.unique():
        plt.plot(data_test.loc[data_test.event==event, "Time"], data_test.loc[data_test.event==event, "CS_modulating"], label=event)
    plt.legend(loc=6, prop={'size': 20})
    plt.ylabel("CS modulating")

    plt.subplot(212)
    for event in data_test.event.unique():
        plt.plot(data_test.loc[data_test.event==event, "Time"], data_test.loc[data_test.event==event, "Inferior_colliculus"], label=event)
    plt.xlabel("time [s]")
    plt.ylabel("Inferior Colliculus")
    plt.legend(prop={'size': 20})
    plt.ylim([-1.5e-3, 1.5e-3])
    plt.savefig(
        "test_waveform.jpg",
        format="jpg",
        bbox_inches='tight', 
        dpi=150
    )
    plt.show()
    
    data = data_test
    
    event_number = 2

    event_data = select_event_window(
        df=data, 
        event_name=f"event_{event_number}", 
        samples_before=0, 
        samples_after=0
    )

    y = event_data.Inferior_colliculus.to_numpy()

    y = (y - y.min())/(y.max()-y.min())
    y -= y.mean()
    u = event_data.CS_modulating.to_numpy()
    
    degree_of_non_linearity: int = 4
    max_y_delays: int = 2
    max_u_delays: int = 2


    err_obj = err(
        ny=max_y_delays,
        nu=max_u_delays,
        n_lin=degree_of_non_linearity,
        yid=y,
        uid=u,
        cte=True
    )

    start_time = perf_counter()

    err_out, termos, psi = err_obj.run(print_result=True)

    end_time = perf_counter()
    
    Phi = np.array([])

    ordery = 0
    orderu = 0

    aic = np.array([])
    parameters = np.array([])

    for i in range(len(psi.T)):
        
        if i == 0:
            Phi = psi[:,err_out["ordem"][i]][None]
        else:
            Phi = np.vstack([Phi, psi[:,err_out["ordem"][i]]])  
                
        thetaMQ, residuos = MQ(Phi.T, y[max_y_delays:])
        
        var_xi = np.var(residuos)
        
        n_theta = len(thetaMQ)
        
        aic = np.append(aic, AIC(n_theta, len(y), var_xi))
        
        parameters = np.append(parameters, i+1)
        
    plt.plot(parameters, aic)
    plt.title("Critério de Informação de Akaike")
    xlabel = "# parâmetros"
    caption = f"Figura {fig}: Simulação livre para validação do modelo não-linear estimado "; fig +=1
    plt.xlabel(xlabel+"\n"+caption)
    plt.ylabel("AIC($n_{\\theta}$)")
    plt.xticks(parameters, rotation=90, ha='right', fontsize=10)
    plt.grid()
    plt.show()