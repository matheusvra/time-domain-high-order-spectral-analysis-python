import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from high_order_spectra_analysis.time_domain_bispectrum.tdbs import tdbs
from pathos.multiprocessing import ProcessingPool as Pool
import os


def load_data():
    BASE_PATH = os.getcwd() + "/bispectrum_real_data_analysis/data"
    data = pd.read_csv(f"{BASE_PATH}/data_matrix.csv", delimiter=',', encoding="utf8")
    events_index = pd.read_csv(f"{BASE_PATH}/events_index.csv", delimiter=',', encoding="utf8")
    events_index_data_array = np.full((len(data),), None)

    for start, end, event_idx in zip(events_index.start, events_index.end, np.arange(1, len(events_index))):
        events_index_data_array[start:end] = event_idx
        
    data = data.assign(events_index=events_index_data_array)
    events_index_timestamp = pd.read_csv(f"{BASE_PATH}/events_index_timestamp.csv", delimiter=',', encoding="utf8")
    events_behavior_TS_LFP_index = pd.read_csv(f"{BASE_PATH}/events_behavior_TS_LFPindex.csv", delimiter=',', encoding="utf8")

    # Inserting the events behavior data in the dataframe as a column

    events_behavior_TS_LFP_index_array = np.full((len(data),), None)

    for start, end, event_idx in zip(events_behavior_TS_LFP_index.start, events_behavior_TS_LFP_index.end, np.arange(1, len(events_behavior_TS_LFP_index))):
        events_behavior_TS_LFP_index_array[start:end] = event_idx
        
    data = data.assign(events_behavior_TS_LFP_index=events_behavior_TS_LFP_index_array)

    events_behavior_TS_LFPsec = pd.read_csv(f"{BASE_PATH}/events_behavior_TS_LFPsec.csv", delimiter=',', encoding="utf8")

    return data, events_index, events_index_timestamp, events_behavior_TS_LFP_index, events_behavior_TS_LFPsec, BASE_PATH

def select_event_window(
    df: pd.DataFrame, 
    event_number: int, 
    samples_before: int = 0, 
    samples_after: int = 0
) -> pd.DataFrame:
  """
  Method to extract the slice of the dataframe which contais the event, with some data before and after, 
  given number of samples to add to the begin and end, respectively.
  """
  
  window_index = np.argwhere(data.events_index.to_numpy() == event_number).flatten()
  begin_index = window_index[0] - samples_before
  end_index = window_index[-1] + samples_after
  return df[begin_index:end_index]


def decimate(data, desired_frequency_sampling):
    backup_data = data.copy()
    time = backup_data.Time.to_numpy()
    TimeSampling = round(np.mean(time[1:] - time[:-1]), 6)
    FrequencySampling = 1.0/TimeSampling
    print(f"The time sampling is {TimeSampling} seconds and the frequency is "
        f"{FrequencySampling/float(1000**(FrequencySampling<=1000))} {'k'*bool(FrequencySampling>=1000)}Hz")

    newTimeSampling = 1.0/desired_frequency_sampling
    decimation_rate = np.ceil(newTimeSampling/TimeSampling).astype(int)
    print(f"The data will be decimated by the rate 1:{decimation_rate}")

    data = data[::decimation_rate]

    TimeSampling = newTimeSampling
    
    FrequencySampling = 1.0/TimeSampling
    print(f"The new time sampling is {np.round(TimeSampling, 5)} s and the new frequency is "
    f"{FrequencySampling/float(1000**(FrequencySampling>=1000))} {'k'*bool(FrequencySampling>=1000)}Hz")
    
    return data, TimeSampling, FrequencySampling, backup_data


def process_tdbs(
    column: list[str], 
    df: pd.DataFrame, 
    args_tdbs: dict
) -> dict:
    signal = df[column]
     
    return {
        "column": column, 
        "result": tdbs(
            signal=signal,
            frequency_sampling=args_tdbs["frequency_sampling"],
            time=args_tdbs["time"],
            fmin=args_tdbs["fmin"],
            fmax=args_tdbs["fmax"],
            freq_step=args_tdbs["freq_step"],
            phase_step=args_tdbs["phase_step"],
            enable_progress_bar=False
        )
    }


if __name__ == "__main__":

    generate_plots: bool = False

    data, events_index, events_index_timestamp, events_behavior_TS_LFP_index, events_behavior_TS_LFPsec, BASE_PATH = load_data()

    samples_before = 0
    samples_after = 0
    event_number = 1

    event_data = select_event_window(
        df=data,
        event_number=event_number,
        samples_before=samples_before,
        samples_after=samples_after
    )

    desired_frequency_sampling = 200

    data, TimeSampling, FrequencySampling, backup_data = decimate(event_data, desired_frequency_sampling=desired_frequency_sampling)

    time = event_data.Time.to_numpy()

    spectrum_df = pd.DataFrame()
    bispectrum_df = pd.DataFrame()
    
    with Pool() as pool:
        channels_columns = event_data.columns[2:18]

        for result in pool.map(
            lambda column: process_tdbs(
                column, 
                event_data, 
                {
                    "frequency_sampling": FrequencySampling,
                    "time": time,
                    "fmin": 52,
                    "fmax": 55,
                    "freq_step": 0.1,
                    "phase_step": 0.1
                }
            ), 
            channels_columns
        ):
            column = result["column"]
            signal = event_data[column].to_numpy()
            frequency_array, spectrum, phase_spectrum, bispectrum, phase_bispectrum = result["result"]

            if "frequency" not in spectrum_df.columns or "frequency" not in bispectrum_df.columns:
                spectrum_df = spectrum_df.assign(frequency=frequency_array)
                bispectrum_df = bispectrum_df.assign(frequency=frequency_array)

            spectrum_df = spectrum_df.assign(**{f"tds_{column}": list(zip(spectrum, phase_spectrum))})
            bispectrum_df = bispectrum_df.assign(**{f"tdbs_{column}": list(zip(bispectrum, phase_bispectrum))})

            if generate_plots:
                fig = make_subplots(rows=3, cols=1)

                fig.append_trace(go.Scatter(
                    x=time,
                    y=signal,
                ), row=1, col=1)

                fig.append_trace(go.Scatter(
                    x=frequency_array,
                    y=spectrum,
                ), row=2, col=1)

                fig.append_trace(go.Scatter(
                    x=frequency_array,
                    y=bispectrum,
                ), row=3, col=1)

                fig.show()

    spectrum_df.to_csv(f"{BASE_PATH}/spectrum_df.csv", index=False)
    bispectrum_df.to_csv(f"{BASE_PATH}/bispectrum_df.csv", index=False)

    