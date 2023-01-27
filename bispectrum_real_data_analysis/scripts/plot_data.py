import pandas as pd
import plotly.express as px
import os
from plotly.subplots import make_subplots


if __name__ == "__main__":

    BASE_PATH = os.getcwd() + "/bispectrum_real_data_analysis/data"

    df = pd.read_csv(f"{BASE_PATH}/bispectrum_df.csv", delimiter=',', encoding="utf8")

    amplitudes = df.iloc[:, 1:17]
    phases = df.iloc[:, 17:33]
    fig = make_subplots(rows=3, cols=1)
    amplitudes = px.line(df, x=df.iloc[:, 0], y=amplitudes.columns)
    phases = px.line(df, x=df.iloc[:, 0], y=phases.columns)
    
    amplitudes.show()
    phases.show()
