import numpy as np
import pandas as pd

def prepare_data(wheater_data: str, revenue_data: str) -> pd.DataFrame:
    """
    Reads, cleans, and merges weather data with revenue and holiday data.

    Parameters:
    wheater_data (str): Path to the CSV file containing weather data. 
    revenue_data (str): Path to the CSV file containing revenue and holiday data. 

    Returns:
    df: A merged DataFrame containing both weather and revenue data. The weather data has missing columns removed, while the revenue data replaces `1` with `True` for holidays and fills missing values with `False`.
    """
    #read and prepare wheater data
    wheater_df = pd.read_csv(wheater_data, parse_dates=True, index_col=0)
    #delete the rows where we have no information. all other columns have no NaN value
    wheater_df = wheater_df.dropna(axis=1) 

    # read and prepare revenua and holiday data
    revenue_df = pd.read_csv(revenue_data, sep=';', parse_dates=True, index_col=0)
    revenue_df = revenue_df.fillna(False)
    revenue_df = revenue_df.replace(1,True)

    df = pd.merge(wheater_df, revenue_df, left_index=True, right_index=True)
    return df
