import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os
import pandas as pd

def read_all_csv(folder_path):
    """
    Read all CSV files in the specified folder and concatenate them into a single DataFrame.

    :param folder_path: Path to the folder containing CSV files.
    :return: A single concatenated DataFrame of all CSV files.
    """
    all_dataframes = []  # List to store each DataFrame

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            all_dataframes.append(df)

    # Concatenate all DataFrames in the list
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df

# Usage
# Replace '../data/' with the path to your folder containing CSV files
combined_df = read_all_csv('Q:\\Users\\adria\\PycharmProjects\\Personal\\Projects\\ClimateAnalysisModel\\data\\station_data')
