#CHIP5A data loading

import pandas as pd
import json

file_path2 = "Q:\\Users\\adria\\PycharmProjects\\Personal\\Projects\\ClimateAnalysisModel\\data\\CMIP5A_data\\CHIP5A_data.csv"

combined_df = pd.read_csv(file_path2, header=None)

json_string = combined_df.iloc[0, 0]

column_names_dict = json.loads(json_string)

column_names = list(column_names_dict.keys())

combined_df.columns = column_names

combined_df = combined_df.iloc[1:]

print(combined_df.head())


