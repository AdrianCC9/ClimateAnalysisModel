#CHIP5A data loading

import pandas as pd
file_path2 = "Q:\\Users\\adria\\PycharmProjects\\Personal\\Projects\\ClimateAnalysisModel\\data\\CMIP5A_data\\CHIP5A_data.csv"

dfA = pd.read_csv(file_path2)

print(dfA.tail())
print(dfA.info())
