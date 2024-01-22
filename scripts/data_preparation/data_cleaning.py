import pandas as pd
import os

#Loading Data
normals_data_dir = '/Projects/ClimateAnalysisModel/data/raw_data/normals_data'
csv_files = [f for f in os.listdir(normals_data_dir) if f.endswith('.csv')]

dfs = []

for csv_file in csv_files:
    file_path = os.path.join(normals_data_dir, csv_file)
    try:
        df = pd.read_csv(file_path)
        dfs.append(df)
    except pd.errors.ParserError:
        print(f'Error reading {file_path}')
    except Exception as e:
        print('Unexpected error has occured')

all_data = pd.concat(dfs, ignore_index=True)

#Initial Inspection
print("Initial Data Overview:")
print(all_data.shape)
print(all_data.head())
print("\nData Types:")
print(all_data.dtypes)
print("\nDescriptive Statistics:")
print(all_data.describe())

#Recheck for missing values
print("\nMissing Values After Cleaning:")
missing_values = all_data.isnull().sum()
print(missing_values[missing_values > 0])

#Confirm Data Readiness
if missing_values.sum() == 0:
    print("\nData is clean and ready for download")
else:
    print("\nData still contains missing values")

#Save Cleaned Data
clean_data = 'Q:\\Users\\adria\\PycharmProjects\\Personal\\Projects\\ClimateAnalysisModel\\data\\clean_data\\clean_data.csv'
try:
    all_data.to_csv(clean_data, index=False)
    print(f"\nCleaned data saved to {clean_data}")
except PermissionError:
    print(f"Permission denied for {clean_data}")
except Exception as e:
    print(f"Error: {e}")
