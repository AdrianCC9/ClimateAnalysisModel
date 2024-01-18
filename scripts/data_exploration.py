#Explore organization and structure of dataset
from Projects.ClimateAnalysisModel.scripts.dataloading_achhd import combined_df

print(combined_df.head())
print(combined_df.info())
print(combined_df.describe())
print(combined_df.isnull().sum())

print(combined_df.hist())
print(combined_df.show())

