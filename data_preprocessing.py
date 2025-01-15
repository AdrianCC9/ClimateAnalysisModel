# data_preprocessing.py

import pandas as pd

def preprocess_data(
    filepath: str,
    valid_flags: list = None,  # Flags we consider "valid"
    invalid_flags: list = None # Flags we want to exclude
) -> pd.DataFrame:
    """
    Loads and preprocesses the climate data from a CSV file.
    Adjust the flags to match your dataset's reality.
    """

    # 1. Load dataset
    #    If you have mixed dtypes, set low_memory=False or specify dtype per column
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded data from {filepath} with shape: {df.shape}")

    # 2. Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # 3. Create new columns: 'year' and 'day_of_year'
    df['year'] = df['time'].dt.year
    df['day_of_year'] = df['time'].dt.dayofyear

    # ---- Inspect flags before filtering (Optional) ----
    print("tas_flag unique:", df['tas_flag'].unique())
    print("tasmax_flag unique:", df['tasmax_flag'].unique())
    print("tasmin_flag unique:", df['tasmin_flag'].unique())

    # 4. Drop rows based on flags
    # Option A: Keep only certain valid flags
    if valid_flags is not None:
        df = df[
            df['tas_flag'].isin(valid_flags)
            & df['tasmax_flag'].isin(valid_flags)
            & df['tasmin_flag'].isin(valid_flags)
        ]

    # Option B: Exclude certain invalid flags
    if invalid_flags is not None:
        df = df[
            ~df['tas_flag'].isin(invalid_flags)
            & ~df['tasmax_flag'].isin(invalid_flags)
            & ~df['tasmin_flag'].isin(invalid_flags)
        ]

    # (You can do Option A or B, or both, depending on your data logic.)

    # 5. Drop columns we don't need
    #    Example: station info, station_name, plus the flag columns themselves
    drop_columns = [
        'station',
        'station_name',
        'tas_flag',
        'tasmax_flag',
        'tasmin_flag'
    ]
    # If you want to drop tasmax/tasmin as well:
    # drop_columns += ['tasmax', 'tasmin']

    df.drop(columns=[col for col in drop_columns if col in df.columns],
            inplace=True,
            errors='ignore')

    # 6. Interpolate numeric columns
    df.interpolate(method='linear', inplace=True)

    print(f"Data shape after cleaning and interpolation: {df.shape}")
    print(df.head(5))

    return df

    



if __name__ == "__main__":
    csv_path = r"Q:\adria\Documents\climate_data.csv"
    # Example: Let's keep rows that have flags either 'E', 'A', or NaN
    # (assuming these are acceptable in your data).
    # We'll skip invalid_flags for this example.

    cleaned_df = preprocess_data(
        filepath=csv_path,
        valid_flags=['E', 'A', None, '', ' '],  # adjust as needed
        invalid_flags=None
    )

