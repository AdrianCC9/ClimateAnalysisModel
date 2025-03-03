import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath: str, valid_flags: list = None, invalid_flags: list = None) -> pd.DataFrame:

    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded data from {filepath} with shape: {df.shape}")

    # Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # Create new columns: 'year' and 'day_of_year'
    df['year'] = df['time'].dt.year
    df['day_of_year'] = df['time'].dt.dayofyear

    # Filter based on valid/invalid flags
    if valid_flags:
        df = df[df['tas_flag'].isin(valid_flags) & df['tasmax_flag'].isin(valid_flags) & df['tasmin_flag'].isin(valid_flags)]
    if invalid_flags:
        df = df[~df['tas_flag'].isin(invalid_flags) & ~df['tasmax_flag'].isin(invalid_flags) & ~df['tasmin_flag'].isin(invalid_flags)]

    # Drop unnecessary columns
    drop_columns = ['station', 'station_name', 'tas_flag', 'tasmax_flag', 'tasmin_flag']
    df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True, errors='ignore')

    # Interpolate missing numeric values
    df.interpolate(method='linear', inplace=True)

    print(f"Data shape after cleaning and interpolation: {df.shape}")
    return df

def split_and_scale_data(df: pd.DataFrame, features: list, target: str, test_size: float = 0.2, random_state: int = 42):
    
    df = df.dropna(subset=features + [target]).copy()
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
