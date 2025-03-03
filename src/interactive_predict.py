# interactive_predict.py
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_preprocessing import load_clean_data

# File and feature settings
filepath = "cleaned_climate_data.csv"
features = ["lon", "lat", "elev", "year", "day_of_year", "tasmax", "tasmin"]

# Load the full dataset (for calculating default values)
df = load_clean_data(filepath)

# Load the saved model
model = tf.keras.models.load_model("climate_model.keras")
print("Model loaded successfully.")

# Recompute a scaler on the full dataset to mimic training scaling (for prediction defaults)
scaler = StandardScaler()
scaler.fit(df[features])

def get_user_input(prompt, default):
    user_input = input(prompt)
    return float(user_input) if user_input.strip() else default

def predict_temperature():
    print("Welcome to the Temperature Prediction Tool!")
    print("Leave any input blank to use the default value from the dataset.")

    lon = get_user_input(f"Enter longitude (default: {df['lon'].mean():.2f}): ", df["lon"].mean())
    lat = get_user_input(f"Enter latitude (default: {df['lat'].mean():.2f}): ", df["lat"].mean())
    elev = get_user_input(f"Enter elevation (default: {df['elev'].mean():.2f}): ", df["elev"].mean())
    year = get_user_input(f"Enter year (default: {df['year'].min()}): ", df["year"].min())
    day_of_year = get_user_input(f"Enter day of year (default: {df['day_of_year'].mean():.0f}): ", df["day_of_year"].mean())
    tasmax = get_user_input(f"Enter max temperature (tasmax) (default: {df['tasmax'].mean():.2f}): ", df["tasmax"].mean())
    tasmin = get_user_input(f"Enter min temperature (tasmin) (default: {df['tasmin'].mean():.2f}): ", df["tasmin"].mean())

    # Combine inputs into an array, scale, and predict
    input_data = np.array([[lon, lat, elev, year, day_of_year, tasmax, tasmin]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    
    print(f"\nPredicted Average Temperature (tas): {prediction[0][0]:.2f}°C")

if __name__ == "__main__":
    predict_temperature()
