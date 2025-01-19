import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset for default values
df = pd.read_csv("cleaned_climate_data.csv")
scaler = StandardScaler()

# Define features and target
features = ["lon", "lat", "elev", "year", "day_of_year", "tasmax", "tasmin"]
df = df.dropna(subset=features).copy()  # Ensure no missing values in features

# Fit the scaler on the dataset
scaler.fit(df[features])

# Load the saved model
model = tf.keras.models.load_model("climate_model.keras")
print("Model loaded successfully.")

# Function to get user input with default fallback
def get_user_input(prompt, default):
    user_input = input(prompt)
    return float(user_input) if user_input.strip() else default

# Main prediction function
def predict_temperature():
    print("Welcome to the Temperature Prediction Tool!")
    print("You can leave any input blank to use the default value from the dataset.")

    # Get user inputs, providing dataset means as defaults
    lon = get_user_input(f"Enter longitude (default: {df['lon'].mean():.2f}): ", df["lon"].mean())
    lat = get_user_input(f"Enter latitude (default: {df['lat'].mean():.2f}): ", df["lat"].mean())
    elev = get_user_input(f"Enter elevation (default: {df['elev'].mean():.2f}): ", df["elev"].mean())
    year = get_user_input(f"Enter year (default: {df['year'].min()}): ", df["year"].min())  # Allow from earliest year
    day_of_year = get_user_input(f"Enter day of year (default: {df['day_of_year'].mean():.0f}): ", df["day_of_year"].mean())
    tasmax = get_user_input(f"Enter max temperature (tasmax) (default: {df['tasmax'].mean():.2f}): ", df["tasmax"].mean())
    tasmin = get_user_input(f"Enter min temperature (tasmin) (default: {df['tasmin'].mean():.2f}): ", df["tasmin"].mean())

    # Combine inputs into a single array
    input_data = np.array([[lon, lat, elev, year, day_of_year, tasmax, tasmin]])

    # Scale input data
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)
    print(f"\nPredicted Average Temperature (tas): {prediction[0][0]:.2f}°C")

if __name__ == "__main__":
    predict_temperature()
