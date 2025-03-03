# evaluate_model.py
import numpy as np
import tensorflow as tf
from data_preprocessing import load_clean_data, split_and_scale_data

# File and feature settings
filepath = "cleaned_climate_data.csv"
features = ["lon", "lat", "elev", "year", "day_of_year", "tasmax", "tasmin"]
target = "tas"

# Load and preprocess data
df = load_clean_data(filepath)
X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(df, features, target)

# Load the saved model
model = tf.keras.models.load_model("climate_model.keras")
print("Model loaded successfully.")

# Make predictions on the test set
predictions = model.predict(X_test_scaled).flatten()

# Calculate error metrics
mae = np.mean(np.abs(y_test.values - predictions))
mse = np.mean((y_test.values - predictions) ** 2)

# Display sample predictions vs. actual values
print("\nSample Predictions vs Actual Values:")
for true, pred in zip(y_test.values[:10], predictions[:10]):
    print(f"Actual: {true:.2f}, Predicted: {pred:.2f}")

print(f"\nMean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
