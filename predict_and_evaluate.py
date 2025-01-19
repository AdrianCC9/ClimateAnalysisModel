import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf  # Import TensorFlow


# 1. Load the cleaned data
print("Loading cleaned data...")
df = pd.read_csv("cleaned_climate_data.csv")
print("Data loaded. Shape:", df.shape)

# 2. Define features and target
features = ["lon", "lat", "elev", "year", "day_of_year", "tasmax", "tasmin"]
target = "tas"

print("Dropping rows with missing values...")
df = df.dropna(subset=features + [target]).copy()
X = df[features]
y = df[target]

# 3. Split into training and testing sets (same split as before)
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale the test data (use the scaler trained on training data)
print("Scaling test data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit scaler on training data
X_test_scaled = scaler.transform(X_test)        # Transform test data only

# 5. Load the saved model
# Load the saved model in the Keras format
print("Loading the saved model...")
model = tf.keras.models.load_model("climate_model.keras")
print("Model loaded successfully.")

# 6. Make predictions on the test data
print("Making predictions on the test set...")
predictions = model.predict(X_test_scaled)
predictions = predictions.flatten()  # Flatten the predictions to 1D array for easier comparison

# 7. Compare predictions with actual values
print("\nSample Predictions vs Actual Values:")
for true, pred in zip(y_test[:10], predictions[:10]):  # Show first 10 comparisons
    print(f"Actual: {true:.2f}, Predicted: {pred:.2f}")

# 8. Calculate overall metrics
mae = np.mean(np.abs(y_test.values - predictions))
mse = np.mean((y_test.values - predictions) ** 2)
print(f"\nTest Set Performance:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
