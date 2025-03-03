# train_model.py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from data_utils import load_clean_data, split_and_scale_data

# File and feature settings
filepath = "cleaned_climate_data.csv"
features = ["lon", "lat", "elev", "year", "day_of_year", "tasmax", "tasmin"]
target = "tas"

# Load the dataset and process it
df = load_clean_data(filepath)
X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(df, features, target)

# Build a simple feed-forward neural network (MLP)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)  # Single output for regression
])

# Compile the model with Adam optimizer and MSE loss
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Save the trained model
model.save("climate_model.keras")
print("Model saved as 'climate_model.keras'.")
