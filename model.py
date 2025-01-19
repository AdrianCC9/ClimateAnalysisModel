print("Starting imports...")
import pandas as pd
print("pandas imported.")
import matplotlib.pyplot as plt
print("matplotlib imported.")
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("CPU devices:", tf.config.list_physical_devices('CPU'))
from tensorflow.keras import Sequential
print("keras Sequential imported.")
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    print("Starting script...")

    # 1. Import your cleaned CSV
    print("Loading cleaned CSV...")
    df = pd.read_csv("cleaned_climate_data.csv")
    print("Cleaned data loaded. Shape:", df.shape)
    print(df.head())

    # 2. Define features and target
    features = ["lon", "lat", "elev", "year", "day_of_year", "tasmax", "tasmin"]
    target = "tas"

    print("Dropping rows with missing values...")
    df = df.dropna(subset=features + [target]).copy()
    print(f"Data shape after dropping missing values: {df.shape}")

    X = df[features]
    y = df[target]

    # 3. Split into training and testing
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # 4. Scale numeric columns
    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on TRAIN, transform TRAIN
    X_test_scaled = scaler.transform(X_test)        # Transform TEST only
    print("Scaling complete.")

    # 5. Build a simple Keras model
    print("Building Keras model...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),  # 20% dropout
        Dense(64, activation='relu'),
        Dense(1)  # single output neuron for 'tas'
    ])
    print("Model built.")

    # 6. Compile the model
    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='mse',    # Mean Squared Error
        metrics=['mae']  # Mean Absolute Error
    )
    print("Model compiled.")

    # 7. Train the model
    print("Starting training...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1
    )
    print("Training complete.")

    # 8. Evaluate on the test set
    print("Evaluating model on test set...")
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test MSE (loss): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # 9. Plot training vs validation loss
    print("Plotting training history...")
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Training History: MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.show()
    print("Plotting complete.")

    # 10. Save the trained model
    # Save the trained model in the native Keras format
    print("Saving the trained model...")
    model.save("climate_model.keras", save_format="keras")
    print("Model saved as 'climate_model.keras'.")



if __name__ == "__main__":
    main()
