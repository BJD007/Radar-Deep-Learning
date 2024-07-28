import tensorflow as tf

def create_doa_estimation_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(num_antennas, num_samples)),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')  # or 2 for azimuth and elevation
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the model
doa_model = create_doa_estimation_model()
doa_model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Use the model for prediction
predicted_doa = doa_model.predict(X_test)
