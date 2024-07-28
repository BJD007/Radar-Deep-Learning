import tensorflow as tf

def create_doppler_estimation_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(time_steps, num_channels)),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the model
doppler_model = create_doppler_estimation_model()
doppler_model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Use the model for prediction
predicted_doppler = doppler_model.predict(X_test)
