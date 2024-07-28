import tensorflow as tf

def create_range_estimation_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(time_steps, num_channels)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the model
range_model = create_range_estimation_model()
range_model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Use the model for prediction
predicted_range = range_model.predict(X_test)
