import tensorflow as tf

def create_detection_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(time_steps, num_channels)),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
detection_model = create_detection_model()
detection_model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Use the model for prediction
detection_probabilities = detection_model.predict(X_test)
