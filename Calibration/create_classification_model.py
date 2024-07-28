import tensorflow as tf

def create_classification_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(time_steps, num_channels)),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
classification_model = create_classification_model(num_classes)
classification_model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Use the model for prediction
predicted_classes = classification_model.predict(X_test)
