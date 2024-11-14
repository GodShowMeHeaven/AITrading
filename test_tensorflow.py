"""
Test TensorFlow installation
"""
import tensorflow as tf
import numpy as np

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Create a simple model
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(8, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

print("Model created successfully!")

# Create some dummy data
X = np.random.random((100, 10))
y = np.random.random((100, 1))

# Try to compile and fit
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=1, batch_size=32, verbose=1)

print("Test completed successfully!")
