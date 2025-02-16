import tensorflow as tf
import numpy as np

# Sample data: [fur length, tail type]
# 0 = short, 1 = long (for fur length)
# 0 = pointy, 1 = fluffy (for tail type)
X = np.array([[0, 0],  # short fur, pointy tail (dog)
              [1, 1],  # long fur, fluffy tail (cat)
              [0, 1],  # short fur, fluffy tail (cat)
              [1, 0]]) # long fur, pointy tail (dog)

# Labels: 0 = dog, 1 = cat
y = np.array([0, 1, 1, 0])  # 0 is dog, 1 is cat

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),  # Input layer with 2 features
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer: 0 or 1 (dog or cat)
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=20)

model.save("model.keras")  # Saves in Keras format
# OR
# model.save("model.h5")  # Saves in HDF5 format

model.save("saved_model.keras")


# Test the model
test_input = np.array([[0, 0]])  # short fur, pointy tail
prediction = model.predict(test_input)
print(f"Prediction: {'Dog' if prediction[0] < 0.5 else 'Cat'}")