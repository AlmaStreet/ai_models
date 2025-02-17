import tensorflow as tf
import numpy as np

# Updated sample data with 4 features (fur length, tail type, ear shape, speed)
X = np.array([
    [0, 0, 0, 0],  # Dog: short fur, pointy tail, small ears, slow
    [1, 1, 1, 1],  # Cat: long fur, fluffy tail, large ears, fast
    [0, 1, 1, 0],  # Fox: short fur, fluffy tail, large ears, slow
    [1, 0, 0, 1]   # Rabbit: long fur, pointy tail, small ears, fast
]).astype(np.float32)  # Convert to float32

# One-hot encoded labels for 4 categories: Dog, Cat, Fox, Rabbit
y = np.array([
    [1, 0, 0, 0],  # Dog
    [0, 1, 0, 0],  # Cat
    [0, 0, 1, 0],  # Fox
    [0, 0, 0, 1]   # Rabbit
]).astype(np.float32)  # Convert to float32

# Rebuild the model (DO NOT reuse weights due to shape mismatch)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,), name="hidden_layer"),  # Increased hidden units
    tf.keras.layers.Dense(4, activation='softmax', name="output_layer")  # Multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the categorical model for 50 epochs
model.fit(X, y, epochs=300)

# Save the categorical model
model.save("saved_model_categorical.keras")

# Test the model with a new input
test_input = np.array([[1, 1, 0, 1]])  # Example: long fur, fluffy tail, small ears, fast
prediction = model.predict(test_input)

# Convert the output probabilities to the predicted category
animal_classes = ["Dog", "Cat", "Fox", "Rabbit"]
predicted_label = animal_classes[np.argmax(prediction)]
print(f"Prediction: {predicted_label}")
