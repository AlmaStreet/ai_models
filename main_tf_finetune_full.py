import tensorflow as tf
import numpy as np

# Load the pre-trained model
base_model = tf.keras.models.load_model("saved_model.keras")

# Freeze all layers except the last one (output layer)
for layer in base_model.layers:
    layer.trainable = True

# Replace the output layer with a new one (for multi-class classification)
new_output_layer = tf.keras.layers.Dense(4, activation='softmax', name="new_output_layer")

# Create a new model by replacing the last layer
new_model = tf.keras.Sequential([
    base_model.layers[0],  # Keep the original input and hidden layers
    new_output_layer       # Replace the old output layer
])

# Compile the new model
new_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Multi-class classification
                  metrics=['accuracy'])

# Define new dataset (keeping only 2 input features)
X = np.array([
    [0, 0],  # Dog: short fur, pointy tail
    [1, 1],  # Cat: long fur, fluffy tail
    [0, 1],  # Fox: short fur, fluffy tail
    [1, 0]   # Rabbit: long fur, pointy tail
]).astype(np.float32)

# One-hot encoded labels for 4 categories
y = np.array([
    [1, 0, 0, 0],  # Dog
    [0, 1, 0, 0],  # Cat
    [0, 0, 1, 0],  # Fox
    [0, 0, 0, 1]   # Rabbit
]).astype(np.float32)


# Custom Early Stopping Callback
class StopAtAccuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") == 1.00:  # Stop when accuracy reaches 1.00
            print(f"\n✅ Stopping early! Accuracy reached 1.00 at epoch {epoch + 1}.")
            self.model.stop_training = True
            self.model.save("finetuned_model_full.keras")  # Save the model

# Train the fine-tuned model with early stopping
new_model.fit(X, y, epochs=500, callbacks=[StopAtAccuracy()])



# # Fine-tune the model with 100 epochs
# new_model.fit(X, y, epochs=100)

# # Save the fine-tuned model
# new_model.save("finetuned_model_head_reinit.keras")

# Test the fine-tuned model
test_input = np.array([[1, 0]]).astype(np.float32)  # Example input
prediction = new_model.predict(test_input)

# Convert probability output to class label
animal_classes = ["Dog", "Cat", "Fox", "Rabbit"]
predicted_label = animal_classes[np.argmax(prediction)]
print(f"Prediction: {predicted_label}")
