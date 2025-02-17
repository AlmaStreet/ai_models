import tensorflow as tf
import numpy as np

# Load pre-trained model
base_model = tf.keras.models.load_model("saved_model.keras")

# Freeze base layers so they don't get modified
for layer in base_model.layers:
    layer.trainable = False

# Add new layers on top
fine_tuned_model = tf.keras.Sequential([
    base_model,  # Keep existing trained model
    tf.keras.layers.Dense(4, activation='relu', name="new_hidden_layer"),  # Extra learning capacity
    tf.keras.layers.Dense(2, activation='softmax', name="new_output_layer")  # Change output to 2 categories
])

# Compile the model
fine_tuned_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

# Fine-tune training data (same input, but labels need to match new output shape)
X_finetune = np.array([[0, 0], [1, 1], [0, 1], [1, 0]]).astype(np.float32)  # Same input data
y_finetune = np.array([0, 1, 1, 0])  # Same labels, but converted to categorical

# Custom Early Stopping Callback
class StopAtAccuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") == 1.00:  # Stop when accuracy reaches 1.00
            print(f"\n✅ Stopping early! Accuracy reached 1.00 at epoch {epoch + 1}.")
            self.model.stop_training = True
            self.model.save("finetuned_model_add_layers.keras")  # Save the model

# Train the fine-tuned model with early stopping
fine_tuned_model.fit(X_finetune, y_finetune, epochs=500, callbacks=[StopAtAccuracy()])

# Test the fine-tuned model
test_input = np.array([[0, 0]]).astype(np.float32)  # short fur, pointy tail
prediction = fine_tuned_model.predict(test_input)

# Convert probability output to class label
predicted_label = "Dog" if np.argmax(prediction) == 0 else "Cat"
print(f"Prediction: {predicted_label}")
