import tensorflow as tf
import numpy as np

# Load the fine-tuned model
model = tf.keras.models.load_model("saved_model_categorical.keras")

# Define test function
def test_model(input_data, expected_output):
    test_input = np.array([input_data])  # Convert to NumPy array with batch dimension
    prediction = model.predict(test_input)

    # Convert the output probabilities to the predicted category
    animal_classes = ["Dog", "Cat", "Fox", "Rabbit"]
    predicted_label = animal_classes[np.argmax(prediction)]

    print(f"Prediction: {predicted_label}")
    assert predicted_label == expected_output, f"Test failed! Expected {expected_output}, but got {predicted_label}"

# Test cases
test_model([0, 0, 0, 0], "Dog")    # short fur, pointy tail, small ears, slow → Dog
test_model([1, 1, 1, 1], "Cat")    # long fur, fluffy tail, large ears, fast → Cat
test_model([0, 1, 1, 0], "Fox")    # short fur, fluffy tail, large ears, slow → Fox
test_model([1, 0, 0, 1], "Rabbit") # long fur, pointy tail, small ears, fast → Rabbit

print("✅ All tests passed successfully! 🎉")
