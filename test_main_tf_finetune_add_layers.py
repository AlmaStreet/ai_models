import tensorflow as tf
import numpy as np

# Load the fine-tuned model
fine_tuned_model = tf.keras.models.load_model("finetuned_model_add_layers.keras")

# Define test function
def test_model(input_data, expected_output):
    test_input = np.array([input_data]).astype(np.float32)  # Ensure correct dtype
    prediction = fine_tuned_model.predict(test_input)

    # Convert probability output to class label
    animal_classes = ["Dog", "Cat"]
    predicted_label = animal_classes[np.argmax(prediction)]

    print(f"Input: {input_data} -> Prediction: {predicted_label}")
    assert predicted_label == expected_output, f"Test failed! Expected {expected_output}, but got {predicted_label}"

# Test cases
test_model([0, 0], "Dog")  # Short fur, pointy tail → Expected: Dog
test_model([1, 1], "Cat")  # Long fur, fluffy tail → Expected: Cat
test_model([0, 1], "Cat")  # Short fur, fluffy tail → Expected: Cat
test_model([1, 0], "Dog")  # Long fur, pointy tail → Expected: Dog

print("✅ All tests passed successfully!")
