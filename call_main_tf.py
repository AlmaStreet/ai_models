from tensorflow.keras.models import load_model
import numpy as np

model = load_model("saved_model.keras")  # OR load_model("model.h5")
# model = tf.keras.models.load_model("saved_model/")

test_input = np.array([[0, 0]])  
prediction = model.predict(test_input)
print(f'prediction: {prediction}')
expected_output = "Dog"
print(f"Prediction: {'Dog' if prediction[0] < 0.5 else 'Cat'}")
assert 'Dog' if prediction[0] < 0.5 else 'Cat' == expected_output, f"Test failed! Expected {expected_output}, but got {prediction}"

test_input = np.array([[1, 1]])  
prediction = model.predict(test_input)
expected_output = "Cat"
print(f"Prediction: {'Dog' if prediction[0] < 0.5 else 'Cat'}")
assert 'Dog' if prediction[0] < 0.5 else 'Cat' == expected_output, f"Test failed! Expected {expected_output}, but got {prediction}"

test_input = np.array([[0, 1]])  
prediction = model.predict(test_input)
expected_output = "Cat"
print(f"Prediction: {'Dog' if prediction[0] < 0.5 else 'Cat'}")
assert 'Dog' if prediction[0] < 0.5 else 'Cat' == expected_output, f"Test failed! Expected {expected_output}, but got {prediction}"

test_input = np.array([[1, 0]])  
prediction = model.predict(test_input)
expected_output = "Dog"
print(f"Prediction: {'Dog' if prediction[0] < 0.5 else 'Cat'}")
assert 'Dog' if prediction[0] < 0.5 else 'Cat' == expected_output, f"Test failed! Expected {expected_output}, but got {prediction}"