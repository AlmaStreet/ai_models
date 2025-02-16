import torch
import numpy as np

# Define the model architecture (must match the trained model)
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(2, 4)  # Input layer (2 features) → Hidden layer (4 neurons)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4, 1)  # Hidden layer → Output layer (1 neuron)
        self.sigmoid = torch.nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Load the model
model = SimpleNN()
model.load_state_dict(torch.load("model.pth"))  # Load trained weights
model.eval()  # Set to evaluation mode

# Define test function
def test_model(input_data, expected_output):
    test_input = torch.tensor(input_data, dtype=torch.float32)  # Convert input to PyTorch tensor
    prediction = model(test_input).item()  # Get scalar output (probability)
    
    # Convert probability to class (threshold = 0.5)
    predicted_label = "Dog" if prediction < 0.5 else "Cat"
    
    print(f"Prediction: {predicted_label}")
    assert predicted_label == expected_output, f"Test failed! Expected {expected_output}, but got {predicted_label}"

# Test cases
test_model([[0, 0]], "Dog")  # short fur, pointy tail → Dog
test_model([[1, 1]], "Cat")  # long fur, fluffy tail → Cat
test_model([[0, 1]], "Cat")  # short fur, fluffy tail → Cat
test_model([[1, 0]], "Dog")  # long fur, pointy tail → Dog

print("All tests passed successfully! 🎉")
