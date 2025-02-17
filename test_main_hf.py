import torch
from transformers import PreTrainedModel, PretrainedConfig
from safetensors.torch import load_file  # Import safetensors

# Define a Hugging Face-compatible model configuration
class SimpleConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_size = 2
        self.output_size = 1

# Define a simplified model (logistic regression) with a single linear layer
class SimpleModel(PreTrainedModel):
    config_class = SimpleConfig

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.input_size, config.output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, features):
        # Compute logits directly from a single layer
        logits = self.linear(features.float())
        output = self.sigmoid(logits)
        return {"logits": output}

# Load the model
config = SimpleConfig()
model = SimpleModel(config)

# Load the model weights from `model.safetensors`
weights = load_file("saved_hf_model/model.safetensors")
model.load_state_dict(weights)  # Now the keys match: "linear.weight", "linear.bias"
model.eval()  # Set the model to evaluation mode

# Define test function
def test_model(input_data, expected_output):
    test_input = torch.tensor(input_data, dtype=torch.float32)  # Convert input to PyTorch tensor
    with torch.no_grad():
        logits = model(test_input)["logits"]
    probability = logits.item() if logits.numel() == 1 else logits.squeeze().item()
    
    # Convert probability to class (threshold = 0.5)
    predicted_label = "Dog" if probability < 0.5 else "Cat"
    
    print(f"Input: {input_data}, Probability: {probability:.4f}, Prediction: {predicted_label}, Expected: {expected_output}")
    assert predicted_label == expected_output, f"Test failed! Expected {expected_output}, but got {predicted_label}"

# Test cases
test_model([[0, 0]], "Dog")  # short fur, pointy tail → Dog
test_model([[1, 1]], "Cat")  # long fur, fluffy tail → Cat
test_model([[0, 1]], "Cat")  # short fur, fluffy tail → Cat
test_model([[1, 0]], "Dog")  # long fur, pointy tail → Dog

print("All tests passed successfully! 🎉")
