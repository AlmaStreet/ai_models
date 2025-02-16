import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Sample data: [fur length, tail type]
# 0 = short, 1 = long (for fur length)
# 0 = pointy, 1 = fluffy (for tail type)
X = np.array([[0, 0],  # short fur, pointy tail (dog)
              [1, 1],  # long fur, fluffy tail (cat)
              [0, 1],  # short fur, fluffy tail (cat)
              [1, 0]]) # long fur, pointy tail (dog)

y = np.array([0, 1, 1, 0])  # Labels: 0 = dog, 1 = cat

# Convert NumPy arrays to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Reshape to (4,1)

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # Input layer (2 features) → Hidden layer (4 neurons)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)  # Hidden layer → Output layer (1 neuron)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Create the model
model = SimpleNN()

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()  # Zero the gradients
    output = model(X_tensor)  # Forward pass
    loss = criterion(output, y_tensor)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "model.pth")  # Saves model weights
print("Model saved as model.pth")

# Load the model
loaded_model = SimpleNN()
loaded_model.load_state_dict(torch.load("model.pth"))
loaded_model.eval()  # Set to evaluation mode

# Test the model
test_input = torch.tensor([[0, 0]], dtype=torch.float32)  # short fur, pointy tail
prediction = loaded_model(test_input).item()  # Convert tensor to scalar value

# Convert probability to class (threshold = 0.5)
predicted_label = "Dog" if prediction < 0.5 else "Cat"
print(f"Prediction: {predicted_label}")
