import torch
import numpy as np
from torch import nn
from transformers import Trainer, TrainingArguments, PreTrainedModel, PretrainedConfig
from datasets import Dataset

# For reproducibility
torch.manual_seed(42)

# Sample data: [fur length, tail type]
# Dog: [0,0] and [1,0] | Cat: [1,1] and [0,1]
X = np.array([[0, 0],  # short fur, pointy tail → Dog
              [1, 1],  # long fur, fluffy tail → Cat
              [0, 1],  # short fur, fluffy tail → Cat
              [1, 0]], dtype=np.float32)  # long fur, pointy tail → Dog
y = np.array([0, 1, 1, 0], dtype=np.float32)  # 0 = Dog, 1 = Cat

# Convert to a Hugging Face Dataset
dataset = Dataset.from_dict({"features": X.tolist(), "labels": y.tolist()})

# Define a minimal configuration
class SimpleConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_size = 2
        self.output_size = 1

# Define a very simple model (logistic regression)
class SimpleModel(PreTrainedModel):
    config_class = SimpleConfig

    def __init__(self, config):
        super().__init__(config)
        # A single linear layer is sufficient for this task
        self.linear = nn.Linear(config.input_size, config.output_size)

    def forward(self, features, labels=None):
        # Compute raw logits
        logits = self.linear(features.float())
        loss = None
        if labels is not None:
            # Make sure labels have shape (batch_size, 1)
            labels = labels.unsqueeze(1).float()
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

# Custom data collator (full-batch)
def collate_fn(batch):
    features = torch.tensor([b["features"] for b in batch], dtype=torch.float32)
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.float32)
    return {"features": features, "labels": labels}

# Use a higher learning rate since the model is very simple and dataset is tiny.
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=300,                # Train long enough to see convergence
    per_device_train_batch_size=4,       # Use all 4 samples in one batch
    learning_rate=1e-2,                  # A bit higher for this tiny problem
    weight_decay=0.0,
    warmup_steps=0,
    lr_scheduler_type="constant",        # Keep learning rate fixed
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no"
)

# Initialize the model and trainer
config = SimpleConfig()
model = SimpleModel(config)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

# Train the model
trainer.train()

# Save the model in Hugging Face format
model.save_pretrained("saved_hf_model")
print("Model saved in Hugging Face format.")

# Load and test the model
loaded_model = SimpleModel.from_pretrained("saved_hf_model")

# Create test inputs and compute predictions
test_inputs = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=torch.float32)
with torch.no_grad():
    outputs = loaded_model(test_inputs)
# Apply sigmoid to convert logits to probabilities
predictions = torch.sigmoid(outputs["logits"]).numpy()

# Print predictions: probability > 0.5 → Cat, else Dog.
for i, pred in enumerate(predictions):
    predicted_label = "Dog" if pred < 0.5 else "Cat"
    print(f"Test {i+1}: Prediction = {predicted_label}, Probability = {pred[0]:.4f}")
