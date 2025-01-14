import os
import torch.optim as optim
import Tacotron2
from torch import nn
from data_prep import prepare_dataset
from data_loader import create_dataloader


# Training loop
def train(model, dataloader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            text, mel = batch  # Assume text and mel are preprocessed
            text, mel = text.to(device), mel.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(text)

            # Compute loss
            loss = criterion(output, mel)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Example usage
model = Tacotron2(input_dim=50, output_dim=80)  # Example dimensions
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming dataloader is defined
# Prepare dataset and DataLoader
data_dir = "./dataset"
metadata_file = "metadata.csv"
metadata = prepare_dataset(data_dir, metadata_file)

# Create DataLoader
batch_size = 16
dataloader = create_dataloader(metadata, data_dir, batch_size=batch_size)

# Initialize model, criterion, and optimizer
model = Tacotron2(input_dim=128, output_dim=80)  # Adjust input_dim for tokenized text
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, dataloader, criterion, optimizer, num_epochs=10)


