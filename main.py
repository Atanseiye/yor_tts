from data_prep import prepare_dataset
import Tacotron2
from torch import nn
from torch.optim import optim
from data_loader import create_dataloader
from training_loop import train


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
