import sys
import numpy as np
from beaker_kmer_generator import KmerGenerator as kmer_generator
import torch
from torch import nn
from torch.utils.data import DataLoader

# Input parameters are:
# 1. k
# 2. dims 
# 3. activation - relu, swish, linear
# 4. loss - huber, mae, mse
# 5. bias - True, False

# Set the model parameters here
# Number of dimensions we are encoding the vector as
k = int(sys.argv[1])
dims = int(sys.argv[2])
batch_size = 512  # Previously 128

activation = str(sys.argv[3])
loss = str(sys.argv[4])
bias_arg = sys.argv[5]

if bias_arg == "True":
    bias = True
else:
    bias = False

print("Bias is....")
print(bias)

kmer_space = 5 ** k

# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset

# Assuming beaker_kmer_generator is a module you have
from beaker_kmer_generator import KmerGenerator as kmer_generator

# Set up the data generators
kg = kmer_generator()
kg.set_threads(4)
kg.set_k(k)
kg.set_seed(42)
kg.start()

vkg = kmer_generator()
vkg.set_threads(4)
vkg.set_k(k)
vkg.set_seed(1010)
vkg.start()

# Define the dataset class
class InfiniteKmerDataset(IterableDataset):
    def __init__(self, kg):
        self.kg = kg

    def __iter__(self):
        while True:
            data = self.kg.generate_pairs()
            for i in data:
                (k1a, k2a, score) = i
                k1a = np.array(k1a, dtype=np.float32)
                k2a = np.array(k2a, dtype=np.float32)
                score = np.array(score, dtype=np.float32)
                k1a_reshaped = np.reshape(k1a, (k, 5))
                k2a_reshaped = np.reshape(k2a, (k, 5))
                yield (k1a, k2a), (score, k1a_reshaped, k2a_reshaped)

# Create datasets and data loaders
train_dataset = InfiniteKmerDataset(kg)
val_dataset = InfiniteKmerDataset(vkg)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

# Define the model
class KmerModel(nn.Module):
    def __init__(self, k, dims, activation, bias):
        super(KmerModel, self).__init__()
        input_dim = k * 5

        # Magic layer
        self.magic = nn.Linear(input_dim, dims, bias=bias)
        nn.init.normal_(self.magic.weight, mean=0.0, std=0.05)
        if bias:
            nn.init.zeros_(self.magic.bias)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # Swish activation
        elif activation == 'linear':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Reverso layer
        reverso_hidden_dim = k * 5 * 3 * dims
        self.reverso = nn.Sequential(
            nn.Linear(dims, reverso_hidden_dim),
            nn.SiLU(),  # Using Swish activation
            nn.Linear(reverso_hidden_dim, k * 5)
        )

    def forward(self, x):
        # x is of shape (batch_size, 2, k * 5)
        input1_flat = x[:, 0, :]
        input2_flat = x[:, 1, :]

        # Pass through Magic layer
        k1m = self.magic(input1_flat)
        k1m = self.activation(k1m)
        k2m = self.magic(input2_flat)
        k2m = self.activation(k2m)

        # Compute absolute difference and sum over features
        subtracted = torch.abs(k1m - k2m)
        output = torch.sum(subtracted, dim=1)  # Shape: (batch_size,)

        # Reconstruct inputs
        k1r = self.reverso(k1m)  # Shape: (batch_size, k * 5)
        k2r = self.reverso(k2m)

        # Reshape to (batch_size, k, 5)
        k1r = k1r.view(-1, k, 5)
        k2r = k2r.view(-1, k, 5)

        return output, k1r, k2r

# Instantiate the model
model = KmerModel(k, dims, activation, bias)

# Define loss functions
if loss == 'huber':
    main_loss_fn = nn.SmoothL1Loss()
elif loss == 'mae':
    main_loss_fn = nn.L1Loss()
elif loss == 'mse':
    main_loss_fn = nn.MSELoss()
else:
    raise ValueError(f"Unsupported loss function: {loss}")

reconstruction_loss_fn = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# Training loop parameters
num_epochs = 64
steps_per_epoch = 1024  # Number of steps per epoch
loss_weights = [1, 0.3, 0.3]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader):
        if step >= steps_per_epoch:
            break

        # Get inputs and targets
        (k1a, k2a), (score, k1_targets, k2_targets) = data

        # Convert to tensors
        k1a = torch.tensor(k1a, dtype=torch.float32).to(device)
        k2a = torch.tensor(k2a, dtype=torch.float32).to(device)
        score = torch.tensor(score, dtype=torch.float32).to(device)
        k1_targets = torch.tensor(k1_targets, dtype=torch.float32).to(device)
        k2_targets = torch.tensor(k2_targets, dtype=torch.float32).to(device)

        # Stack inputs
        inputs = torch.stack([k1a, k2a], dim=1)  # Shape: (batch_size, 2, k*5)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output, k1r, k2r = model(inputs)

        # Reshape outputs and targets for CrossEntropyLoss
        k1r_flat = k1r.view(-1, 5)  # Shape: (batch_size * k, 5)
        k2r_flat = k2r.view(-1, 5)

        # Convert targets to class indices
        k1_targets_indices = k1_targets.argmax(dim=2).view(-1).long()
        k2_targets_indices = k2_targets.argmax(dim=2).view(-1).long()

        # Compute losses
        loss_main = main_loss_fn(output, score)
        loss_k1r = reconstruction_loss_fn(k1r_flat, k1_targets_indices)
        loss_k2r = reconstruction_loss_fn(k2r_flat, k2_targets_indices)

        # Total loss
        total_loss = (
            loss_weights[0] * loss_main
            + loss_weights[1] * loss_k1r
            + loss_weights[2] * loss_k2r
        )

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    # Print average loss for the epoch
    avg_loss = running_loss / steps_per_epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save model weights
torch.save(
    model.state_dict(),
    f"weights/kmer_model_k{k}_dims_{dims}_activation_{activation}_loss_{loss}_bias_{bias}.pth",
)

# Save individual layers
torch.save(
    model.magic.state_dict(),
    f"weights/magic_layer_k{k}_dims_{dims}_activation_{activation}_loss_{loss}_bias_{bias}.pth",
)
torch.save(
    model.reverso.state_dict(),
    f"weights/reverso_layer_k{k}_dims_{dims}_activation_{activation}_loss_{loss}_bias_{bias}.pth",
)