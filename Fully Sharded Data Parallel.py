import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.nn import FullyShardedDataParallel
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.optim.fused_adam import FusedAdam


# Instantiate the GPT-2 model (choose one of the modified models)
model = GPT2SmallWithRotary(vocab_size)

# Initialize distributed training
dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=4, rank=0)
torch.cuda.set_device(0)
model = FullyShardedDataParallel(model.cuda())

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = FusedAdam(model.parameters(), lr=0.001)

# Training loop for FSDP
def train_fsdp(model, data_loader, criterion, optimizer, num_epochs=5):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader)}")

# Example usage
# train_fsdp(model, train_data_loader, criterion, optimizer)
