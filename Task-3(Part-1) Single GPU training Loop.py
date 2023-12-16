import torch
import torch.nn as nn
import torch.optim as optim

# Instantiate the GPT-2 model (choose one of the modified models)
model = GPT2SmallWithRotary(vocab_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_single_gpu(model, data_loader, criterion, optimizer, num_epochs=5):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

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
# train_single_gpu(model, train_data_loader, criterion, optimizer)
