import torch
import torch.nn as nn

class GPT2Small(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_encoder_layers=12):
        super(GPT2Small, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Transformer layers
        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layers, 
            num_layers=num_encoder_layers,
        )

        # Output layer
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x

# Instantiate the model
vocab_size = 50000  # Replace with the actual vocabulary size
model = GPT2Small(vocab_size)

# Load GPT-2 125M model checkpoints (for validation)
checkpoint_path = "path/to/gpt2_125M_checkpoint.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Perform a sample prediction
input_sequence = torch.randint(0, vocab_size, (1, 10))  # Replace 10 with the desired sequence length
output_sequence = model(input_sequence)
print(output_sequence)
