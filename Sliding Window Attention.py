import torch
import torch.nn as nn


class SlidingWindowAttention(nn.Module):
    def __init__(self, window_size, attention_module):
        super(SlidingWindowAttention, self).__init__()
        self.window_size = window_size
        self.attention_module = attention_module

    def forward(self, x):
        seq_len = x.size(1)
        output = []
        for i in range(0, seq_len, self.window_size):
            window_input = x[:, i:i+self.window_size, :]
            window_output = self.attention_module(window_input)
            output.append(window_output)
        return torch.cat(output, dim=1)

class GPT2SmallWithSlidingWindowAttention(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_encoder_layers=12, window_size=4):
        super(GPT2SmallWithSlidingWindowAttention, self).__init__()

        # Embedding layer with Rotary Positional Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rotary_embedding = RotaryEmbedding(d_model)

        # Sliding Window Attention layer
        self.sliding_window_attention = SlidingWindowAttention(
            window_size, 
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
            )
        )

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

    def forward(self, x, position):
        x = self.embedding(x)
        x = self.rotary_embedding(x, position)

        # Apply Sliding Window Attention
        x = self.sliding_window_attention(x)

        x = self.transformer(x)
        x = self.fc(x)
        return x

# Instantiate the model with Sliding Window Attention
model_with_sliding_attention = GPT2SmallWithSlidingWindowAttention(vocab_size)

# Load GPT-2 125M model checkpoints (for validation)
checkpoint_path = "path/to/gpt2_125M_checkpoint.pth"
checkpoint = torch.load(checkpoint_path)
model_with_sliding_attention.load_state_dict(checkpoint['model_state_dict'])

# Perform a sample prediction
input_sequence = torch.randint(0, vocab_size, (1, 10))  # Replace 10 with the desired sequence length
position = torch.arange(10).unsqueeze(0)  # Positional information
output_sequence = model_with_sliding_attention(input_sequence, position)
print(output_sequence)
