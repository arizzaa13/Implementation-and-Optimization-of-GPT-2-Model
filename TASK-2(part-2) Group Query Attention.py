import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(GroupQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def forward(self, x):
        # Reshape x for group query attention
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()  # Reshape for attention

        # Perform group query attention
        attn_scores = F.softmax(x, dim=-1)
        attn_output = torch.matmul(attn_scores, x)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        return attn_output

class GPT2SmallWithGroupQueryAttention(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_encoder_layers=12):
        super(GPT2SmallWithGroupQueryAttention, self).__init__()

        # Embedding layer with Rotary Positional Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rotary_embedding = RotaryEmbedding(d_model)

        # Group Query Attention layer
        self.group_query_attention = GroupQueryAttention(d_model, num_heads=nhead)

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

        # Apply Group Query Attention
        x = self.group_query_attention(x)

        x = self.transformer(x)
        x = self.fc(x)
        return x

# Instantiate the model with Group Query Attention
model_with_group_attention = GPT2SmallWithGroupQueryAttention(vocab_size)

# Load GPT-2 125M model checkpoints (for validation)
checkpoint_path = "path/to/gpt2_125M_checkpoint.pth"
checkpoint = torch.load(checkpoint_path)
model_with_group_attention.load_state_dict(checkpoint['model_state_dict'])

# Perform a sample prediction
input_sequence = torch.randint(0, vocab_size, (1, 10))  # Replace 10 with the desired sequence length
position = torch.arange(10).unsqueeze(0)  # Positional information
output_sequence = model_with_group_attention(input_sequence, position)
print(output_sequence)
