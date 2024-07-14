import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
    super(EncoderLayer, self).__init__()
    self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
    self.feed_forward = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Linear(d_ff, d_model)
    )
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

# The first x is the query.
# The second x is the key.
# The third x is the value.

  def forward(self, x, src_mask=None):
    # Self-attention and residual connection
    attn_output, _ = self.self_attention(x, x, x, attn_mask = src_mask)
    x = x + self.dropout1(attn_output)
    x = self.norm1(x)

    # Feed-forward and residual connection
    ff_output = self.feed_forward(x)
    x = x + self.dropout2(ff_output)
    x = self.norm2(x)
    return x

# Testing of encoder layer
def test_encoder_layer():
  d_model = 512
  num_heads = 8
  d_ff = 2048
  seq_len = 10
  batch_size = 2

  encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
  dummy_input = torch.randn(seq_len, batch_size, d_model)

  # Apply the encoder layer
  output = encoder_layer(dummy_input)

  print("Input Shape:", dummy_input.shape)
  print("Output Shape:", output.shape)

test_encoder_layer()

# Outputs: 
# Input Shape: torch.Size([10, 2, 512])
# Output Shape: torch.Size([10, 2, 512])
