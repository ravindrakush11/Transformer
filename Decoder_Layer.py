import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
    super(DecoderLayer, self).__init__()
    self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout = dropout)
    self.encoder_attention = nn.MultiheadAttention(d_model, num_heads, dropout = dropout)
    self.feed_forward = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Linear(d_ff, d_model),
    )
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

  def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
    attn_output, _ = self.self_attention(x, x, x, attn_mask=tgt_mask)  # Self-attention and residual connection
    x = x + self.dropout1(attn_output)
    x = self.norm1(x)

    # Encoder-Decoder Attention and residual connection
    attn_output, _ = self.encoder_attention(x, enc_output, enc_output, attn_mask=memory_mask)
    x = x + self.dropout2(attn_output)
    x = self.norm2(x)

    # Feed-forward and residual connection
    ff_output = self.feed_forward(x)
    x = x + self.dropout3(ff_output)
    x = self.norm3(x)

    return x
    
# Testing of the Decoder layer
def test_decoder_layer():
  d_model = 512
  num_heads = 8
  d_ff = 2048
  seq_len = 10
  batch_size = 2

  decoder_layer = DecoderLayer(d_model, num_heads, d_ff)

  dummy_input = torch.randn(seq_len, batch_size, d_model)  # (target sequence length, batch size, d_model)
  dummy_enc_output = torch.randn(seq_len, batch_size, d_model)

  # Apply the decoder layer
  output = decoder_layer(dummy_input, dummy_enc_output)
  print("Input Shape:", dummy_input.shape)
  print("Encoder Output Shape:", dummy_enc_output.shape)
  print("Output Shape:", output.shape)

test_decoder_layer()

# Output: 
# Input Shape: torch.Size([10, 2, 512])
# Encoder Output Shape: torch.Size([10, 2, 512])
# Output Shape: torch.Size([10, 2, 512])
