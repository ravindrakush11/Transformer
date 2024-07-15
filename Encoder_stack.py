import torch
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size,max_seq_len, dropout=0.1):
    super(Encoder, self).__init__()
    self.embedding = nn.Embedding(input_vocab_size, d_model)
    self.positional_encoder = self.positional_encoding(max_seq_len, d_model)
    self.layers = nn.ModuleList([
        EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
    ])
    self.dropout = nn.Dropout(dropout)

  def positional_encoding(self, max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

  def forward(self, x):
    seq_len = x.size(1)
    x = self.embedding(x)   # shape: (batch_size, seq_len, d_model)
    x = x + self.positional_encoder[:, :seq_len, :].to(x.device)
    x = self.dropout(x)

    for layer in self.layers:
      x = layer(x)
    return x

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

  def forward(self, x, src_mask=None):
    attn_output, _ = self.self_attention(x, x, x, attn_mask=src_mask)
    x = x + self.dropout1(attn_output)
    x = self.norm1(x)

    ff_output = self.feed_forward(x)
    x = x + self.dropout2(ff_output)
    x = self.norm2(x)
    return x

# Tesing of the encoder stack
def test_encoder():
  num_layers = 6
  d_model = 512
  num_heads = 8
  d_ff = 2048
  input_vocab_size = 10000
  max_seq_len = 50
  batch_size = 2
  seq_len = 10

  encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, max_seq_len)
  
  dummy_input = torch.randint(0, input_vocab_size, (batch_size, seq_len))
  output = encoder(dummy_input)

  print("Input Shape:", dummy_input.shape)
  print("Output Shape:", output.shape)

test_encoder()

# Output:
# Input Shape: torch.Size([2, 10])
# Output Shape: torch.Size([2, 10, 512])
