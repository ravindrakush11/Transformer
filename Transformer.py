import torch
import torch.nn as nn

class Transformer(nn.Module):
  def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_len, dropout=0.1):
    super(Transformer, self).__init__()
    self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, max_seq_len, dropout)
    self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, target_vocab_size, max_seq_len, dropout)
    self.final_layer = nn.Linear(d_model, target_vocab_size)

  def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
      enc_output = self.encoder(src, src_mask)
      dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)
      final_output = self.final_layer(dec_output)
      return final_output

class Encoder(nn.Module):
  def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, max_seq_len, dropout=0.1):
    super(Encoder, self).__init__()
    self.embedding = nn.Embedding(input_vocab_size, d_model)
    self.positional_encoding = self.positional_encoding(max_seq_len, d_model)
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

  def forward(self, x, src_mask=None):
    seq_len = x.size(1)
    x = self.embedding(x)  # Shape: (batch_size, seq_len, d_model)
    x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
    x = self.dropout(x)

    for layer in self.layers:
      x = layer(x, src_mask)
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
    x = x + self.dropout1(attn_output)
    x = self.norm1(x)

    ff_output = self.feed_forward(x)
    x = x + self.dropout2(ff_output)
    x = self.norm2(x)
    return x

class Decoder(nn.Module):
  def __init__(self, num_layers, d_model, num_heads, d_ff, target_vocab_size, max_seq_len, dropout=0.1):
    super(Decoder, self).__init__()
    self.embedding = nn.Embedding(target_vocab_size, d_model)
    self.positional_encoding = self.positional_encoding(max_seq_len, d_model)
    self.layers = nn.ModuleList([
        DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
    ])
    self.dropout = nn.Dropout(dropout)

  def positional_encoding(self, max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position *  div_term)
    pe[:, 1::2] = torch.cos(position *  div_term)
    return pe.unsqueeze(0)

  def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
    seq_len = x.size(1)
    x = self.embedding(x)   # Shape: (batch_size, seq_len, d_model)
    x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
    x = self.dropout(x)

    for layer in self.layers:
      x = layer(x, enc_output, tgt_mask, memory_mask)
    return x

class DecoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
    super(DecoderLayer, self).__init()
    self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
    self.encoder_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
    self.feed_forward = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.ReLU(),
        nn.Linear(d_ff, d_model)
    )
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

  def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
    # Self-Attention and residual connection
    attn_output, _ = self.self_attention(x, x, x, attn_mask=tgt_mask)
    x = x + self.dropout1(attn_output)
    x = self.norm1(x)

    # Encoder-Decoder Attention and residual connection
    attn_output, _ = self.encoder_attention(x, enc_output, enc_output, attn_mask=memory_mask)
    x = x + self.dropout2(attn_output)
    x = self.norm2(x)

    # Feed-Forward and residual connection
    ff_output = self.feed_forward(x)
    x = x + self.dropout3(ff_output)
    x = self.norm3(x)
    return x

# Testing of the code
def test_transformer():
  num_layers = 6
  d_model = 512
  num_heads = 8
  d_ff = 2048
  input_vocab_size = 10000
  target_vocab_size = 10000
  max_seq_len = 50
  batch_size = 2
  seq_len = 10

  transformer = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, max_seq_len)

  # Dummy input tensor
  dummy_src = torch.randint(0, input_vocab_size, (batch_size, seq_len))
  dummy_tgt = torch.randint(0, target_vocab_size, (batch_size, seq_len))

  output = transformer(dummy_src, dummy_tgt)

  print("Source Input Shape:", dummy_src.shape)
  print("Target Input Shape:", dummy_tgt.shape)
  print("Output Shape:", output.shape)

test_transformer()
