# !pip install -q torchtext

import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class TransformerEmbedding(nn.Module):
  def __init__(self, vocab_size, d_model, max_len):
    super(TransformerEmbedding, self).__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoding = self.positional_encoding(max_len, d_model)

  def positional_encoding(self, max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe

  def forward(self, x):
    seq_len = x.size(1)
    x = self.embedding(x)   # (batch_size, seq_len, d_model)
    x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
    return x

# Testing of Embedding
def yield_tokens(data_iter):
  for text in data_iter:
    yield tokenizer(text)

text_data = [
    "This student is writing the component of the transformer"
    "This is a example of the embedding.",
    "Also we included the positional encoding part.",
]

# Initialized tokenizer and build vocabulary
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(yield_tokens(text_data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Encode text data into token indices
encoded_data = [[vocab[token] for token in tokenizer(text)] for text in text_data]
max_len = max(len(tokens) for tokens in encoded_data)

# Pad sequences to the same length
padded_data = [tokens + [0] * (max_len - len(tokens)) for tokens in encoded_data]
input_tensor = torch.tensor(padded_data)

# Embedding dimension and other parameters
vocab_size = len(vocab)
d_model = 512
max_len = max_len
batch_size = len(text_data)

embedding_layer = TransformerEmbedding(vocab_size, d_model, max_len)
output = embedding_layer(input_tensor)

print("Input Tensor:", input_tensor)
print("Output Tensor:", output.shape)
print("Embeddings: ", output)

# Output: 
# Input Tensor: tensor([[15, 14,  3, 18,  1,  7,  4,  1, 16,  3,  5, 10,  4,  1,  8,  2],
#         [ 6, 17, 11,  1, 13,  9, 12,  2,  0,  0,  0,  0,  0,  0,  0,  0]])
# Output Tensor: torch.Size([2, 16, 512])
# Embeddings:  tensor([[[-0.0524,  1.5336,  0.2755,  ...,  0.8744,  1.1214, -0.4775],
#          [-0.4387, -0.0494,  1.3844,  ...,  1.0103,  0.7991,  0.5231],
#          [ 1.9215, -0.0487,  1.0553,  ...,  3.0484, -0.1100,  0.1619],
#          ...,
#          [ 0.2520,  2.3302,  1.0918,  ...,  1.0663,  1.1064,  1.7172],
#          [ 0.0350, -0.6944, -0.0811,  ...,  0.1836, -0.9991, -0.0390],
#          [ 0.5751, -1.1936,  0.8093,  ...,  0.3020,  1.2177,  0.6500]],

#         [[-0.2033,  1.0191, -1.4175,  ...,  0.2857, -2.0796,  0.5166],
#          [ 0.4156,  1.9208, -0.9092,  ...,  0.5576, -0.8449, -1.4030],
#          [-0.1001, -2.3876,  0.6364,  ...,  1.5085, -1.2749, -0.2501],
#          ...,
#          [ 2.6252,  2.0756,  0.6284,  ...,  1.3934,  0.1602,  0.7245],
#          [ 3.1957,  1.3049,  1.4611,  ...,  1.3934,  0.1603,  0.7245],
#          [ 2.8553,  0.4085,  1.5993,  ...,  1.3934,  0.1604,  0.7245]]],
#        grad_fn=<AddBackward0>)
