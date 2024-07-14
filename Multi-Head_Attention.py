import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    self.depth = d_model // num_heads

    self.wq = nn.Linear(d_model, d_model)
    self.wk = nn.Linear(d_model, d_model)
    self.wv = nn.Linear(d_model, d_model)
    self.dense = nn.Linear(d_model, d_model)

  def split_heads(self, x, batch_size):
    x = x.view(batch_size, -1, self.num_heads, self.depth)
    return x.permute(0,2,1,3) # (batch_size, num_heads, seq_len, depth)
  
  def forward(self, query, key, value, mask = None):
    batch_size = query.size(0)

    query = self.wq(query)
    key = self.wk(key)
    value = self.wv(value)

    query = self.split_heads(query, batch_size)   # (batch_size, seq_len, d_model)
    key = self.split_heads(key, batch_size)
    value =  self.split_heads(value, batch_size)

    scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)

    # Concatenate heads and pass through final linear layer
    scaled_attention = scaled_attention.permute(0, 2, 1, 3) 
    concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model)
    output = self.dense(concat_attention)
    return output

  def scaled_dot_product_attention(self, query, key, value, mask):
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    dk = key.size(-1)
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype = torch.float32))

    if mask is not None:
      scaled_attention_logits = scaled_attention_logits + mask * -1e9

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights      

# Testing the Multi-Head Attention
def test_multi_head_attention():
  d_model = 512
  num_heads = 8
  seq_len = 10
  batch_size = 2

  mha = MultiHeadAttention(d_model, num_heads)

  query = torch.randn(batch_size, seq_len, d_model)
  key = torch.randn(batch_size, seq_len, d_model)
  value = torch.randn(batch_size, seq_len, d_model)

  mask = None

  output = mha(query, key, value, mask)
  print("Output shape: ", output.shape)

test_multi_head_attention()

# Output shape:  torch.Size([2, 10, 512])
