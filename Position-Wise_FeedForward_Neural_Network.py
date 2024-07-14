import torch
import torch.nn as nn

# d_ff: dimension of the feed-forward layer
# droput: probability to prevent overfitting
# fc1, fc2: initializes two linear layers

class PositionWiseFeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super(PositionWiseFeedForward, self).__init__()
    self.fc1 = nn.Linear(d_model, d_ff)
    self.fc2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)
    self.relu = nn.ReLU()

# fc1 transform the dimension from d_model to d_ff
# fc2 transform the dimension from d_ff to d_model

  def forward(self, x):
    x = self.fc1(x)  # shape: (batch_size, seq_len, d_ff)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)  # shape: (batch_size, seq_len, d_model)
    return x

def test_position_wise_feed_forward():
  d_model = 512
  d_ff = 2048     #  to provide a sufficiently large capacity for the model to learn complex representations.
  seq_len = 10
  batch_size = 2

  ffn = PositionWiseFeedForward(d_model, d_ff)
  
  dummy_input = torch.randn(batch_size, seq_len, d_model)
  output = ffn(dummy_input)

  print("Input Shape:", dummy_input.shape)
  print("Output Shape:", output.shape)

test_position_wise_feed_forward()

# Output: 
# Input Shape: torch.Size([2, 10, 512])
# Output Shape: torch.Size([2, 10, 512])
