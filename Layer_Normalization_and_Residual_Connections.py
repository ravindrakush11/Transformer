import torch
import torch.nn as nn

class ResidualConnectionLayerNorm(nn.Module):
  def __init__(self, d_model, dropout=0.1):
    super(ResidualConnectionLayerNorm, self).__init__()
    self.layer_norm = nn.LayerNorm(d_model)    # Initializes layer normalization
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x, sublayer):
    return x + self.dropout(self.layer_norm(sublayer(x)))

# Testing of Residual Connection
def test_residual_connection_layer_norm():
  d_model = 512
  seq_len = 10
  batch_size = 2

  residual_layer = ResidualConnectionLayerNorm(d_model)
  dummy_input = torch.randn(batch_size, seq_len, d_model)

  sublayer = nn.Linear(d_model, d_model)  # Simple linear layer
  output = residual_layer(dummy_input, sublayer)

  print("Input Shape:", dummy_input.shape)
  print("Output Shape:", output.shape)

test_residual_connection_layer_norm()

# Output: 
# Input Shape: torch.Size([2, 10, 512])
# Output Shape: torch.Size([2, 10, 512])
