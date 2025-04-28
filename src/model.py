import torch
import torch.nn as nn
import torch.optim as optim

class PriceModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)  # Output shape: (batch, seq_len, hidden_size)
        x = self.linear(x[:, -1, :])  # Chỉ lấy output của timestep cuối cùng
        return x
    