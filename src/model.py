import torch
import torch.nn as nn
import torch.optim as optim

class PriceModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=50, num_layers=2, dropout=0.2, sequence_length=3):
        super().__init__()
        self.input_size = input_size
        self.seq_len = sequence_length

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            dropout=dropout if num_layers > 1 else 0, 
                            batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.output = nn.Linear(25, 1)
        

    
    def forward(self, x):
        # Check x shape
        if x.shape[1] != self.seq_len or x.shape[2] != self.input_size:
            raise ValueError(f"Input shape must be (batch, {self.seq_len}, {self.input_size}), got {x.shape}")
        
        x, _ = self.lstm(x)  # Output shape: (batch, seq_len, hidden_size)
        x = x[:, -1, :]  # Chỉ lấy output của timestep cuối cùng
        x = self.dropout(x)
        x = self.dense(x) # (batch, 2)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.output(x) # (batch, 1)
        return x

    def config_optimizers(self, learning_rate=0.001):
        return optim.Adam(self.parameters(), lr=learning_rate) 
    