import torch
import torch.nn as nn
import torch.optim as optim

class PriceModel(nn.Module):
    def __init__(self, input_size=12, sequence_length=20, hidden_size=64, dropout=0.4):
        super().__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length  # Sử dụng sequence_length thay vì seq_len
        self.hidden_size = hidden_size

        # Tính kích thước đầu vào phẳng: sequence_length * input_size
        self.flat_input_size = input_size * sequence_length

        # Xây dựng MLP với các tầng Linear
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # Kiểm tra hình dạng đầu vào
        if x.shape[1] != self.sequence_length or x.shape[2] != self.input_size:
            raise ValueError(f"Input shape must be (batch, {self.sequence_length}, {self.input_size}), got {x.shape}")

        # Làm phẳng dữ liệu từ (batch, sequence_length, input_size) thành (batch, sequence_length * input_size)
        x = x.reshape(x.size(0), -1)  # Kết quả: (batch, sequence_length * input_size)

        # Truyền qua MLP
        x = self.mlp(x)
        return x

    def config_optimizers(self, learning_rate=5e-4, weight_decay=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return optimizer, scheduler