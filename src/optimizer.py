import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

class Optimizer:
    def __init__(self, model, X_train, y_train, batch_size=16, learning_rate=1e-3, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size

        # Check X_train and y_train shape
        if len(X_train.shape) != 3:
            raise ValueError(f"X_train must have shape (samples, seq_len, input_size), got {X_train.shape}")
        if X_train.shape[2] != model.input_size:
            raise ValueError(f"X_train input_size {X_train.shape[2]} does not match model input_size {model.input_size}")
        
        if len(y_train.shape) != 1:
            raise ValueError(f"y_train must have shape (samples,), got {y_train.shape}")

        # Hàm tối ưu hóa và hàm mất mát
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Chuyển X_train và y_train thành tensor
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(1)  # Định dạng (batch_size, 1)

        # Dữ liệu TensorDataset và DataLoader
        self.dataset = data.TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)