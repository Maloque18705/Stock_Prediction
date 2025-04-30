import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

class Optimizer:
    def __init__(self, model, X_train, y_train, batch_size=8, learning_rate=1e-3, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        
        # Hàm tối ưu hóa và hàm mất mát
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Chuyển X_train và y_train thành tensor
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(-1)  # Định dạng (batch_size, 1)

        # Dữ liệu TensorDataset và DataLoader
        self.dataset = data.TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)