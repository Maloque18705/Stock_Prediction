import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

class Optimizer:
    def __init__(self, model, X_train, y_train, batch_size, learning_rate, device, weight_decay):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Check X_train and y_train shape
        if len(X_train.shape) != 3:
            raise ValueError(f"X_train must have shape (samples, seq_len, input_size), got {X_train.shape}")
        if X_train.shape[2] != model.input_size:
            raise ValueError(f"X_train input_size {X_train.shape[2]} does not match model input_size {model.input_size}")
        if X_train.shape[1] != model.sequence_length:
            raise ValueError(f"X_train sequence_length {X_train.shape[1]} does not match model sequence_length {model.sequence_length}")
        
        if len(y_train.shape) != 1:
            raise ValueError(f"y_train must have shape (samples,), got {y_train.shape}")

        # Hàm tối ưu hóa và hàm mất mát
        self.optimizer, self.scheduler = model.config_optimizers(learning_rate=learning_rate, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()
        
        # Chuyển X_train và y_train thành tensor
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(1)

        # Dữ liệu TensorDataset và DataLoader
        self.dataset = data.TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)