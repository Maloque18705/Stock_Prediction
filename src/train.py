import torch
import numpy as np
from pathlib import Path

class Trainer:
    def __init__(self, model, optimizer, loss_fn, loader,
                 X_train_tensor, y_train_tensor,
                 X_test, y_test, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loader = loader
        self.device = device


        # Kiểm tra dữ liệu rỗng
        if len(X_train_tensor) == 0 or len(y_train_tensor) == 0:
            raise ValueError("Empty train dataset")
        if len(X_test) == 0 or len(y_test) == 0:
            raise ValueError("Empty test dataset")

        # Kiểm tra shape
        if len(X_train_tensor.shape) != 3:
            raise ValueError(f"X_train_tensor must have shape (samples, seq_len, input_size), nhận được {X_train_tensor.shape}")
        if len(X_test.shape) != 3:
            raise ValueError(f"X_test must have shape (samples, seq_len, input_size), nhận được {X_test.shape}")

        
        
        self.X_train_tensor = X_train_tensor.to(self.device)
        self.y_train_tensor = y_train_tensor.to(self.device)

        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device).unsqueeze(1)

    def train(self, n_epochs=1000, eval_every=100, patience=10):
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0


            for X_batch, y_batch in self.loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_loss / num_batches

            if epoch % eval_every == 0 or epoch == n_epochs-1:
                train_metric, test_metric = self.evaluate(epoch)
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Train RMSE = {train_metric:.4f}, Test RMSE = {test_metric:.4f}")

            
            # #Early stopping
            # val_loss = test_metric
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     epochs_no_improve = 0
            #     self.save(f"best_model_epoch_{epoch}.pth")
            # else:
            #     epochs_no_improve += 1
            #     if epochs_no_improve >= patience:
            #         print(f"Early stopping at epoch {epoch}")
            #         break

    def evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            y_train_pred = self.model(self.X_train_tensor)
            train_loss = self.loss_fn(y_train_pred, self.y_train_tensor)
            train_rmse = torch.sqrt(train_loss).item()

            y_test_pred = self.model(self.X_test_tensor)
            test_loss = self.loss_fn(y_test_pred, self.y_test_tensor)
            test_rmse = torch.sqrt(test_loss).item()
        
        return train_rmse, test_rmse


    def predict(self):
        """Dự đoán trên tập test và trả về y_test_pred."""
        self.model.eval()
        with torch.no_grad():
            y_test_pred = self.model(self.X_test_tensor)
            return y_test_pred.cpu().numpy()  # Chuyển về NumPy array


    def save(self, x):
        MODEL_PATH = Path("models")
        MODEL_PATH.mkdir(parents=True, exist_ok=True)

        MODEL_NAME = x
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

        print(f"Saving model to : {MODEL_SAVE_PATH}")
        torch.save(obj=self.model.state_dict(),
                f=MODEL_SAVE_PATH)