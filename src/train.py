import torch
import numpy as np

class Trainer:
    def __init__(self, model, optimizer, loss_fn, loader, X_train_tensor, y_train_tensor, X_val, y_val, X_test, y_test, device, patience=15, eval_every=10):
        self.model = model.to(device)
        self.optimizer = optimizer  # Đây là optimizer PyTorch (torch.optim.Adam)
        self.scheduler = optimizer.scheduler if hasattr(optimizer, 'scheduler') else None
        self.loss_fn = loss_fn
        self.loader = loader
        self.X_train_tensor = X_train_tensor
        self.y_train_tensor = y_train_tensor
        self.X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        self.y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device).unsqueeze(1)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device).unsqueeze(1)
        self.device = device
        self.patience = patience
        self.eval_every = eval_every

    def train(self, n_epochs):
        best_val_rmse = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(n_epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in self.loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(self.loader.dataset)
            train_rmse = np.sqrt(train_loss)

            # Evaluate on validation set
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.X_val_tensor)
                val_loss = self.loss_fn(val_outputs, self.y_val_tensor).item()
                val_rmse = np.sqrt(val_loss)

            # Update scheduler
            if self.scheduler:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_rmse)
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"Epoch {epoch}: Learning rate reduced from {old_lr} to {new_lr}")

            # Evaluate on test set for reporting
            with torch.no_grad():
                test_outputs = self.model(self.X_test_tensor)
                test_loss = self.loss_fn(test_outputs, self.y_test_tensor).item()
                test_rmse = np.sqrt(test_loss)

            if epoch % self.eval_every == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}, Test RMSE = {test_rmse:.4f}")

            # Early Stopping
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}. Best Val RMSE: {best_val_rmse:.4f}")
                    self.model.load_state_dict(best_model_state)
                    break

        # Final evaluation on test set
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(self.X_test_tensor)
            test_loss = self.loss_fn(test_outputs, self.y_test_tensor).item()
            test_rmse = np.sqrt(test_loss)
        print(f"Final Test RMSE: {test_rmse:.4f}")

        return best_val_rmse

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.X_test_tensor).cpu().numpy()  # Sửa lỗi cú pháp
        return predictions