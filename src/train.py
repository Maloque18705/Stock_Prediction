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

        self.X_train_tensor = X_train_tensor.to(device)
        self.y_train_tensor = y_train_tensor.to(device).unsqueeze(1)

        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device).unsqueeze(1)

    def train(self, n_epochs=5000, eval_every=100):
        for epoch in range(n_epochs):
            self.model.train()
            total_loss=0

            for X_batch, y_batch in self.loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if epoch % eval_every == 0:
                self.evaluate(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            y_train_pred = self.model(self.X_train_tensor).detach().cpu().numpy()
            train_rmse = np.sqrt(
                self.loss_fn(torch.tensor(y_train_pred), self.y_train_tensor.cpu()).item()
            )

            y_test_pred = self.model(self.X_test_tensor).detach().cpu().numpy()
            test_rmse = np.sqrt(
                self.loss_fn(torch.tensor(y_test_pred), self.y_test_tensor.cpu()).item()
            )
        
        print(f"Epoch {epoch}: Train RMSE = {train_rmse:.4f} Test RMSE = {test_rmse:.4f}")

    def save(self, x):
        MODEL_PATH = Path("models")
        MODEL_PATH.mkdir(parents=True, exist_ok=True)

        MODEL_NAME = x
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

        print(f"Saving model to : {MODEL_SAVE_PATH}")
        torch.save(obj=self.model.state_dict(),
                f=MODEL_SAVE_PATH)