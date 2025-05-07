import matplotlib.pyplot as plt
import numpy as np
import torch

class Plotter:
    def __init__(self, model, X_train_tensor, y_train, X_test_tensor, y_test, device="cpu"):
        self.model = model.to(device)
        self.X_train_tensor = X_train_tensor.to(device)
        self.X_test_tensor = X_test_tensor.to(device)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        self.device = device

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            y_train_pred = self.model(self.X_train_tensor).cpu().numpy().squeeze(-1)
            y_test_pred = self.model(self.X_test_tensor).cpu().numpy().squeeze(-1)
        return y_train_pred, y_test_pred
    
    def plot(self):
        y_train_pred, y_test_pred = self.predict()

        plt.figure(figsize=(12,6))

        plt.plot(range(len(self.y_train)), self.y_train, label="Actual Train Prices", color='blue')
        plt.plot(range(len(self.y_train), len(self.y_train) + len(self.y_test)), self.y_test, label="Actual Test Prices", color='red')

        plt.plot(range(len(self.y_train)), y_train_pred, label="Predicted Train Prices", linestyle='dashed', color='red')
        plt.plot(range(len(self.y_train), len(self.y_train) + len(self.y_test)), y_test_pred, label="Predicted Test Prices", linestyle="dashed", color='green')

        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.title("Stock Price Prediction with LSTM")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()