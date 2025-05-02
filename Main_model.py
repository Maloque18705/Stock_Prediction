import torch
import numpy as np
import os
import pandas as pd
from data.process import process
from src.model import PriceModel
from src.optimizer import Optimizer
from src.train import Trainer
from src.plot import Plotter

def main(dataframe):
 
    sequence_length = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    for ticker, ticker_data in dataframe.groupby('ticker'):
        print(f"Processing ticker : {ticker}")

        processor = process(scaler_path=f"./scalers/{ticker}_scaler.save")

        df = processor.fetch_data(ticker_data)
        df = processor.data_scaling(df, fit=not os.path.exists(processor.scaler_path))

        if df.empty:
            print(f"No data for ticker {ticker}")
            continue
    
        valid_start = df.index[sequence_length]
        valid_end = df.index[-1]
    
    
        window_df = processor.df_to_windowed_df(df, valid_start, valid_end, n=sequence_length)
        if window_df.empty:
            print("Create window failed")
            continue
    
        dates, X, y = processor.window_df_to_date_X_y(window_df)

        if len(X) == 0:
            print(f"No training samples for {ticker}")
            continue

        X = np.array(X).reshape(-1, sequence_length, 1)
        y = np.array(y)

        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
 

        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        model = PriceModel(input_size=1)
        optimizer_setup = Optimizer(model, X_train, y_train, batch_size=16, learning_rate=1e-3, device=device)


        trainer = Trainer(
            model=optimizer_setup.model,
            optimizer=optimizer_setup.optimizer,
            loss_fn=optimizer_setup.loss_fn,
            loader=optimizer_setup.loader,
            X_train_tensor=optimizer_setup.X_train_tensor,
            y_train_tensor=optimizer_setup.y_train_tensor,
            X_test=X_test,
            y_test=y_test,
            device=device,
        )

        trainer.train(n_epochs=8000, eval_every=100)
    
        os.makedirs('./saved_model', exist_ok=True)
        model_save_path = f'./saved_model/{ticker}_model.pth'
        torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    df = pd.read_csv("./misc/CTG_2024_2025.csv")
    df['ticker'] = 'CTG'
    df = pd.DataFrame(df)
    main(df)
