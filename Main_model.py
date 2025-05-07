import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from data.process import process
from src.model import PriceModel
from src.optimizer import Optimizer
from src.train import Trainer

def main(dataframe, ticker):
 
    sequence_length = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Processing ticker: {ticker}")

    processor = process(scaler_path=f"./scalers/{ticker}_scaler.save")

    df = pd.DataFrame(dataframe)

    ### START CHECKING ###

    required_cols = ['close', 'volume', 'high', 'low']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns for ticker {ticker}: {missing_cols}")
    
    # Removing NaN rows due to the technical indicators requirement
    if df[required_cols].isna().any().any():
        print(f"NaN values found in {required_cols} for ticker {ticker}. Removing...")
        df = df.dropna(subset=required_cols)
    
    if df.empty or len(df) < 20:
        print(f"Data too short or empty for ticker {ticker} (len={len(df)})")
    
    # Set Time as Index 
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)

    # END OF CHECK


    df = processor.compute_technical_indicators(df)
    # Scale Data
    scaled_data, valid_index = processor.data_scaling(df)
    
    if df.empty:
        print(f"No data for ticker {ticker}")


    features = ['close', 'volume', 'SMA', 'EMA', 'RSI', 'MACD', 
                'MACD_Signal', 'MACD_Hist', 'BB_High', 'BB_Low', 
                'Stoch_K', 'Stoch_D']
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=valid_index)

    # Create windows
    valid_start = scaled_df.index[sequence_length]
    valid_end = scaled_df.index[-1]
    window_df = processor.df_to_windowed_df(scaled_df, valid_start, valid_end, n=sequence_length)
    if window_df.empty:
        print("Create window failed")
        

    dates, X, y = processor.window_df_to_date_X_y(window_df)
    if len(X) == 0:
        print(f"No training samples for {ticker}")


    # Split dataset
    split_data = processor.split_data(dates, X, y)
    if split_data is None:
        print(f"Failed to split data for {ticker}")


    dates_train, X_train, y_train = split_data['train']
    dates_val, X_val, y_val = split_data['val']
    dates_test, X_test, y_test = split_data['test']


    # Check if dataset is empty
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Empty train or test data for {ticker}")

    # Model initiate
    model = PriceModel(input_size=12)
    optimizer_setup = Optimizer(model, X_train, y_train, batch_size=16, learning_rate=1e-5, device=device)


    # Trainer
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

    # Debug loss_fn trong Trainer
    print(f"Loss function type in Trainer: {type(trainer.loss_fn)}")

    # Start training
    trainer.train(n_epochs=1200, eval_every=100, patience=10)

    # Predict
    y_test_pred = trainer.predict()

    # Reverse scaling
    y_test = processor.inverse_scale(y_test, feature_idx=0)
    y_test_pred = processor.inverse_scale(y_test_pred, feature_idx=0)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label='Actual Price', color='blue')
    plt.plot(dates_test, y_test_pred, label='Predicted Price', color='red', linestyle='--')
    plt.title(f'Actual vs Predicted Prices for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot
    os.makedirs('./plots', exist_ok=True)
    plt.savefig(f'./plots/{ticker}_price_prediction.png')
    plt.close()

    # Save model
    os.makedirs('./saved_model', exist_ok=True)
    model_save_path = f'./saved_model/{ticker}_model.pth'
    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    df = pd.read_csv("./misc/TCB_2015-01-01_2025-03-31_7D.csv")
    symbol = "TCB"
    main(df, ticker=symbol)
