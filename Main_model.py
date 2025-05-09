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
    # Hyperparameters
    sequence_length = 30
    input_size = 12
    hidden_size = 32
    dropout = 0.5
    batch_size = 16
    learning_rate = 5e-4
    weight_decay = 5e-3
    n_epochs = 100
    patience = 15
    eval_every = 10
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
    
    print("NaN in input data before processing:", df[required_cols].isna().sum())
    if df[required_cols].isna().any().any():
        print(f"NaN values found in {required_cols} for ticker {ticker}. Removing...")
        df = df.dropna(subset=required_cols)
    
    if df.empty or len(df) < sequence_length:
        print(f"Data too short or empty for ticker {ticker} (len={len(df)})")
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    ### END OF CHECK

    df = processor.compute_technical_indicators(df)
    print(f"Data shape after computing indicators: {df.shape}")
    
    scaled_data, valid_index = processor.data_scaling(df)
    print(f"Data shape after scaling: {scaled_data.shape}")
    print(f"Scaler min: {processor.scaler.data_min_}")
    print(f"Scaler max: {processor.scaler.data_max_}")
    
    if df.empty:
        print(f"No data for ticker {ticker}")

    features = ['close', 'volume', 'SMA', 'EMA', 'RSI', 'MACD', 
                'MACD_Signal', 'MACD_Hist', 'BB_High', 'BB_Low', 
                'Stoch_K', 'Stoch_D']
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=valid_index)

    valid_start = scaled_df.index[sequence_length]
    valid_end = scaled_df.index[-1]
    window_df = processor.df_to_windowed_df(scaled_df, valid_start, valid_end, n=sequence_length)
    if window_df.empty:
        print("Create window failed")

    dates, X, y = processor.window_df_to_date_X_y(window_df)
    if len(X) == 0:
        print(f"No training samples for {ticker}")

    split_data = processor.split_data(dates, X, y)
    if split_data is None:
        print(f"Failed to split data for {ticker}")

    dates_train, X_train, y_train = split_data['train']
    dates_val, X_val, y_val = split_data['val']
    dates_test, X_test, y_test = split_data['test']

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Empty train or test data for {ticker}")

    # Model initiate
    model = PriceModel(
        input_size=input_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        dropout=dropout
    )

    optimizer_setup = Optimizer(
        model=model,
        X_train=X_train,
        y_train=y_train,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        weight_decay=weight_decay
    )

    # Trainer
    trainer = Trainer(
        model=optimizer_setup.model,
        optimizer=optimizer_setup.optimizer,
        loss_fn=optimizer_setup.loss_fn,
        loader=optimizer_setup.loader,
        X_train_tensor=optimizer_setup.X_train_tensor,
        y_train_tensor=optimizer_setup.y_train_tensor,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        device=device,
        patience=patience,
        eval_every=eval_every
    )

    print(f"Loss function type in Trainer: {type(trainer.loss_fn)}")

    # Start training
    trainer.train(n_epochs=n_epochs)

    # Predict on all sets
    trainer.model.eval()
    with torch.no_grad():
        # Train predictions
        y_train_pred = trainer.model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
        y_train_pred = processor.inverse_scale(y_train_pred, feature_idx=0)
        # Validation predictions
        y_val_pred = trainer.model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()
        y_val_pred = processor.inverse_scale(y_val_pred, feature_idx=0)
        # Test predictions
        y_test_pred = trainer.model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        y_test_pred = processor.inverse_scale(y_test_pred, feature_idx=0)

    # Inverse scale actual values
    y_train = processor.inverse_scale(y_train, feature_idx=0)
    y_val = processor.inverse_scale(y_val, feature_idx=0)
    y_test = processor.inverse_scale(y_test, feature_idx=0)

    # Debug scaler
    print("y_train (scaled, first 10):", y_train[:10])
    print("y_train (unscaled, first 10):", y_train[:10])
    print("y_train_pred (unscaled, first 10):", y_train_pred[:10])
    print("y_val (scaled, first 10):", y_val[:10])
    print("y_val (unscaled, first 10):", y_val[:10])
    print("y_val_pred (unscaled, first 10):", y_val_pred[:10])
    print("y_test (scaled, first 10):", y_test[:10])
    print("y_test (unscaled, first 10):", y_test[:10])
    print("y_test_pred (unscaled, first 10):", y_test_pred[:10])

    # Plot
    plt.figure(figsize=(15, 6))
    # Train
    plt.plot(dates_train, y_train, label='Train Actual', color='green', alpha=0.5)
    plt.plot(dates_train, y_train_pred, label='Train Predicted', color='green', linestyle='--', alpha=0.5)
    # Validation
    plt.plot(dates_val, y_val, label='Val Actual', color='orange', alpha=0.5)
    plt.plot(dates_val, y_val_pred, label='Val Predicted', color='orange', linestyle='--', alpha=0.5)
    # Test
    plt.plot(dates_test, y_test, label='Test Actual', color='blue')
    plt.plot(dates_test, y_test_pred, label='Test Predicted', color='red', linestyle='--')
    plt.title(f'Actual vs Predicted Prices for {ticker} (Train, Val, Test)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs('./plots', exist_ok=True)
    plt.savefig(f'./plots/{ticker}_price_prediction_all.png')
    plt.close()

    os.makedirs('./saved_model', exist_ok=True)
    model_save_path = f'./saved_model/{ticker}_model.pth'
    torch.save(model.state_dict(), model_save_path)
if __name__ == "__main__":
    df = pd.read_csv("./misc/TCB_2015-01-01_2025-03-31_1D.csv")
    symbol = "TCB"
    main(df, ticker=symbol)
