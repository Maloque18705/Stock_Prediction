from src.model import PriceModel
from data.process import process
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def predict_future(ticker, dataframe, n_days_future=7, sequence_length=3, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        model_path = f"./saved_model/{ticker}_model.pth"

    model = PriceModel(input_size=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    processor = process(ticker)

    # Process the input dataframe
    try:
        df = pd.DataFrame(dataframe)

        # Handle time column
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
    except ValueError as e:
        print(f"Error processing data: {e}")
        return None

    if df.empty or len(df) <= sequence_length:
        print("Not enough data to predict.")
        return None

    original_close = df['close'].copy()
    df = processor.data_scaling(df, fit=False)

    last_sequence = df['close'].values[-sequence_length:]  # Assuming 'close' is the target column
    last_sequence = last_sequence.reshape(1, sequence_length, 1)  # (batch, seq_len, feature)
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32).to(device)

    predictions = []
    
    model.eval()
    for _ in range(n_days_future):
        with torch.no_grad():
            pred = model(last_sequence)  # Output (batch, 1)
            pred_value = pred.item()
            predictions.append(pred_value)

        # Update input for the next prediction
        last_sequence = last_sequence.squeeze(0).cpu().numpy()  # (seq_len, feature)
        last_sequence = np.append(last_sequence, pred_value)
        last_sequence = last_sequence[-sequence_length:]
        last_sequence = torch.tensor(last_sequence.reshape(1, sequence_length, 1), dtype=torch.float32).to(device)

    # Inverse scaling predictions
    predictions = np.array(predictions).reshape(-1, 1)
    y_pred = processor.inverse_scale(predictions)

    # Create dates for forecasted data
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days_future, freq='B')  # 'B' = business days

    result_df = pd.DataFrame({
        "date": future_dates,
        "predicted_price": y_pred.flatten()
    })

    # Plot historical and predicted data
    plt.figure(figsize=(12, 6))
    # Plot historical close prices
    plt.plot(original_close.index, original_close.values, label='Historical Close Price', color='blue')
    # Plot predicted prices
    plt.plot(future_dates, y_pred.flatten(), label='Predicted Price', color='red', linestyle='--')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'./plot_predictions/{ticker}_price_prediction.png')

    return result_df

if __name__ == "__main__":
    # Load the CSV file into a DataFrame
    df = pd.read_csv("./misc/TCB_2018-01-01_2025-03-31_1D.csv")
    df_future = predict_future(
        ticker="TCB",
        dataframe=df,
        n_days_future=15,
        sequence_length=3
    )
    print(df_future)