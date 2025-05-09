from src.model import PriceModel
from data.process import process
import os
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def predict_future(dataframe, ticker, n_days_future, sequence_length=30, model_path=None, scaler_path=None, holidays=None, historical_days=60, initial_temp_df_size=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        model_path = f"./saved_model/{ticker}_model.pth"
    if scaler_path is None:
        scaler_path = f"./scalers/{ticker}_scaler.save"
    processor = process(scaler_path)

    # Initialize model (assuming MLP; adjust for LSTM if needed)
    try:
        model = PriceModel(input_size=12, sequence_length=sequence_length, hidden_size=32, dropout=0.5)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Read and validate DataFrame
    df = dataframe.copy()
    required_cols = ['close', 'volume', 'high', 'low']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return None

    try:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        df = df.sort_index()
    except ValueError as e:
        print(f"Error processing dataframe: {e}")
        return None

    # Check for NaN and remove
    if df[required_cols].isna().any().any():
        print(f"NaN values found in {required_cols}. Removing...")
        df = df.dropna(subset=required_cols)

    if df.empty or len(df) <= sequence_length:
        print(f"Not enough data to predict (len={len(df)}, required={sequence_length})")
        return None

    # Handle outliers in volume
    volume_threshold = df['volume'].quantile(0.99)
    df = df[df['volume'] <= volume_threshold]
    df['volume'] = np.log1p(df['volume'])

    # Compute technical indicators
    df = processor.compute_technical_indicators(df)
    scaled_data, valid_index = processor.data_scaling(df, fit=False)
    if scaled_data is None or len(scaled_data) == 0:
        print("No valid scaled data for prediction.")
        return None

    # Create DataFrame from scaled_data
    features = ['close', 'volume', 'SMA', 'EMA', 'RSI', 'MACD', 
                'MACD_Signal', 'MACD_Hist', 'BB_High', 'BB_Low', 
                'Stoch_K', 'Stoch_D']
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=valid_index)

    # Debug scaler
    print(f"Scaler min: {processor.scaler.data_min_}")
    print(f"Scaler max: {processor.scaler.data_max_}")

    # Initialize temporary DataFrame with more historical data
    initial_temp_df_size = max(initial_temp_df_size, sequence_length + 1)
    temp_df = df.tail(initial_temp_df_size).copy()
    predictions = []
    future_dates = []

    # Process holidays
    holiday_dates = []
    if holidays is not None:
        try:
            for holiday in holidays:
                if isinstance(holiday, str):
                    holiday = pd.to_datetime(holiday)
                elif not isinstance(holiday, (datetime, pd.Timestamp)):
                    raise ValueError(f"Invalid holiday format. Use 'YYYY-MM-DD' or datetime")
                holiday_dates.append(holiday)
        except Exception as e:
            print(f"Error processing holidays: {e}")
            return None

    # Generate future business days
    last_date = df.index[-1]
    if last_date in holiday_dates:
        print(f"Last date {last_date} is a holiday. Adjusting...")
        last_date = last_date - pd.Timedelta(days=1)
        while last_date in holiday_dates:
            last_date = last_date - pd.Timedelta(days=1)

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                 periods=n_days_future*2, 
                                 freq='B')  # Business days
    business_days = [date for date in future_dates if date not in holiday_dates][:n_days_future]
    if len(business_days) < n_days_future:
        print(f"Not enough business days after excluding holidays")
        return None

    # Predict future prices
    for i in range(n_days_future):
        # Debug NaN and size
        print(f"Iteration {i+1}, temp_df size before indicators: {len(temp_df)}")
        print(f"NaN in temp_df before indicators: {temp_df[required_cols].isna().sum()}")

        # Get last sequence
        if len(scaled_df) < sequence_length:
            print(f"Not enough data in scaled_df (len={len(scaled_df)}, required={sequence_length})")
            break
        last_sequence = scaled_df[features].values[-sequence_length:]
        print(f"last_sequence shape: {last_sequence.shape}")
        last_sequence = last_sequence.reshape(1, sequence_length, len(features))
        last_sequence = torch.tensor(last_sequence, dtype=torch.float32).to(device)
        print(f"last_sequence tensor shape: {last_sequence.shape}")

        # Predict
        with torch.no_grad():
            pred = model(last_sequence)  # (batch, 1)
            pred_value = pred.item()

        # Inverse scale prediction for temporary DataFrame
        temp = np.zeros((1, len(features)))
        temp[0, 0] = pred_value
        pred_unscaled = processor.scaler.inverse_transform(temp)[0, 0]

        # Append prediction to temporary DataFrame
        new_row = pd.DataFrame({
            'close': pred_unscaled,
            'volume': df['volume'].iloc[-1],  # Keep last volume
            'high': pred_unscaled,  # Approximate
            'low': pred_unscaled,   # Approximate
        }, index=[business_days[i]])
        temp_df = pd.concat([temp_df, new_row])

        # Recalculate technical indicators
        temp_df = processor.compute_technical_indicators(temp_df)
        
        # Debug NaN after indicators
        print(f"NaN in temp_df after indicators: {temp_df[features].isna().sum()}")

        # Fill NaN with forward/backward fill
        temp_df[features] = temp_df[features].ffill().bfill()

        # Scale data
        scaled_data, valid_index = processor.data_scaling(temp_df, fit=False)
        if scaled_data is None or len(scaled_data) == 0:
            print(f"No valid scaled data after iteration {i+1}. Stopping.")
            break
        scaled_df = pd.DataFrame(scaled_data, columns=features, index=valid_index)

        # Store prediction
        predictions.append(pred_unscaled)

    # Create result DataFrame
    result_df = pd.DataFrame({
        "date": business_days[:len(predictions)],
        "predicted_price": predictions
    })

    # Plot historical and predicted prices
    plt.figure(figsize=(12, 6))
    # Historical prices (last historical_days)
    historical_prices = df['close'].tail(historical_days)
    plt.plot(historical_prices.index, historical_prices, label='Historical Price', color='blue')
    # Predicted prices
    plt.plot(result_df['date'], result_df['predicted_price'], label='Predicted Price', color='red', linestyle='--')
    plt.title(f'Historical and Predicted Prices for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('./plots', exist_ok=True)
    plt.savefig(f'./plots/{ticker}_future_prediction.png')
    plt.close()

    return result_df

if __name__ == "__main__":
    df = pd.read_csv("./misc/TCB_2015-01-01_2025-03-31_1D.csv")
    holidays = []    
    df_future = predict_future(
        dataframe=df,
        ticker="TCB",
        n_days_future=30,
        sequence_length=30,
        historical_days=60,
        holidays=holidays,
        initial_temp_df_size=100
    )
    if df_future is not None:
        print(df_future)