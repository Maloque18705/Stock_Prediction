from src.model import PriceModel
from data.process import process
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def predict_future(ticker, n_days_future=7, sequence_length=3, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        model_path = f"./saved_model/{ticker}_model.pth"

    model = PriceModel(input_size=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    processor = process(scaler_path=f"./scalers/{ticker}_scaler.save")

    # Read DataFrame
    try:
        df = pd.read_csv("./misc/FPT_2023-01-01_2025-03-31_7D.csv")
        df = pd.DataFrame(df)

        # Thêm đoạn xử lý thời gian
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
    except ValueError as e:
        print(f"Error fetching data: {e}")
        return None

    if df.empty or len(df) <= sequence_length:
        print("Not enough data to predict.")
        return None

    df = processor.compute_technical_indicators(df)
    scaled_data = processor.data_scaling(df, fit=False)
    if scaled_data is None or len(scaled_data) == 0:
        print("No valid scaled data for prediction.")
        return None

    # Tạo DataFrame từ scaled_data
    features = ['close', 'volume', 'SMA', 'EMA', 'RSI', 'MACD', 
                'MACD_Signal', 'MACD_Hist', 'BB_High', 'BB_Low', 
                'Stoch_K', 'Stoch_D']
    scaled_df = pd.DataFrame(scaled_data, columns=features, index=df.index)

    last_sequence = scaled_df[features].values[-sequence_length:]  # 
    last_sequence = last_sequence.reshape(1, sequence_length, 12)  # (batch, seq_len, feature)
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32).to(device)

    predictions = []
    
    # model.eval()
    for _ in range(n_days_future):
        with torch.no_grad():
            pred = model(last_sequence)  # (batch, 1)
            pred_value = pred.item()
            predictions.append(pred_value)

        # cập nhật input cho lần dự đoán tiếp theo
        last_sequence = last_sequence.squeeze(0).cpu().numpy()  # (seq_len, feature)
        new_row = last_sequence[-1].copy()  # Lấy hàng cuối
        new_row[0] = pred_value  #update 'close'
        last_sequence = np.append(last_sequence[1:], [new_row], axis=0)  # (seq_len, 12)
        last_sequence = torch.tensor(last_sequence.reshape(1, sequence_length, 12), dtype=torch.float32).to(device)

    # Inverse scaling predictions
    predictions = np.array(predictions).reshape(-1, 1)
    y_pred = processor.inverse_scale(predictions, features_idx=0)

    # Tạo ngày cho dữ liệu dự báo
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days_future, freq='B')  # 'B' = business days

    result_df = pd.DataFrame({
        "date": future_dates,
        "predicted_price": y_pred.flatten()
    })

    return result_df


if __name__ == "__main__":
    df_future = predict_future(
        ticker="FPT",
        n_days_future=7,
        sequence_length=3
    )
    print(df_future)
