from src.model import PriceModel
from data.build import process
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def predict_future(ticker, n_days_future=7, sequence_length=3, model_path="./saved_model/model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PriceModel(input_size=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    processor = process(ticker)

    # truyen dataframe tu process, khong co fetch_data nua
    try:
        df = processor.fetch_data("2023-01-01", "2025-04-28")  # lấy toàn bộ dữ liệu có sẵn
    except ValueError as e:
        print(f"Error fetching data: {e}")
        return None

    if df.empty or len(df) <= sequence_length:
        print("Not enough data to predict.")
        return None

    df = processor.data_scaling(df, fit=False)

    last_sequence = df['close'].values[-sequence_length:]  # giả sử cột giá cần dự đoán là 'Close'
    last_sequence = last_sequence.reshape(1, sequence_length, 1)  # (batch, seq_len, feature)
    last_sequence = torch.tensor(last_sequence, dtype=torch.float32).to(device)

    predictions = []
    
    model.eval()
    for _ in range(n_days_future):
        with torch.no_grad():
            pred = model(last_sequence)  # đầu ra (batch, 1)
            pred_value = pred.item()
            predictions.append(pred_value)

        # cập nhật input cho lần dự đoán tiếp theo
        last_sequence = last_sequence.squeeze(0).cpu().numpy()  # (seq_len, feature)
        last_sequence = np.append(last_sequence, pred_value)
        last_sequence = last_sequence[-sequence_length:]
        last_sequence = torch.tensor(last_sequence.reshape(1, sequence_length, 1), dtype=torch.float32).to(device)

    # Inverse scaling predictions
    predictions = np.array(predictions).reshape(-1, 1)
    y_pred = processor.inverse_scale(predictions)

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
        sequence_length=3,
        model_path="./saved_model/model.pth"
    )
    print(df_future)