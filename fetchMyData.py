from vnstock import Vnstock
import pandas as pd
import os

# Thông tin cần lấy
ticker = "FPT"
start_date = "2023-01-01"
end_date = "2025-03-31"
interval = "7D"  # Lấy dữ liệu theo ngày

# Lấy dữ liệu
stock = Vnstock().stock(ticker)
df = stock.quote.history(start=start_date, end=end_date)

# Lưu ra CSV
os.makedirs("./misc", exist_ok=True)
output_path = f"./misc/{ticker}_{start_date}_{end_date}_{interval}.csv"
df.to_csv(output_path, index=False)

print(f"Dữ liệu đã được lưu tại {output_path}")
