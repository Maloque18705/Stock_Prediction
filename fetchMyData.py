from vnstock import Vnstock
import pandas as pd

# Thông tin cần lấy
ticker = "CTG"
start_date = "2024-01-01"
end_date = "2025-03-31"
interval = "1D"  # Lấy dữ liệu theo ngày

# Lấy dữ liệu
stock = Vnstock().stock(ticker)
df = stock.quote.history(start=start_date, end=end_date)

# Lưu ra CSV
output_path = "./CTG_2024_2025.csv"
df.to_csv(output_path, index=False)

print(f"Dữ liệu đã được lưu tại {output_path}")
