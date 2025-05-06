# 📘 Hướng dẫn sử dụng hàm `main()` với một input DataFrame

Hàm `main(dataframe)` được sử dụng để huấn luyện mô hình dự đoán giá cổ phiếu dựa trên dữ liệu lịch sử. Để hoạt động đúng, DataFrame cần được chuẩn hóa theo định dạng cụ thể.

---

## ✅ Yêu cầu đầu vào

DataFrame truyền vào hàm `main()` phải có định dạng sau:

| time         | close | ticker |
|--------------|-------|--------|
| 2024-01-02   | 32100 | CTG    |
| 2024-01-03   | 32300 | CTG    |
| ...          | ...   | ...    |

### 📌 Trong đó:
- `time`: ngày giao dịch (kiểu `datetime` hoặc chuỗi định dạng `YYYY-MM-DD`)
- `close`: giá đóng cửa trong ngày
- `ticker`: mã cổ phiếu tương ứng (CTG, VNM, v.v.) — cần thiết để hàm `groupby()` trong `main()` hoạt động đúng

---

## ✅ Cách đọc và tiên xử lý file CSV

Giả sử bạn đã tải dữ liệu từ vnstock và có file CSV như sau:

```python
import pandas as pd
from main_model import main  # thay thành tên file của bạn

# Đọc file
df = pd.read_csv("./misc/CTG_2024_2025.csv")

# Thêm cột ticker nếu chưa có
df['ticker'] = 'CTG'

# Gọi hàm main
main(df)
```

---

## ✅ Dữ liệu nhiều mã cổ phiếu

Nếu bạn dử đoán nhiều mã (CTG, VNM, VCB...), DataFrame cần có dạng sau:

| time         | close | ticker |
|--------------|-------|--------|
| 2024-01-02   | 32100 | CTG    |
| 2024-01-02   | 72000 | VNM    |
| 2024-01-02   | 91500 | VCB    |

---

## ⚠️ Lưu ý quan trọng

- Cột `time` sẽ được dùng làm **index thời gian** trong pipeline.
- Cố gắng đảm bảo các giá trị `close` là số thực (float/int)
- Nếu bạn dùng DataFrame tự tạo, đảm bảo có đủ:
  - Cột `close`
  - Cột thời gian tên là `time` hoặc index đã được chuyển sang datetime

---

## 🔧 Tùy chỉnh khác

Muốn thay đổi số ngày quan sát (sequence), chỉnh tham số sau trong `main()`:

```python
sequence_length = 3  # có thể thay = 5 hoặc 10
```

---

## 📁 Thư mục quan trọng

- `CTG_2024_2025.csv` : file dữ liệu dầu vào
- `main_model.py` : chứa hàm `main()`
- `data/process.py` : chứa class `process`
- `./scalers/` : thư mục lưu scaler đã fit theo mã
- `./saved_model/` : thư mục lưu model

---

## 🧪 Chạy demo nhanh

```bash
python3 main_model.py
```
---

## 📥 Cách sử dụng `predict_future()`

```python
from predict import predict_future

df_result = predict_future(
    ticker="CTG",                 # Mã cổ phiếu
    n_days_future=7,              # Số ngày muốn dự báo
    sequence_length=3,           # Độ dài chuỗi lịch sử dùng làm đầu vào
    model_path="./saved_model/model.pth"   # Đường dẫn mô hình đã huấn luyện
)

print(df_result)
```

### 📤 Kết quả

Hàm sẽ trả về một `DataFrame` có dạng:

| date       | predicted_price |
|------------|-----------------|
| 2025-04-01 | 26.45           |
| 2025-04-02 | 26.73           |
| ...        | ...             |

---