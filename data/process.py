import datetime
import pandas as pd
import os
import joblib
from data.dataloader import stock_data_fetcher
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class process:
    # Train theo dataframe cua tung ma
    # Khong lay du lieu tu dataloader nay nua ma process se xu ly tu ticker luon (chi lay symbol, trading_date,close)
    def __init__(self, scaler_path=None):
        # stock_data_fetcher.__init__(self, ticker)
        self.scaler = None
        self.scaler_path = scaler_path or f"./scalers/scaler.save"

    def fetch_data(self, dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        df = dataframe.copy()
        
        # Kiểm tra xem cột 'time' có tồn tại không, nếu không thì giả sử index là thời gian
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
        
        # Chỉ giữ lại các cột cần thiết (giả sử ít nhất có cột 'close')
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        return df

    def data_scaling(self, dataframe, fit=True):
        os.makedirs('./scalers', exist_ok=True)

        close_prices = dataframe[["close"]].values

        if fit or not os.path.exists(self.scaler_path):
            self.scaler = MinMaxScaler()
            scaled_close = self.scaler.fit_transform(close_prices)
            joblib.dump(self.scaler, self.scaler_path)
        else:
            self.scaler = joblib.load(self.scaler_path)
            scaled_close = self.scaler.transform(close_prices)
        
        dataframe["close"] = scaled_close.flatten()
        return dataframe

    def inverse_scale(self, values):
        if self.scaler:
            value = np.array(values).reshape(-1,1)
            return self.scaler.inverse_transform(values).flatten()
        else:
            raise ValueError("Scaler not initialized. Run data_scaling() first")

    @staticmethod
    def str_to_datetime(s):
        """Chuyển đổi chuỗi ngày tháng (yyyy-mm-dd) thành datetime"""
        if isinstance(s, pd.Timestamp):
            s = s.strftime('%Y-%m-%d')
        year, month, day = map(int, s.split('-'))
        return datetime.datetime(year, month, day)

    def df_to_windowed_df(self, dataframe, first_date_str, last_date_str, n=3):
        dataframe.index = pd.to_datetime(dataframe.index)  # ✅ Đảm bảo index là datetime
        first_date = self.str_to_datetime(first_date_str)
        last_date = self.str_to_datetime(last_date_str)

        target_date = first_date
        dates, X, Y = [], [], []
        last_time = False

        while True:
            # ✅ Truy vấn với datetime index
            df_subset = dataframe.loc[:target_date].tail(n + 1)

            if len(df_subset) != n + 1:
                print(f'Error: Window of size {n} is too large for date {target_date}')
                return pd.DataFrame()

            values = df_subset['close'].to_numpy()
            x, y = values[:-1], values[-1]

            dates.append(target_date)
            X.append(list(x))
            Y.append(y)

            next_dates = dataframe.loc[target_date:].index
            next_dates = next_dates[next_dates > target_date]

            if len(next_dates) < 2:
                break

            next_date = next_dates[1]

            if last_time:
                break

            target_date = next_date

            if target_date == last_date:
                last_time = True

        return pd.DataFrame({'Target_Date': dates, 'X': X, 'Y': Y})

    
    def split_data(self, dates, X, y):
        if len(dates) > 0:
            q_80 = int(len(dates) * 0.8)
            q_90 = int(len(dates) * 0.9)

            dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
            dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
            dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

            return {
                'train': (dates_train, X_train, y_train),
                'val': (dates_val, X_val, y_val),
                'test': (dates_test, X_test, y_test)
            }
        else:
            print("No data available to split.")
            return None
        
    def window_df_to_date_X_y(self, window_dataframe):
        if window_dataframe.empty:
            return np.array([]), np.array([]), np.array([])
        
        dates = window_dataframe['Target_Date'].to_numpy()

        X_list = window_dataframe['X'].to_list()
        X = np.array(X_list).reshape(len(dates), -1, 1)

        Y = window_dataframe['Y'].to_numpy()

        return dates, X.astype(np.float32), Y.astype(np.float32)
    