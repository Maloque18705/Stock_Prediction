import datetime
import pandas as pd
import os
import joblib
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
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
        self.features = ['close', 'volume', 'SMA', 'EMA', 'RSI', 'MACD', 
                         'MACD_Signal', 'MACD_Hist', 'BB_High', 'BB_Low', 
                         'Stoch_K', 'Stoch_D']


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
        
        required_cols = ['close', 'volume', 'high', 'low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing columns: {missing_cols}")
        
        return df
    
    # Computing technical indicators 
    def compute_technical_indicators(self, df):
        sma = SMAIndicator(close=df['close'], window = 14)
        ema = EMAIndicator(close=df['close'], window = 14)
        rsi = RSIIndicator(close=df['close'], window = 14)
        macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        bb = BollingerBands(close=df['close'], window=20)
        stoch = StochasticOscillator(
            close=df['close'], 
            high=df['high'], 
            low=df['low'], 
            window=14,
            smooth_window=3,
        )

        df['SMA'] = sma.sma_indicator()
        df['EMA'] = ema.ema_indicator()
        df['RSI'] = rsi.rsi()
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

        return df
    
    
    def data_scaling(self, dataframe, fit=True):
        os.makedirs('./scalers', exist_ok=True)
        
        if dataframe.empty:
            raise ValueError("Empty dataframe")
        if len(dataframe) < 10:
            raise ValueError("DataFrame under 10 rows which is not enough to normalize")

        # Kiểm tra các cột đặc trưng
        missing_cols = [col for col in self.features if col not in dataframe.columns]
        if missing_cols:
            raise KeyError(f"DataFrame missing column: {missing_cols}")

        
        data = dataframe[self.features].values
        valid_index = dataframe.index

        if np.any(np.isnan(data)):
            print("Removing NaN values")
            valid_rows = ~np.any(np.isnan(data), axis=1)
            data = data[valid_rows]
            valid_index = valid_index[valid_rows]
            if len(data) == 0:
                raise ValueError("Sau khi loại bỏ NaN, không còn dữ liệu để chuẩn hóa.")



        if fit or not os.path.exists(self.scaler_path):
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(data)
            joblib.dump(self.scaler, self.scaler_path)
        else:
            self.scaler = joblib.load(self.scaler_path)
            scaled_data = self.scaler.transform(data)
        
        # dataframe["close"] = scaled_data.flatten()
        return scaled_data, valid_index
    

    def inverse_scale(self, values, feature_idx=0):
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Run data_scaling() first")
        
        # Tạo mảng đầy đủ với số cột bằng số đặc trưng
        full_values = np.zeros((len(values), len(self.features)))
        full_values[:, feature_idx] = values.flatten()  # Chỉ điền cột được chọn
        inversed = self.scaler.inverse_transform(full_values)
        return inversed[:, feature_idx]

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
            df_subset = dataframe.loc[:target_date].tail(n + 1)

            if len(df_subset) != n + 1:
                print(f'Error: Window of size {n} is too large for date {target_date}')
                return pd.DataFrame()

            values = df_subset[self.features].to_numpy()
            x, y = values[:-1], values[-1,0]

            dates.append(target_date)
            X.append(x.tolist())
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
            q_70 = int(len(dates) * 0.7)
            q_85 = int(len(dates) * 0.85)

            dates_train, X_train, y_train = dates[:q_70], X[:q_70], y[:q_70]
            dates_val, X_val, y_val = dates[q_70:q_85], X[q_70:q_85], y[q_70:q_85]
            dates_test, X_test, y_test = dates[q_85:], X[q_85:], y[q_85:]

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
        X = np.array(X_list)
        Y = window_dataframe['Y'].to_numpy()

        return dates, X.astype(np.float32), Y.astype(np.float32)
    