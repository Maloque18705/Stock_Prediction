from vnstock import Vnstock
import pandas as pd
import os
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator

def fetch_data(ticker, start_date, end_date):
    stock = Vnstock().stock(ticker)
    df = stock.quote.history(start = start_date, end = end_date)
    return df[['close', 'volume', 'high', 'low']]

def compute_technical_indicators(df):
    sma = SMAIndicator(close=df['close'], window = 14)
    ema = EMAIndicator(close=df['close'], window = 15)
    rsi = RSIIndicator(close=df['close'], window = 14)
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    stoch = StochasticOscillator(close=df['close'], high=df['high'], low=df['low'], window = 14)

    df['SMA'] = sma.sma_indicator()
    df['EMA'] = ema.ema_indicator()
    df['RSI'] = rsi.rsi()
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    return df

if __name__ == "__main__":
    ticker = "FPT"
    start_date = "2023-01-01"
    end_date = "2025-04-01"
    
    df = fetch_data(ticker, start_date, end_date)
    df = compute_technical_indicators(df)
    if df is not None:
        df = df.dropna()
    print(df.head(50))
