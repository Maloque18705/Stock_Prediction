from vnstock import Vnstock
import pandas as pd

class stock_data_fetcher:
    def __init__(self, ticker):
        self.ticker = ticker
    
    def fetch_data(self, start_date, end_date):
        stock = Vnstock().stock(symbol=self.ticker)
        df = stock.quote.history( start=start_date, end=end_date)

        # Convert index to datetime
        df['time'] = pd.to_datetime(df['time'])  
        df.set_index('time', inplace=True)  

        return pd.DataFrame(df)
    