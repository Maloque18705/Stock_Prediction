import torch
from data.build import process

class Main:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.processor = process(ticker)

    def run(self):
        print(f"Fetching {self.ticker} Stock Data from {self.start_date} to {self.end_date}")
        df = self.processor.fetch_data(self.start_date, self.end_date)

        if df.empty:
            print("No Data")
            return
        
        print("Data fetched!")
        print(df.head())

        n = 3
        valid_start = df.index[n]
        valid_end = df.index[-1]
        window_df = self.processor.df_to_windowed_df(df, valid_start, valid_end,n=n)
        
        if window_df.empty:
            print("Create window failed")
            return
        
        dates, X, y = self.processor.window_df_to_date_X_y(window_df)
        
        print(dates.shape, X.shape, y.shape)

        data_splits = self.processor.split_data(dates, X, y)
        if data_splits is None:
            print("Failed to split")
            return
        print("Data split completed!")
        print(f"Train size: {len(data_splits['train'][0])}")
        print(f"Validation size: {len(data_splits['val'][0])}")
        print(f"Test size: {len(data_splits['test'][0])}")

if __name__ == "__main__":
    ticker = "VCB"
    start_date = "2024-01-01"
    end_date = "2025-01-01"

    main_program = Main(ticker, start_date, end_date)
    main_program.run()
