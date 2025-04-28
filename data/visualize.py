from dataloader import stock_data_fetcher
import matplotlib.pyplot as plt

def stock_plot():
    ticker = "VCB"
    start_date = "2024-01-01"
    end_date = "2025-01-01"

    fetcher = stock_data_fetcher(ticker)    
    data = fetcher.fetch_data(start_date, end_date)

    print(f"Stock data from {start_date} to {end_date}:")
    print(data.head())

    plt.figure(figsize=(12,6))
    plt.title("Close Price")
    plt.plot(data.index, data.close)
    plt.xlabel("Close Price")
    plt.ylabel("Date")
    plt.legend()
    plt.grid(True)
    plt.show()

stock_plot()