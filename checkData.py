from vnstock import Vnstock

stock = Vnstock().stock(symbol="FPT", source="VCI")
df = stock.quote.history(start='2025-04-01', end='2025-04-09')
print(df)