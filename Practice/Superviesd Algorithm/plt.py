import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

stock_name = "AAPL" 

ticker = yf.Ticker(stock_name)
stock_info = ticker.history("1mo")

# stock_info.index = stock_info.index.tz_localize(None)
# stock_info.to_excel("AAPL_StockData.xlsx")

# print(stock_info.index)
print(stock_info.columns)

y1 = stock_info["Close"]
y2 = stock_info["Volume"]
x = list(range(y1.size))

k, b = np.polyfit(x, y1, 1)
Var_price = np.var(y1)

'''
plt.figure()

plt.subplot(1, 2, 1)
plt.plot(x, y1, color='r',marker='o')
plt.xlabel("TimeIndex")
plt.ylabel("AAPL Stock Price")
plt.title("APPLE Price")

plt.subplot(1, 2, 2)
plt.plot(x, y2, color='b', marker='x')
plt.xlabel("TimeIndex")
plt.ylabel("AAPL Volume")
plt.title("AAPLE Volume")
plt.show()
'''

plt.figure()
plt.subplot(1, 2, 1)
plt.boxplot(y1)
plt.ylabel("Price")
plt.subplot(1, 2, 2)
plt.boxplot(y2)
plt.ylabel("Volumn")
plt.grid(True)
plt.show()


# normal distribution
x_n = np.linspace(-5, 5, num=1000)
t_n = - x_n**2 / 2
constant = np.sqrt(2*np.pi)
y_n = np.exp(t_n) / constant
fig, ax = plt.subplots(figsize=(15, 10))
ax.plot(x_n, y_n, color='red', linewidth=3)
ax.set_ylim(0, 0.5)
ax.set_ylabel("Normal")
ax.set_title("Normal Distri.")
plt.show()

