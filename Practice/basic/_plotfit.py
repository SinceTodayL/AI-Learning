import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

stock_name = "GOOG"
ticker = yf.Ticker(stock_name)
stock_info = ticker.history("3mo")
y_open = stock_info["Open"]
y_close = stock_info["Close"]
y_change = (y_close - y_open) / y_open
x = list(range(1, y_open.size+1))

coef = np.polyfit(x, y_change, 6)
p = np.poly1d(coef)

ticker = yf.Ticker("^IXIC")
index_info = ticker.history("3mo")
y_indexinfo_open = index_info["Open"]
y_indexinfo_close = index_info["Close"]
y_indexinfo_change = (y_indexinfo_close - y_indexinfo_open) / y_indexinfo_open

fig, ax = plt.subplots(1, 2, figsize=(16, 12))
ax[0].plot(x, y_change, color='r', linewidth=3, label=stock_name)
ax[0].plot(x, p(x), color='b', linewidth=1, label="Polyfit")
ax[0].set_xlabel("TimeIndex")
ax[0].legend()
ax[0].set_title(stock_name)
#plt.show()

ax[1].plot(x, y_indexinfo_change, color='y', linewidth=1, label="IXIC Index")
ax[1].set_title("IXIC Index")
ax[1].legend()
plt.show()


# Cov: need what?
beta_poly = np.polyfit(y_indexinfo_change, y_change, 1)
print(beta_poly)
cov_mat = np.cov(y_indexinfo_change, y_change)
cov_two = cov_mat[0][1]
var_index = np.var(y_indexinfo_change)
beta_cal = cov_two / var_index
print(beta_cal)

