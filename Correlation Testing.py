import xarray as xr
import numpy as np

import qnt.ta as qnta
import qnt.data as qndata
import qnt.output as qnout
import qnt.stats as qns

dataxr = qndata.stocks.load_ndx_data(min_date = "2020-10-01")

data = dataxr.sel(field = 'close').to_numpy()
tickers = dataxr['asset'].to_numpy()
dates = dataxr['time'].to_numpy()
close = dataxr.sel(field="close").to_numpy()
close = close.transpose()
coef = np.corrcoef(close)
coef = np.nan_to_num(coef)
n = len(coef)
coef_flat = coef.flatten()
most_negative = np.argsort(coef_flat)[: 200]
most_negative = most_negative[::2]
most_negative = [(i // n, i % n) for i in most_negative]
most_negative_tickers = [(tickers[i], tickers[j]) for i, j in most_negative]

import matplotlib.pyplot as plt
def value_plot(pair):
    index1, index2 = pair
    data1 = close[index1]
    data2 = close[index2]
    fig, ax1 = plt.subplots(figsize=(5, 5))
    ax2 = ax1.twinx()
    ax1.plot(dates, data1)
    ax2.plot(dates, data2, color = 'red')
    ax1.set_ylabel(tickers[index1], fontsize=14)
    ax2.tick_params(axis = 'y', labelcolor = 'red')
    ax2.set_ylabel(tickers[index2], color = 'red', fontsize=14)
    plt.title("Correlation:" + str(coef[index1][index2]))
    
for i in range(10):
    value_plot(most_negative[i])
