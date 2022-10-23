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
coef = coef.flatten()
most_negative = np.argsort(coef)[: 100]
most_negative = [(i // n, i % n) for i in most_negative]
most_negative_tickers = [(tickers[i], tickers[j]) for i, j in most_negative]
most_negative_tickers
