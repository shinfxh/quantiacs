import xarray as xr
import numpy as np

import qnt.ta as qnta
import qnt.data as qndata
import qnt.output as qnout
import qnt.stats as qns

dataxr = qndata.stocks.load_ndx_data(min_date = "2015-01-01")
data = dataxr.sel(field = 'close').to_numpy()
tickers = dataxr['asset'].to_numpy()
dates = dataxr['time'].to_numpy()
close = dataxr.sel(field="close").to_numpy()
close = close.transpose()
coef = np.corrcoef(close)[0]
print(coef)
print(tickers)
