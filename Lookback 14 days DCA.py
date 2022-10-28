import xarray as xr
import numpy as np

import qnt.ta as qnta
import qnt.data as qndata
import qnt.output as qnout
import qnt.stats as qns

start_date = "2001-01-01"
dataxr = qndata.stocks.load_ndx_data(min_date = start_date)

close     = dataxr.sel(field="close")
is_liquid = dataxr.sel(field="is_liquid")
sma_slow  = qnta.sma(close, 200)

sma_prev = sma_slow.copy(deep = True)
n = 14
for t in range(n, len(sma_slow)):
    sma_prev[t] = sma_slow[t - n].copy(deep = True)
    
weights = xr.where((sma_slow - sma_prev)/(sma_slow) > 0.03, 3, 0.1/sma_slow)
weights = weights * is_liquid

# calc stats
weights = qnout.clean(weights, dataxr, "stocks_nasdaq100")
qnout.check(weights, dataxr, "stocks_nasdaq100")
qnout.write(weights)
stats = qns.calc_stat(dataxr, weights.sel(time=slice(start_date,None)))

# graph
performance = stats.to_pandas()["equity"]
import qnt.graph as qngraph

qngraph.make_plot_filled(performance.index, performance, name="PnL (Equity)", type="log")
