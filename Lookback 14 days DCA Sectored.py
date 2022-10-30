import xarray as xr
import numpy as np

import qnt.ta as qnta
import qnt.data as qndata
import qnt.output as qnout
import qnt.stats as qns

start_date = "2006-01-01"
dataxr = qndata.stocks.load_ndx_data(min_date = start_date)

close     = dataxr.sel(field="close")
is_liquid = dataxr.sel(field="is_liquid")
sma_slow  = qnta.sma(close, 200)

sma_prev = sma_slow.copy(deep = True)
n = 14
for t in range(n, len(sma_slow)):
    sma_prev[t] = sma_slow[t - n].copy(deep = True)
    
stocks_list = qndata.stocks.load_ndx_list(min_date=start_date)

sector_to_ticker = {}
ticker_to_sector = {}
ticker_to_id = {}
for i in range(len(stocks_list)):
    stock_description = stocks_list[i]
    sector = stock_description['sector']
    ticker = stock_description['id']
    sector_to_ticker[sector] = sector_to_ticker.get(sector, set()) | {ticker}
    ticker_to_sector[ticker] = ticker_to_sector.get(ticker, set()) | {sector}
    ticker_to_id[ticker] = i

sectors = list(sector_to_ticker.keys())
is_sector = [0 for i in range(len(dataxr.asset))]
sector = sectors[1] #change this to change sector
for ticker in sector_to_ticker[sector]:
    is_sector[ticker_to_id[ticker]] = 1
sector = sectors[3] #change this to change sector
for ticker in sector_to_ticker[sector]:
    is_sector[ticker_to_id[ticker]] = 1
    
weights = xr.where((sma_slow - sma_prev)/(sma_slow) > 0.03, 3, 0.05/sma_slow)

weights = weights * is_liquid * is_sector

# calc stats
weights = qnout.clean(weights, dataxr, "stocks_nasdaq100")
qnout.check(weights, dataxr, "stocks_nasdaq100")
qnout.write(weights)
stats = qns.calc_stat(dataxr, weights.sel(time=slice(start_date,None)))

# graph
performance = stats.to_pandas()["equity"]
import qnt.graph as qngraph

qngraph.make_plot_filled(performance.index, performance, name="PnL (Equity)", type="log")
