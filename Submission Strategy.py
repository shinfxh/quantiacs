import xarray as xr
import numpy as np

import qnt.ta as qnta
import qnt.data as qndata
import qnt.output as qnout
import qnt.stats as qns
import qnt.backtester as qnbt

start_date = "2005-01-01"
data = qndata.stocks.load_ndx_data(min_date = start_date)

close     = data.sel(field="close")
is_liquid = data.sel(field="is_liquid")
sma_slow  = qnta.sma(close, 200)

sma_prev = sma_slow.copy(deep = True)
n = 13
for t in range(n, len(sma_slow)):
    sma_prev[t] = sma_slow[t - n].copy(deep = True)

stocks_list = qndata.stocks.load_ndx_list(min_date = start_date)


#edit these dictionaries for classification of sectors
sector_to_ticker = {}
ticker_to_sector = {}
ticker_to_id = {}
for i in stocks_list:
    if i['symbol'] == 'GOOG' or i['symbol'] == 'GOOGL' or i['symbol'] == 'ZM' or i['symbol'] == 'JAVA' or i['symbol'] == 'AVGO':
        i['sector'] = 'IT/Telecommunications'
for i in range(len(stocks_list)):
    stock_description = stocks_list[i]
    sector = stock_description['sector']
    ticker = stock_description['id']
    sector_to_ticker[sector] = sector_to_ticker.get(sector, set()) | {ticker}
    ticker_to_sector[ticker] = ticker_to_sector.get(ticker, set()) | {sector}
    ticker_to_id[ticker] = i

sector_weights = {
'Consumer Goods': -5.05789127e-01,
'IT/Telecommunications' : 8.46884424e-01,
'Finance' : 2.47910997e+02,
'Energy' : -8.09693665e+01
}

weights_by_sector = [0 for i in range(len(data.asset))]
for sector in sector_weights:
    if sector not in sector_to_ticker:
        continue
    for ticker in sector_to_ticker[sector]:
        weights_by_sector[ticker_to_id[ticker]] = sector_weights[sector]

weights = xr.where((sma_slow - sma_prev)/(sma_slow) > 0.029, 1, 7.28334471e-02/sma_slow)
weights = weights * is_liquid * weights_by_sector
weights = qnout.clean(weights, data, "stocks_nasdaq100")

qnout.check(weights, data, "stocks_nasdaq100")
qnout.write(weights)
