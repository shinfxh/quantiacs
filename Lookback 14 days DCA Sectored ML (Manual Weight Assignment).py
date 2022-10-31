import xarray as xr
import numpy as np

import qnt.ta as qnta
import qnt.data as qndata
import qnt.output as qnout
import qnt.stats as qns

start_date = "2005-12-31"
dataxr = qndata.stocks.load_ndx_data(min_date = start_date)
data = qndata.stocks_load_ndx_data(min_date=start_date)

close     = dataxr.sel(field="close")
is_liquid = dataxr.sel(field="is_liquid")
sma_slow  = qnta.sma(close, 200)

sma_prev = sma_slow.copy(deep = True)
n = 13
for t in range(n, len(sma_slow)):
    sma_prev[t] = sma_slow[t - n].copy(deep = True)
    
stocks_list = qndata.stocks.load_ndx_list(min_date=start_date)


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
    

sma_tst = (sma_slow - sma_prev)/(sma_slow)
sectors = list(sector_to_ticker.keys())

def get_sharpe(market_data, weights):
    rr = qns.calc_relative_return(market_data, weights)
    sharpe = qns.calc_sharpe_ratio_annualized(rr).values[-1]
    return sharpe
  
def strategy(params):
    sector_weights = {
    'Consumer Goods': params[0],
    'IT/Telecommunications' : params[1],
    'Finance' : params[2],
    'Energy' : params[3]
    }
    
    weights_by_sector = [0 for i in range(len(dataxr.asset))]
    for sector in sector_weights:
        for ticker in sector_to_ticker[sector]:
            weights_by_sector[ticker_to_id[ticker]] = sector_weights[sector]
    
    weights = xr.where((sma_slow - sma_prev)/(sma_slow) > 0.029, 1, params[4]/sma_slow)
    
            
    weights = weights * is_liquid * weights_by_sector

    return weights

def performance(params):
    return get_sharpe(dataxr, strategy(params))
  
from scipy.optimize import minimize
guess = [-5.05789127e-01,  8.46884424e-01,  2.47910997e+02, -8.09693665e+01,
        7.28334471e-02]
print(performance(guess))
weights = strategy(guess)
qnout.check(weights, data, "stocks_nasdaq100")
qnout.write(weights)
stats = qns.calc_stat(data, weights.sel(time=slice("2006-01-01",None)))
stats.to_pandas().tail()
