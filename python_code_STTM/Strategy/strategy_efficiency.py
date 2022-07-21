import cvxpy as cp
import cplex
import cvxopt
import empyrical as ep
import heapq
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import quandl
import re
import seaborn as sns
import scipy
import scipy.optimize as scop

from sklearn.model_selection import TimeSeriesSplit
from sklearn.covariance import ShrunkCovariance, LedoitWolf, OAS

from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

def cup(d, instrument, model, t):
    try:
        return d[instrument][model]['preds'].loc[t].values[0]
    except Exception as e:
        return np.nan

def get_invest_inform(d, times, models, instruments, stocks_close, 
                      stocks_open, baseline, gcurve, nlargest=0.5, get_strategy=False):
    portfolio = {}
    
    if get_strategy:
        strategy = {}
    
    for model in models:
        oos_portfolio = []
        
        if get_strategy:
            strategy_df = pd.DataFrame(np.zeros(shape=(len(instruments), len(times[:-1]))), 
                                index=instruments, 
                                columns=times[:-1]).astype(int)
        
        for endweek_1, endweek_2 in zip(times[:-1], times[1:]):
            # forecast by model m
            frcst = {inst: cup(d, inst, model, endweek_1) for inst in instruments}
            frcst = {key:val for key, val in frcst.items() if ~np.isnan(val)}
        
            buy_stocks = heapq.nlargest(int(np.round(len(frcst) * nlargest)), 
                                        frcst, key=frcst.get)
            if get_strategy:
                strategy_df.loc[buy_stocks, endweek_1] = 1
            
            # out of sample prices
            n = stocks_close.loc[endweek_1:endweek_2].shape[0]
            
            oos = stocks_close.loc[:endweek_2, buy_stocks].iloc[-(n+1):, :]
            
            # position size, price * pos=1 on the day of rebalancing
            pos = 1 / oos.iloc[0, :]
            
            # oos portfolio
                        
            oos_value = oos @ pos
            oos_returns = oos_value.pct_change().dropna()
            
            oos_portfolio.append(oos_returns)
            
        portfolio[model] = pd.concat(oos_portfolio)
        
        if get_strategy:
            strategy[model] = strategy_df

    ress={}

    for m in models:
        baseline_ret = baseline.pct_change().loc[baseline.index.intersection(portfolio[m].index)]
        ir = gcurve.loc[gcurve.index.intersection(portfolio[m].index)]['1']/252/100
        res = {}
        res['alpha'], res['beta'] = ep.alpha_beta(returns=portfolio[m],
                                                  factor_returns=baseline_ret,
                                                  risk_free=ir)

        res['maxdrawdown'] =  ep.max_drawdown(returns=portfolio[m])
        res['sharpe_ratio'] =  ep.sharpe_ratio(returns=portfolio[m], risk_free=ir)
        res['annual_volatility'] =  ep.annual_volatility(returns=portfolio[m])
        res['annual_return'] =  ep.annual_return(returns=portfolio[m])
        ress[m] = res
        
    if get_strategy:
        return pd.DataFrame(ress), portfolio, strategy
    else:
        return pd.DataFrame(ress), portfolio


def plot_strategy(strategy, figsize=(50, 10)):

    fig, ax = plt.subplots(figsize=(50, 10))
    heatmap = ax.pcolor(strategy, cmap='cool')


    x_ticks_labels = [str(strategy.columns[i])[:10] 
                      for i in range(len(strategy.columns))]

    positions_x = np.arange(1, len(x_ticks_labels), 5)
    ax.xaxis.set_major_locator(ticker.FixedLocator(positions_x))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(np.array(x_ticks_labels)[positions_x]))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='x', which='minor', width=0)
    ax.tick_params(axis='x', which='major', rotation=45)

    y_ticks_labels = strategy.index.tolist()

    positions_y = np.arange(1, len(y_ticks_labels), 1)
    ax.yaxis.set_major_locator(ticker.FixedLocator(positions_y))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(np.array(y_ticks_labels)[positions_y]))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='y', which='minor', width=0)
    ax.yaxis.tick_left()
    
    ax.set_facecolor('#F5F5F5')
    ax.grid(which='major', linestyle='-', lw=0.5, color='white')
    ax.grid(which='minor', linestyle='-', lw=0.5, color='white')

    plt.show()
    
    
# topn = 0.2
# get_strategy = True

# ress, portfolio, strategy = get_invest_inform(d_slice, times, models, instruments, 
#                                               stocks_close, stocks_open, imoex, 
#                                               gcurve, nlargest=topn, get_strategy=get_strategy)