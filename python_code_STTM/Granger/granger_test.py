from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from scipy.ndimage.interpolation import shift

def get_granger_casuality(timeSeries, timeSeriesMatrix, params={'stat_test': 'ssr_chi2test',
                                                           'maxlag': 5,
                                                           'stationary':True, 
                                                           'ad_fuller_p_value_level':0.05}):
    assert timeSeries.shape == (timeSeriesMatrix.shape[0],)
    
    stat_test = params.get('stat_test')
    stationary = params.get('stationary')
    ad_fuller_p_value_level = params.get('ad_fuller_p_value_level')
    maxlag = params.get('maxlag')
    
    X = timeSeriesMatrix
    target = timeSeries.values
    
    nan_inds = np.unique(np.argwhere(np.isnan(target))[:, 0])
    nan_inds = np.append(nan_inds, np.unique(np.argwhere(np.isnan(X))[:, 0]))
    target = np.delete(target, nan_inds)
    X = np.delete(X, nan_inds, axis=0)
    
    coef = np.zeros(X.shape[1])
    coef2 = np.zeros(X.shape[1])
    opt_lag = np.zeros(X.shape[1])
    
    for i in range(X.shape[1]):
        
        if stationary:
            ad_fuller_p_value_level = params.get('ad_fuller_p_value_level')
            
            x1 = (target - shift(target, 1, cval=np.nan))[1:]
            x2 = (X[:, i] - shift(X[:, i], 1, cval=np.nan))[1:]
            
            if (adfuller(x1)[1] > ad_fuller_p_value_level) or (adfuller(x2)[1] > ad_fuller_p_value_level):
                x1 = (x1 - shift(x1, 1, cval=np.nan))[1:]
                x2 = (x2 - shift(x2, 1, cval=np.nan))[1:]
                
            if (adfuller(x1)[1] > ad_fuller_p_value_level) or (adfuller(x2)[1] > ad_fuller_p_value_level):
                x1 = (x1 - shift(x1, 1, cval=np.nan))[1:]
                x2 = (x2 - shift(x2, 1, cval=np.nan))[1:]

            assert adfuller(x1)[1] <= ad_fuller_p_value_level
            assert adfuller(x2)[1] <= ad_fuller_p_value_level
            
            granger_dict = grangercausalitytests(pd.DataFrame([x1, x2]).T, 
                                                 maxlag=maxlag, verbose=False)
        else:
            granger_dict = grangercausalitytests(pd.DataFrame([target, X[:, i]]).T, 
                                                 maxlag=maxlag, verbose=False)
            
        ind = np.argmin([np.abs(np.round(granger_dict[lag][0][stat_test][1], 4)) 
                          for lag in range(1, maxlag + 1)])
        coef[i] = np.round(granger_dict[ind + 1][0][stat_test][1], 4) 
        coef2[i] = np.round(granger_dict[ind + 1][0][stat_test][0], 4) 
                          
     
        
        opt_lag[i] = 1 + np.argmin([np.abs(np.round(granger_dict[lag][0][stat_test][1], 4)) 
                                    for lag in range(1, maxlag + 1)])
        
    return coef2, coef, opt_lag


# corr_value_granger, p_value_granger, opt_lag = get_granger_casuality(share_timeseries.shift(1)[test_ind], 
#                                                                      sttm_outputs[test_ind], 
#                                                                      params={'stat_test': 'ssr_chi2test', 
#                                                                              'maxlag': 5, 
#                                                                              'stationary': True, 
#                                                                              'ad_fuller_p_value_level': 0.05})
