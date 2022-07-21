from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


# rolling / expanding window

def get_cross_validation_inds(start_date, train_date, end_date, offset, mode='expanding'):

    start_date = pd.to_datetime(start_date)
    train_date = pd.to_datetime(train_date)    
    end_date = pd.to_datetime(end_date)
        
    train_dates = []
    test_dates = []
        
    while train_date + pd.offsets.Day() < end_date:

        train_dates.append([start_date, train_date])
        test_dates.append([train_date + pd.offsets.Day(), train_date + offset])  
                
        if mode == 'rolling':
            start_date = start_date + offset + pd.offsets.Day()
        
        train_date += offset
               
    return train_dates, test_dates
 
# # train/test dates
# val_train_dates, val_test_dates = get_cross_validation_inds('2013-01-01', '2014-12-31', '2021-01-01', 
#                                                             pd.offsets.YearEnd(), mode='expanding')

# lags_endogen = 5
# base_model = RandomForestClassifier()

# endogen_df = pd.DataFrame(share['returns'].values, index=share['returns'].index).dropna()
# for lag in range(1, lags_endogen + 1):
#     endogen_df['lag: {}'.format(str(lag))] = endogen_df[0].shift(int(lag))
# endogen_df = endogen_df[endogen_df.columns[1:]]

# data_dropna = pd.concat([endogen_df, pd.DataFrame(share['returns'])], axis=1).dropna()

# X_train = data_dropna[data_dropna.columns[:-1]][train_dates[0]:train_dates[-1]].values
# X_test = data_dropna[data_dropna.columns[:-1]][test_dates[0]:test_dates[-1]].values

# y_train = np.where(data_dropna['returns'][train_dates[0]:train_dates[-1]].values >= 0, 1, 0)
# y_test = np.where(data_dropna['returns'][test_dates[0]:test_dates[-1]].values >= 0, 1, 0)

# model = base_model.fit(X_train, y_train)
# probs = model.predict_proba(X_test)[:, 1]