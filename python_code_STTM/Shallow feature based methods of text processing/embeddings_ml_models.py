from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from navec import Navec

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
    
def embed_word2vec(x):
    ar = []
    for i in x:
        ar.append(np.mean([my_dict[j] for j in i], axis=0))
    return np.mean(ar, axis=0)

def embed_navec(x):
    ar = []
    for i in x:
        ar.append(np.mean([navec[j] if j in navec else 0 for j in i], axis=0))
    return np.mean(ar, axis=0)

def embed_ft(x):
    ar = []
    for i in x:
        ar.append(np.mean([my_dict_ft[j] for j in i], axis=0))
    return np.mean(ar, axis=0)

def _train_test_split(embeddings):
    
    emb_df = pd.DataFrame(embeddings)
    emb_df = emb_df.reset_index()
    emb_df = pd.concat([emb_df, pd.DataFrame(emb_df)['prerpoc'].apply(pd.Series)], axis=1).drop('prerpoc', 1)
    emb_df = emb_df.set_index('issuedate')

    train_dict = {}
    test_dict = {}

    for ticker in returns.keys():

        data = emb_df.loc[returns[ticker].index.min():]
        n_years = data.index.year.nunique()
        data['year'] = data.index.year

        train_indx = []
        test_indx = []
        train = []
        test = []

        if n_years == 2:
            start_year = data.index.year.unique()[0]
            end_year = data.index.year.unique()[1]
            train.append([data[data['year'] == start_year]])
            test.append([data[data['year'] == end_year]])
            train = [item for sublist in train for item in sublist]
            test = [item for sublist in test for item in sublist]

        elif n_years > 2:
            for i in range(data.index.year.nunique() - 2):
                train_indx.append(list(range(data.index.year.unique()[0], data.index.year.unique()[i+2])))
                test_indx.append([data.index.year.unique()[i+2]])

        for years in train_indx:
            train.append(data[data.index.year.isin(years)])
        for year in test_indx:
            test.append(data[data.index.year.isin(year)])

        train_dict[ticker] = train
        test_dict[ticker] = test
        
    for ticker in returns.keys():
        for i in range(len(train_dict[ticker])):
            train_dict[ticker][i] = pd.concat([train_dict[ticker][i], returns[ticker].loc[train_dict[ticker][i].index]], 1)
        for j in range(len(test_dict[ticker])):
            test_dict[ticker][j] = pd.concat([test_dict[ticker][j], returns[ticker].loc[test_dict[ticker][j].index]], 1)

    for ticker in returns.keys():
        for i in range(len(train_dict[ticker])):
            train_dict[ticker][i] = train_dict[ticker][i].drop('year', 1)
        for j in range(len(test_dict[ticker])):
            test_dict[ticker][j] = test_dict[ticker][j].drop('year', 1)
            
    train_dict = {k: v for k, v in train_dict.items() if v}
    test_dict = {k: v for k, v in test_dict.items() if k in train_dict.keys()}
        
    return train_dict, test_dict

def get_expanding_predictions(classifier, train_dict, test_dict):
    preds = []
    
    for ticker in train_dict.keys():
        n_samples = len(train_dict[ticker])
        _preds = []
        classifier = classifier

        for sample in range(n_samples):
            clf = clone(classifier)
            X_train = train_dict[ticker][sample].dropna().drop(ticker, 1)
            y_train = train_dict[ticker][sample].dropna()[ticker]
            X_test = test_dict[ticker][sample].dropna().drop(ticker, 1)
            
            clf.fit(X_train, y_train),
            _preds.append(clf.predict_proba(X_test)[:, 1])
            
        pr = np.hstack(_preds)
        _test_list = pd.concat(test_dict[ticker])
        preds.append(pr)

    preds = dict(zip(train_dict.keys(), preds))
    
    for ticker in preds.keys():
        preds[ticker] = pd.concat([pd.DataFrame(pd.concat(test_dict[ticker]).index), 
                                   pd.DataFrame(preds[ticker])], 1).set_index('issuedate').rename({0:ticker}, axis=1)
    
    preds = pd.concat(preds, axis=1)
    preds.columns = preds.columns.get_level_values(0)
    
    return preds


# w2v = Word2Vec(news['preproc'], seed=7, workers=1, min_count=1)
# my_dict = {}
# for idx, key in enumerate(w2v.wv.index_to_key):
#     my_dict[key] = w2v.wv[key]

# emb = news.set_index('issuedate').resample('1W')['preproc'].apply(embed_word2vec)
# train_dict, test_dict = _train_test_split(emb)
# preds_exp_gbm = get_expanding_predictions(GradientBoostingClassifier(), train_dict, test_dict)

