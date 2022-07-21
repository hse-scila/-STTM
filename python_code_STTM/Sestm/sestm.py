import warnings
from copy import deepcopy
from itertools import repeat
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
import multiprocessing as mp

from scipy.stats import spearmanr

from src.data.frame import between_dates
from src.metrics import transform_to_rank
from src.pipeline.pipeline import time_series_split
from src.processing.processing import TextFeatures
from src.data.array import access_via_nd_index
from src.time import add_date_if_not_present, to_date


class SESTM:

    def __init__(self, min_count=None, alpha=0.05, lambda_reg=0.1, count_threshold=None, norm_by_article=True,
                 drop_neutral=False, njobs=16):
        if min_count and count_threshold:
            raise ValueError('Both min_count and count_threshold are provided')

        self.min_count = min_count
        self.count_threshold = count_threshold
        self.desired_alpha = alpha
        self.tonal_pos = None
        self.expected_count = None
        self.fi = None
        self.lambda_reg = lambda_reg
        # self.n_processes = n_processes
        self.norm_by_article = norm_by_article
        if isinstance(drop_neutral, str):
            raise ValueError('drop neutral must have bool type')
        self.drop_neutral = drop_neutral
        self.njobs = njobs

    def fit(self, word_counts, returns):
        nsamples = word_counts.shape[0]

        min_count = self.min_count or int(nsamples * self.count_threshold)
        fi = screen_words(word_counts, returns, min_count, self.norm_by_article)
        self.fi = fi
        o, self.tonal_pos = model_topics(word_counts, returns, fi, self.desired_alpha,
                                         drop_neutral=self.drop_neutral)
        self.expected_count = o
        return self

    def predict(self, word_counts):
        if (self.fi is not None) and (self.expected_count is None):
            return np.full(word_counts.shape[0], 0.5)
        args = self.tonal_pos, self.expected_count, self.lambda_reg
        if self.njobs > 1:
            return score_multiprocess(word_counts, self.njobs, *args)

        return score_articles(word_counts, *args)

    def get_charged_words_count(self, word_counts):
        return np.asarray(word_counts[:, self.tonal_pos].sum(axis=1)).reshape(-1)

    def get_charged_words_info(self):
        if self.expected_count is None:
            return pd.DataFrame()
        res = pd.DataFrame(self.expected_count, columns=['o+', 'o-'])
        res['position'] = self.tonal_pos
        res['fi'] = self.fi[self.tonal_pos]
        return res


def get_marginal_screening_stats():
    pass


def screen_words(word_counts, returns, min_count=10, norm_by_article=False):
    """
    alpha: float
        A threshold for deviation of sentiment words fraction
    count_threshold:
        A threshold to remove infrequent words
    """

    if norm_by_article:
        word_counts = word_counts > 0

    nsamples = len(returns)

    total_counts = np.sum(word_counts, axis=0)
    pos_samples = returns > 0
    pos_counts = np.sum(word_counts[pos_samples], axis=0)
    fi = pos_counts / (total_counts + 1e-6)
    pos_samples_count = np.count_nonzero(pos_samples)
    pi = pos_samples_count / nsamples
    diff = fi - pi
    if min_count:
        diff[total_counts < min_count] = 0
    return np.ravel(diff)


def _define_alpha(nmess, fi, ratio=0.75):
    fi_abs = np.abs(fi)
    idx = np.argsort(fi_abs)
    nwords = int(nmess * ratio)
    p = idx[-nwords - 1]
    alpha = np.abs(fi[p])
    idx = idx[-nwords:]
    idx = idx[(fi_abs > alpha)[idx]]
    if len(idx) == 0:
        alpha = fi_abs.max()
        return np.where(fi_abs == alpha)[0], alpha
    return idx, alpha


def model_topics(word_counts, returns, fi, alpha=0.1, drop_neutral=False):
    """
    alpha: float
        A threshold for deviation of sentiment words fraction
    """
    idx = np.abs(fi) > alpha  # sentiment_charged words
    nmess = word_counts.shape[0]
    ncharged = idx.sum()

    if nmess < ncharged:
        idx, alpha = _define_alpha(nmess, fi)
        idx = np.sort(idx)
        warnings.warn(
            f'Number of messages ({nmess}) is less than number of sentiment charged words'
            f'({ncharged}). Using alpha={alpha}')
    else:
        idx = np.where(idx)[0]

    a = word_counts[:, idx]
    si = a.sum(axis=1).reshape(-1, 1)
    hiT = (a / (si + 1e-6))

    p_hat = get_rank_returns(returns)
    if drop_neutral:
        sel = np.ravel(si > 0)
        hiT = hiT[sel]
        p_hat = p_hat[sel]

    w = np.vstack([p_hat, 1 - p_hat])

    ww = np.dot(w, w.T)
    ww_inv = np.linalg.inv(ww)
    o = hiT.T @ w.T @ ww_inv

    
    o[o < 0] = 0
    o = o / o.sum(axis=0)

    return o, idx


def get_rank_returns(returns):
    rank = (pd.Series(returns).rank() - 1)
    p_hat = rank / (len(returns) - 1)
    return p_hat


def neg_likelihood(p, s_hat, di, o, lambda_):
    p = p.item()
    P = np.array([p, 1 - p]).reshape(-1, 1)
    log = np.log(o @ P)
    sum_ = di @ log
    penalty = lambda_ * np.log(p * (1 - p))
    sum_ = sum_.item()
    return -(sum_ / s_hat + penalty)


def split_sparse_matrix(m, n_chunks):
    chunk_size = int(np.round(m.shape[0] / n_chunks))
    chunks = [m[n * chunk_size: (n + 1) * chunk_size] for n in range(n_chunks)]
    tail = m[n_chunks * chunk_size:]
    if tail.shape[0]:
        chunks.append(tail)
    return chunks


def _score_helper(x, args):
    args, kwargs = args
    return score_articles(x, *args, **kwargs)


def score_multiprocess(word_counts, n_proc=8, *args, **kwargs):
    pool = mp.Pool(n_proc)
    split = split_sparse_matrix(word_counts, n_proc)

    pool_args = zip(split, repeat((args, kwargs)))
    results = pool.starmap(_score_helper, pool_args, chunksize=1)
    return np.hstack(results)


def score_articles(word_counts, charged_words_pos, o, lambda_, tqdm=None, eps=1e-8):
    nitems = word_counts.shape[0]
    s_hat_i = np.ravel(word_counts.sum(axis=1))
    d_i = word_counts[:, charged_words_pos]
    p0 = np.array([0.5])
    p = np.zeros(nitems)
    bounds = np.array([[0.01, 0.99]])
    it = enumerate(zip(d_i, s_hat_i))

    if tqdm:
        it = tqdm(it, total=nitems)

    for i, (d, s_hat) in it:
        s_hat = s_hat.item()
        args = s_hat, d, o, lambda_
        if np.abs(s_hat) < eps:
            p[i] = 0.5
            continue
        opt_res = minimize(neg_likelihood, p0, args=args, bounds=bounds, method='L-BFGS-B')
        p_argmin = opt_res.x[0]
        p[i] = p_argmin
    return p


def save_sestm_results(results, path):
    path = Path(path)
    path.mkdir(exist_ok=True)
    for name, df in results.items():
        filename = name + '.feather'
        df.reset_index(drop=True).to_feather(path / filename)


def load_sestm_results(path):
    path = Path(path)
    pred = pd.read_feather(path / 'predicts.feather')
    # pred['close'] = pd.merge(pred.reset_index(), rates_df, on=['ticker', 'date'], how='left')['close'].values
    # pred['headline'] = messages.loc[pred.message_id].headline.values
    words_info = pd.read_feather(path / 'words_info.feather')
    return pred, words_info


def get_word_tone_by_message(word_counts, message_ids, charged_words):
    X = word_counts[message_ids][:, charged_words.position]
    X = np.asarray(X.todense())
    X = (X > 0).astype(int)
    tone = charged_words.tone
    p = np.multiply(X, tone.values).ravel()
    sel = p != 0.
    mids = np.repeat(np.asarray(message_ids), X.shape[-1])
    index = np.hstack([charged_words.index.values] * X.shape[0])

    p = p[sel]
    index = index[sel]
    mids = mids[sel]

    df = pd.concat([
        pd.DataFrame({'message_id': mids, 'tone': p}),
        charged_words.loc[index].drop(columns=['tone']).reset_index(drop=True)
    ], axis=1)
    return df


def define_sentiment_charged_tokens(pred, sestm_words_sentiment, words_count: TextFeatures, nwords=7,
                                    consider_count=False, eps=1e-6, min_tone=0.01):
    cols = ['ticker', 'split']
    info = sestm_words_sentiment.set_index(cols)

    original_index = pred.index
    pred = pred.reset_index()
    top_words = np.full(len(pred), None, dtype=object)
    words_tone = top_words.copy()

    for (ticker, split), rows in pred.groupby(cols):
        try:
            split_words_sentiment = info.loc[ticker, split]
        except KeyError:
            continue

        pos = split_words_sentiment.position.values
        tone = split_words_sentiment['o+'] - split_words_sentiment['o-']
        sel = rows[(rows.y_pred - 0.5).abs() > min_tone]
        x = words_count[sel.message_id][:, pos]
        if not consider_count:
            x = (x > 0).astype(int)
        x = np.asarray(x.todense())
        t = tone.values
        p = np.multiply(x, t)
        s = np.argsort(np.abs(p), axis=1)[:, -nwords:]
        top_pos = pos[np.ravel(s)].reshape(s.shape)
        top_tokens = access_via_nd_index(words_count.feature_names, top_pos)
        tone_arr = access_via_nd_index(tone.values, s)

        idx1 = np.indices((len(s), nwords), sparse=True)[0]
        mask = np.abs(p[idx1, s]) > eps

        for index, mask_i, tokens_i, tone_i in zip(sel.index, mask, top_tokens, tone_arr):
            top_words[index] = tokens_i[mask_i]
            words_tone[index] = tone_i[mask_i]

    result = pd.DataFrame({
        'tokens': top_words,
        'tone': words_tone,
    }, index=original_index)
    return result


def flatten_charged_words(df):
    df = df.dropna()
    data = {c: np.hstack(df[c].values) for c in df}
    items_in_row = df.iloc[0, 0].shape[0]
    index_arr = np.hstack([np.asarray(df.index).reshape(-1, 1)] * items_in_row)
    index = index_arr.reshape(-1)
    data['pred_id'] = index
    data = pd.DataFrame(data)
    return data


def _join_top_words(words, max_count=7):
    words = words[words.notna()]
    return words.apply(lambda x: ', '.join(x[:max_count]))


def merge_messages_from_one_week(mess, words_count):
    add_date_if_not_present(mess)
    vectors = []
    data_rows = []
    for date, rows in mess.resample('1d', on='datetime'):
        if not len(rows):
            continue

        vectors.append(words_count[rows.index].sum(axis=0))
        data_rows.append((date, rows.returns.iloc[0]))

    x = np.vstack(vectors)
    index_df = pd.DataFrame(data_rows, columns=['datetime', 'returns'])
    index_df['date'] = to_date(index_df.datetime)

    tickers = mess.ticker.unique()
    assert (len(tickers) == 1)

    index_df['ticker'] = tickers[0]

    return index_df, x


def model_ticker(ticker_messages, word_count, **kwargs):
    reind = word_count.positions.reindex(ticker_messages.index)
    index = reind.dropna().index
    mess = ticker_messages.loc[index]
    return get_cv_results(mess, word_count, **kwargs)


def _add_predicts(predicts_list, split_data, x_test, model, split_num, split_name):
    y_pred = model.predict(x_test)
    split_data['y_pred'] = y_pred
    split_data['split_num'] = split_num
    split_data['split'] = split_name
    split_data['charged_words'] = model.get_charged_words_count(x_test)
    y_true = split_data.returns.values
    y_true_rank = transform_to_rank(y_true)
    split_data['y_true_rank'] = y_true_rank
    predicts_list.append(split_data.reset_index())


def get_best_parameters(word_counts, data_train, data_val, config):
    X_train, y_train = get_arrays(data_train, word_counts)
    X_val, y_val = get_arrays(data_val, word_counts)

    def objective(trial: optuna.Trial):
        lamb = trial.suggest_loguniform('lambda_reg', 0.001, 0.2)
        thr = trial.suggest_float('count_threshold', 0.001, 0.08)
        params = deepcopy(config)
        params.update({
            'lambda_reg': lamb,
            'count_threshold': thr
        })
        model = SESTM(**params).fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return spearmanr(y_val, y_pred).correlation

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25)
    params = deepcopy(config)
    params.update(study.best_params)
    trials_df = study.trials_dataframe(('value', 'params'))
    return params, trials_df


def get_cv_results(mess, word_counts, return_train_results=False, dump_models=False, tune_parameters=False,
                   expanding_split=True,
                   train_split_duration_years=3,
                   **kwargs):
    words_info = []

    week_index_df, word_counts_week = merge_messages_from_one_week(mess, word_counts)

    time_splits = time_series_split(
        mess, datetime_col='datetime', train_size=train_split_duration_years, val_size=1,
        split_size='1y', expanding=expanding_split)

    predicts = []
    week_predicts = []
    train_predicts = []
    models = {}

    splits = ['train', 'val']

    ticker = mess.ticker.iloc[0]
    trials = []

    for split_num, (data_train, data_val) in enumerate(time_splits):
        x_train, returns_train = get_arrays(data_train, word_counts)
        config = kwargs

        if tune_parameters:
            config, trials_df = get_best_parameters(word_counts, data_train, data_val, kwargs)
            trials.append(trials_df.assign(split_num=split_num))

        model = SESTM(**config).fit(x_train, returns_train)

        info = model.get_charged_words_info()
        if len(info) != 0:
            words_info.append(info.assign(split=split_num))

        for split, split_data in zip(splits, (data_train, data_val)):
            x_test = word_counts[split_data.index.values]
            _add_predicts(predicts, split_data, x_test, model, split_num, split)
            _add_week_predicts(model, split_data, week_index_df, week_predicts, word_counts_week, split_num, split)

        if dump_models:
            models[split_num] = model

        if not return_train_results:
            continue

        _add_predicts(train_predicts, data_train, x_train, model, split_num, 'train')

    if not predicts:
        return None

    results = {'predicts': predicts, 'week_predicts': week_predicts, 'words_info': words_info}
    if trials:
        results['trials'] = trials

    if dump_models:
        models_dir = Path() / 'models'
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f'{ticker}.pickle'
        print('Saving model to', model_path)
        joblib.dump(models, model_path)

    if return_train_results:
        results['train_predicts'] = train_predicts

    ret = {name: pd.concat(dfs) if dfs else None for name, dfs in results.items()}
    return ret


def get_arrays(data_train, word_counts):
    x_train = word_counts[data_train.index.values]
    returns_train = data_train.returns
    return x_train, returns_train


def _add_week_predicts(model, data_test, week_index_df, week_predicts, word_counts_week, split_num, split):
    start_date = data_test.datetime.min()
    end_date = data_test.datetime.max() + pd.Timedelta(value='1W')
    cols = ['ticker', 'datetime', 'returns']
    test_mess = between_dates(week_index_df, start_date, end_date)[cols]
    x_test = word_counts_week[test_mess.index]
    _add_predicts(week_predicts, test_mess, x_test, model, split_num, split)


def model_rtsi(messages, words_count):
    return get_cv_results(messages, words_count)


class SestmResults:

    def __init__(self, path, messages, tok, words_count):
        pred, self.words_info = load_sestm_results(path)
        pred['headline'] = messages.loc[pred.message_id].headline.values

        # start = time.time()
        top_words_df = define_sentiment_charged_tokens(pred, self.words_info, words_count, nwords=20)
        # print(time.time() - start)
        pred = pd.concat([pred, top_words_df.rename(columns={'tokens': 'top_tokens'})], axis=1)
        pred.index.name = 'pred_id'
        self.pred = pred
        self._define_top_words(tok)

    def _define_top_words(self, tok):
        pred = self.pred.dropna(subset=['top_tokens'])
        tokens = pred.top_tokens
        tokens_arr = np.hstack(tokens.values)
        words_arr = tok.map_tokens(tokens_arr)
        tones_arr = np.hstack(pred.tone.values)
        length = tokens.apply(len)
        index_arr = np.repeat(pred.index, length)
        self.tokens = pd.DataFrame({
            'token': tokens_arr,
            'word': words_arr,
            'tone': tones_arr,
            'pred_id': index_arr,
        })
а