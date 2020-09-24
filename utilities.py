import functools
import datetime
import numpy as np
from scipy.stats import linregress

def timer(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        fn_return = fn(*args, **kwargs)
        end = datetime.datetime.now()
        print('Function {} took {} time'.format(fn.__name__, end - start))
        return fn_return
    return wrapper

# only works for single columns for now. improve in the future
def get_slope(x):
    x = np.ravel(x)
    return linregress(range(len(x)), x)[0]

def get_rmsse(train, test, predict):
    if len(predict) != len(test):
        print('train and test sets have different length')
        return
    if type(predict) != np.ndarray:
        predict = predict.values
    numerator = np.sum((test.values - predict)**2)
    denominator = np.mean((train.shift(-1) - train)**2)
    rmsse = (numerator / denominator / len(test))**0.5
    return rmsse

def save_df_to_pickle(df, file_loc):
    df.to_pickle(file_loc)
    print('df saved to {}'.format(file_loc))

def time_series_train_test_split(y, test_split):
    if test_split <= 0:
        raise ValueError('test_split cannot be <= 0')
    y_len = len(y)
    if test_split < 1: # treat as ratio
        return y[:int(y_len * (1 - test_split))], y[int(y_len * (1 - test_split)):]
    else: # treat as count
        if test_split >= y_len:
            raise ValueError('test_split cannot be greater than number of samples')
        return y[:-test_split], y[-test_split:]
