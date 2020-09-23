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