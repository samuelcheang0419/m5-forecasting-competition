import numpy as np
import pandas as pd
import xgboost as xgb
from itertools import product
from utilities import get_rmsse

### XGBOOST ###
def get_dmatrix(X, y):
    return xgb.DMatrix(X, label = y)

def get_dmatrices(X_train, X_validate, X_test, y_train, y_validate, y_test):
    dtrain, dvalidate, dtest = get_dmatrix(X_train, y_train), \
                               get_dmatrix(X_validate, y_validate), \
                               get_dmatrix(X_test, y_test)
    return dtrain, dvalidate, dtest

def fit_xgb(params, early_stopping_rounds, dtrain, dvalidate):
    evallist = [(dtrain, 'train'), (dvalidate, 'eval')]
    fitted = xgb.train(params = params, 
                       dtrain = dtrain, 
                       early_stopping_rounds = early_stopping_rounds, 
                       evals = evallist)
    return fitted

def grid_search_xgb(X_train, X_validate, X_test, y_train, y_validate, y_test, gs_params_d):
    results_d = {}
    dtrain, dvalidate, dtest = get_dmatrices(X_train, X_validate, X_test, y_train, y_validate, y_test)
    for params in list(product(*gs_params_d.values())):
        param_d = {
            'seed': 0, 
            'base_score': 0
        }
        for param in zip((gs_params_d.keys()), params):
            param_d[param[0]] = param[1]
        early_stopping_rounds = 10
        model = fit_xgb(param_d, early_stopping_rounds, dtrain, dvalidate)
        score = get_rmsse(pd.concat([y_train, y_validate]), \
                             y_test, \
                             model.predict(dtest, ntree_limit = model.best_ntree_limit))
        results_d[params] = XGBResult(param_d, model, score)
    return results_d

class XGBResult(object):
    def __init__(self, params, model, score):
        self.params = params
        self.model = model
        self.score = score

def get_weights(total_sales):
    w = total_sales.groupby(level = 'id').apply(lambda x: x[-28:].sum())
    w /= sum(w)
    w.fillna(0, inplace = True)
    w.name = 'weight'
    return w

def get_weights_by_date(y, total_sales):
    w = get_weights(total_sales)
    w_by_d = y.reset_index().merge(w.reset_index())
    w_by_d.drop(['units_sold'], axis = 1, inplace = True)
    w_by_d.set_index(['id', 'date'], inplace = True)
    return w_by_d['weight']

def get_msd(y): # mean squared difference
    y = (y.groupby(level = 'id').diff(periods = 1)**2).dropna()\
          .groupby(level = 'id').mean()
    y.name = 'msd'
    return y

def get_msd_by_date(y): # , msd
    msd = get_msd(y)
    msd_by_d = y.reset_index().merge(msd.reset_index())
    msd_by_d.drop(['units_sold'], axis = 1, inplace = True)
    msd_by_d.set_index(['id', 'date'], inplace = True)
    return msd_by_d['msd']

def get_obj_fn_weights(total_sales, y):
    w = get_weights(total_sales)
    msd = get_msd(y)
    obj_fn_w = w.reset_index().merge(msd.reset_index())
    obj_fn_w['weight'] = obj_fn_w['weight'] / (obj_fn_w['msd']**0.5)
    obj_fn_w.drop(['msd'], axis = 1, inplace = True)
    obj_fn_w.set_index(['id'], inplace = True)
    return obj_fn_w['weight']

def get_obj_fn_weights_by_date(y, total_sales):
    w = get_weights(total_sales)
    msd = get_msd(y)
    obj_fn_w = w.reset_index().merge(msd.reset_index())
    obj_fn_w['weight'] = obj_fn_w['weight'] / (obj_fn_w['msd']**0.5)
    obj_fn_w.drop(['msd'], axis = 1, inplace = True)
    obj_fn_w.set_index(['id'], inplace = True)
    obj_fn_w.fillna(0, inplace = True)
    
    obj_fn_w_y_d = y.reset_index().merge(obj_fn_w.reset_index())
    obj_fn_w_y_d.drop(['units_sold'], axis = 1, inplace = True)
    obj_fn_w_y_d.set_index(['id', 'date'], inplace = True)
    return obj_fn_w_y_d['weight']

def get_wrmsse(total_sales, train, test, predict):
    weights = get_weights(total_sales).values.flatten()
    
    wrmsse_denominator = train.groupby(level = 'id')\
                              .apply(lambda x: np.mean((x.shift(-1) - x)**2))\
                              .values.flatten()
    
    test_predict = pd.concat([pd.DataFrame(test), 
                              pd.DataFrame(predict, index = test.index)], 
                             axis = 1)
    test_predict.columns = ['actual', 'predict']
    test_predict['diff'] = (test_predict['predict'] - test_predict['actual'])**2
    wrmsse_numerator = test_predict.groupby(level = 'id')['diff'].mean().values.flatten()
    
    wrmsse = np.sum(weights * (wrmsse_numerator / wrmsse_denominator)**0.5)
    return wrmsse
