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
