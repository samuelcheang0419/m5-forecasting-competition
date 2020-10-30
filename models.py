import numpy as np
import pandas as pd
import xgboost as xgb
from itertools import product
from utilities import get_rmsse

### XGBOOST ###
# def get_dmatrix(X, y):
#     return xgb.DMatrix(X, label = y)

# def get_dmatrices(X_train, X_validate, X_test, y_train, y_validate, y_test):
#     dtrain, dvalidate, dtest = get_dmatrix(X_train, y_train), \
#                                get_dmatrix(X_validate, y_validate), \
#                                get_dmatrix(X_test, y_test)
#     return dtrain, dvalidate, dtest

# def fit_xgb(params, early_stopping_rounds, dtrain, dvalidate):
#     evallist = [(dtrain, 'train'), (dvalidate, 'eval')]
#     fitted = xgb.train(params = params, 
#                        dtrain = dtrain, 
#                        early_stopping_rounds = early_stopping_rounds, 
#                        evals = evallist)
#     return fitted

# def grid_search_xgb(X_train, X_validate, X_test, y_train, y_validate, y_test, gs_params_d):
#     results_d = {}
#     dtrain, dvalidate, dtest = get_dmatrices(X_train, X_validate, X_test, y_train, y_validate, y_test)
#     for params in list(product(*gs_params_d.values())):
#         param_d = {
#             'seed': 0, 
#             'base_score': 0
#         }
#         for param in zip((gs_params_d.keys()), params):
#             param_d[param[0]] = param[1]
#         early_stopping_rounds = 10
#         model = fit_xgb(param_d, early_stopping_rounds, dtrain, dvalidate)
#         score = get_rmsse(pd.concat([y_train, y_validate]), \
#                              y_test, \
#                              model.predict(dtest, ntree_limit = model.best_ntree_limit))
#         results_d[params] = XGBResult(param_d, model, score)
#     return results_d

class XGBResult(object):
    def __init__(self, params = None, model = None, score = None):
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

def get_obj_fn_weights(y, total_sales):
    w = get_weights(total_sales)
    msd = get_msd(y)
    obj_fn_w = w.reset_index().merge(msd.reset_index())
    obj_fn_w['weight'] = obj_fn_w['weight'] / (obj_fn_w['msd']**0.5)
    obj_fn_w.drop(['msd'], axis = 1, inplace = True)
    obj_fn_w.set_index(['id'], inplace = True)
    obj_fn_w.fillna(0, inplace = True)
    return obj_fn_w['weight']

def get_obj_fn_weights_by_date(y, total_sales):
    w = get_weights(total_sales)
    msd = get_msd(y)
    obj_fn_w = w.reset_index().merge(msd.reset_index())
    obj_fn_w['weight'] = obj_fn_w['weight'] / (obj_fn_w['msd']**0.5)
    obj_fn_w.drop(['msd'], axis = 1, inplace = True)
    obj_fn_w.set_index(['id'], inplace = True)
    obj_fn_w.fillna(0, inplace = True)
    
    # y only has items in train set, obj_fn_w has all items - even those missing in train set
    # pd.merge default is inner join, so result is only items in training set
    obj_fn_w_y_d = y.reset_index().merge(obj_fn_w.reset_index())
    obj_fn_w_y_d.drop(['units_sold'], axis = 1, inplace = True)
    obj_fn_w_y_d.set_index(['id', 'date'], inplace = True)
    return obj_fn_w_y_d['weight']

def get_wrmsse(total_sales, train, test, predict):
    weights = get_weights(total_sales)
    wrmsse_denominator = train.groupby(level = 'id', observed = True)\
                              .apply(lambda x: np.mean((x.shift(-1) - x)**2))
    test_predict = pd.concat([pd.DataFrame(test), 
                              pd.DataFrame(predict, index = test.index)], 
                             axis = 1)
    test_predict.columns = ['actual', 'predict']
    test_predict['diff'] = (test_predict['predict'] - test_predict['actual'])**2
    wrmsse_numerator = test_predict.groupby(level = 'id', observed = True)['diff'].mean()
    wrmsse = pd.concat([pd.DataFrame(wrmsse_numerator), pd.DataFrame(wrmsse_denominator)], axis = 1)
    wrmsse = pd.concat([wrmsse, pd.DataFrame(weights)], axis = 1, join = 'inner')
    wrmsse['wrmsse'] = wrmsse['weight'] * (wrmsse['diff'] / wrmsse['units_sold'])**0.5
    return wrmsse['wrmsse'].sum()

### GRIDSEARCH CLASS ###
def fit_model(dataset, xgb_params_d, eval_metric_l = ['rmse', 'mae'], early_stopping_rounds = 10):
    model = xgb.XGBRegressor(**xgb_params_d)
    evals = [
        (np.array(dataset.X_train), np.array(dataset.y_train)[:, np.newaxis]), 
        (np.array(dataset.X_validate), np.array(dataset.y_validate)[:, np.newaxis])
    ]
    eval_set_sample_weight_l = [np.array(dataset.w_train), np.array(dataset.w_validate)]
    sample_weight = np.array(dataset.w_train)
    xgb_fit_params = {
        'verbose': True, 
        'early_stopping_rounds': early_stopping_rounds, 
        'eval_set': evals, 
        'sample_weight_eval_set': eval_set_sample_weight_l, 
        'sample_weight': sample_weight, 
        'eval_metric': eval_metric_l
    }
    model.fit(np.array(dataset.X_train), np.array(dataset.y_train)[:, np.newaxis], **xgb_fit_params)
    best_ntree = model.best_ntree_limit
    model.set_params(
        n_estimators = best_ntree
    )
    xgb_fit_params = {
        'sample_weight': np.array(dataset.w_train_validate)
    }
    model.fit(np.array(dataset.X_train_validate), np.array(dataset.y_train_validate)[:, np.newaxis], **xgb_fit_params)
    return model

def fit_model_once_and_keep(dataset, xgb_params_d, eval_metric_l = ['rmse', 'mae'], early_stopping_rounds = 10):
    model = xgb.XGBRegressor(**xgb_params_d)
    evals = [
        (np.array(dataset.X_train), np.array(dataset.y_train)[:, np.newaxis]), 
        (np.array(dataset.X_validate), np.array(dataset.y_validate)[:, np.newaxis])
    ]
    eval_set_sample_weight_l = [np.array(dataset.w_train), np.array(dataset.w_validate)]
    sample_weight = np.array(dataset.w_train)
    xgb_fit_params = {
        'verbose': True, 
        'early_stopping_rounds': early_stopping_rounds, 
        'eval_set': evals, 
        'sample_weight_eval_set': eval_set_sample_weight_l, 
        'sample_weight': sample_weight, 
        'eval_metric': eval_metric_l
    }
    model.fit(np.array(dataset.X_train), np.array(dataset.y_train)[:, np.newaxis], **xgb_fit_params)
    return model

class Gridsearch():
    def __init__(self):
        self.results = {}
        self.dataset_l = None
        self.gs_params_d = None

    def fit_models(self, dataset_l, gs_params_d, fit_model_fn, **kwargs):
        self.dataset_l = dataset_l
        self.gs_params_d = gs_params_d

        for param in product(*self.gs_params_d.values()):
        
            score_l = []
            param_k = tuple(param)
            self.results[param_k] = {}
            xgb_params_d = {}
            xgb_params_d.update({k: v for k, v in zip(gs_params_d.keys(), param)})

            for dataset in dataset_l:
                model = fit_model_fn(dataset, xgb_params_d, **kwargs)

                score = get_wrmsse(
                    dataset.total_sales_train_validate, 
                    dataset.y_train_validate, dataset.y_test, 
                    model.predict(np.array(dataset.X_test))
                    )
                
                xgb_result = XGBResult(
                    model = model, 
                    score = score
                )
                
                self.results[param_k][dataset.st_exclude_last_n_days] = xgb_result
                score_l.append(score)
                
            self.results[param_k]['mean_score'] = np.mean(np.array(score_l))
            self.results[param_k]['std_score'] = np.std(np.array(score_l))
