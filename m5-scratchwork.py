import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic, kpss
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
import functools
from itertools import product

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

def timer(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        fn_return = fn(*args, **kwargs)
        end = datetime.datetime.now()
        print('Function took {} time'.format(end - start))
        return fn_return
    return wrapper

def get_single_item(sp_start_i, sp_end_i, stv_i):
    calendar = pd.read_csv('calendar.csv')
    sp = pd.read_csv('sell_prices.csv', header = 0, skiprows = range(1, sp_start_i + 1), nrows = sp_end_i - sp_start_i + 1) # + 1 because sp_end_i is inclusive
    stv = pd.read_csv('sales_train_validation.csv', header = 0, skiprows = range(1, stv_i + 1), nrows = 1)
    return build_current_item_df(sp, stv, calendar)

def get_all_item_sp(i, current_item, sp_file):
    item_l = []
    if current_item == None:
        current_store_id, current_item_id = None, None
    else:
        current_store_id, current_item_id = current_item[0], current_item[1]
        item_l.append(current_item)
    while True:
        i += 1
        new_item = sp_file.readline().strip().split(',')
        # if new_item is just [''], that means reached end of file object
        if new_item[0] == '':
            break
        new_store_id, new_item_id = new_item[0], new_item[1]
        if current_item_id == None or (current_item_id == new_item_id and current_store_id == new_store_id): # initialize, or still getting all rows for item-store
            # add new item to item_l
            item_l.append(new_item)
            current_store_id, current_item_id = new_store_id, new_item_id
        else: # encountered new item in sell_prices, so get corresponding sales_train_validation rows
            break
    return i, new_item, item_l

def build_current_item_df(sp, stv, calendar):
    if type(stv) == pd.core.series.Series:
        stv = pd.DataFrame(stv.iloc[6:])
    elif type(stv) == pd.DataFrame:
        stv = stv.iloc[[0], 6:].T
    else: 
        raise TypeError('stv is type {}'.format(type(stv)))
    stv.columns = ['units_sold']
    stv['units_sold'] = pd.to_numeric(stv['units_sold'])
    
    sp['wm_yr_wk'] = pd.to_numeric(sp['wm_yr_wk'])
    sp['sell_price'] = pd.to_numeric(sp['sell_price'])
    current_item_df = pd.merge(stv, calendar.set_index('d'), how = 'outer', left_index = True, right_index = True)
    current_item_df = pd.merge(current_item_df, sp, how = 'outer', on = 'wm_yr_wk')
    current_item_df['units_sold'] = current_item_df['units_sold'].fillna(0)
    current_item_df['total_sale'] = current_item_df['units_sold'] * current_item_df['sell_price']
    current_item_df['item_available'] = ~current_item_df['sell_price'].isnull()
    current_state = sp.loc[0, 'store_id'][:2]
    current_item_df['SNAP_allowed'] = (current_item_df['snap_' + current_state]).astype(bool)
    current_item_df['date'] = pd.to_datetime(current_item_df['date'])
    current_item_df['days_in_month'] = current_item_df['date'].dt.daysinmonth
    current_item_df['days_in_year'] = current_item_df['year'] % 4 == 0
    current_item_df['days_in_year'] = current_item_df['days_in_year'].replace({True: 366, False: 365})
    store_id, item_id = sp.loc[0, ['store_id', 'item_id']]
    current_item_df['store_id'].fillna(store_id, inplace = True)
    current_item_df['item_id'].fillna(item_id, inplace = True)
    current_item_df.sort_values(['item_id', 'date'], inplace = True)
    current_item_df.reset_index(drop = True, inplace = True)
    return current_item_df

def get_total_sale_max_row(df):
    total_sale_max_row = df.loc[df['total_sale'].idxmax(), ['date', 'units_sold', 'sell_price', 'total_sale']]
    total_sale_max_row.index = 'max_total_sale_' + total_sale_max_row.index
    return total_sale_max_row

def get_units_sold_max_row(df):
    units_sold_max_row = df.loc[df['units_sold'].idxmax(), ['date', 'units_sold', 'sell_price', 'total_sale']]
    units_sold_max_row.index = 'max_units_sold_' + units_sold_max_row.index
    return units_sold_max_row

def get_availability_s(df):
    item_available_min_date, item_available_max_date = df[df['item_available'] == True]['date'].min(), df[df['item_available'] == True]['date'].max()
    item_unavailable_dates = df.loc[~df['item_available'], 'date']
    item_unavailable_dates = item_unavailable_dates[(item_unavailable_dates >= item_available_min_date) & (item_unavailable_dates <= item_available_max_date)].values
    item_available_day_cnt, item_sold_day_cnt = df[df['item_available'] == True].shape[0], df[df['units_sold'] > 0].shape[0]
    availability_index = ['item_available_min_date', 'item_available_max_date', 'item_unavailable_dates', 'item_available_day_cnt', 'item_sold_day_cnt']
    availability_s = pd.Series([item_available_min_date, item_available_max_date, item_unavailable_dates, item_available_day_cnt, item_sold_day_cnt], index = availability_index)
    return availability_s

def get_grouped_s(df, key):
    sum_columns = ['units_sold', 'total_sale', 'sell_price']
    grouped = df[df['item_available'] == True]
    if 'event_name_1' in key:
        grouped = grouped[grouped['event_name_1'].notnull()]
        grouped['event_name_1'] = grouped.apply(lambda row: row['event_name_1'] if pd.isnull(row['event_name_2']) else row['event_name_1'] + ' & {}'.format(row['event_name_2']), axis = 1)
        grouped['event_type_1'] = grouped.apply(lambda row: row['event_type_1'] if pd.isnull(row['event_name_2']) else row['event_type_1'] + ' & {}'.format(row['event_type_2']), axis = 1)
        if key == 'event_name_1_week_prior':
            for c in sum_columns:
                grouped[c] = grouped[c].rolling(7).sum() / 7
    if key == 'event_name_1_week_prior':
        grouped = grouped.groupby('event_name_1')
    elif key != 'all_time':
        grouped = grouped.groupby(key)
    else:
        grouped = grouped.groupby(lambda x: 'all_time')
    grouped = grouped.agg(['mean', 'std', 'count'])[sum_columns]
    grouped.index = ['all_time'] if key == 'all_time' else grouped.index.astype(str) if key == 'event_name_1' else grouped.index.astype(str) + '_week_prior' if key == 'event_name_1_week_prior' else key + '_' + grouped.index.astype(str)
    idx = pd.IndexSlice
    count_df = grouped.loc[:, idx[:, 'count']]
    count_df = count_df.loc[:, idx['units_sold', 'count']]
    count_df.index = count_df.index + '_day_count'

    grouped = grouped.loc[:, idx[:, ['mean', 'std']]]
    grouped.columns = map(lambda x: x[1] + '_' + x[0] + '_per_day', zip(grouped.columns.get_level_values(0), grouped.columns.get_level_values(1)))
    grouped = grouped.unstack()
    grouped.index = map(lambda x: x[1] + '_' + x[0], zip(grouped.index.get_level_values(0), grouped.index.get_level_values(1)))
    return pd.concat([grouped, count_df])

def get_grouped_special_event_s(df, key, all_time_d):
    sum_columns = ['units_sold', 'total_sale', 'sell_price']
    grouped = df[df['item_available'] == True]
    if 'event_name_1' in key:
        grouped = grouped[grouped['event_name_1'].notnull()]
        grouped['event_name_1'] = grouped.apply(lambda row: row['event_name_1'] if pd.isnull(row['event_name_2']) else row['event_name_1'] + ' & {}'.format(row['event_name_2']), axis = 1)
        grouped['event_type_1'] = grouped.apply(lambda row: row['event_type_1'] if pd.isnull(row['event_name_2']) else row['event_type_1'] + ' & {}'.format(row['event_type_2']), axis = 1)
        if key == 'event_name_1_week_prior':
            for c in sum_columns:
                grouped[c] = grouped[c].rolling(7).sum() / 7
        grouped = grouped.groupby('event_name_1')
#     if key == 'event_name_1_week_prior':
#         grouped = grouped.groupby('event_name_1')
#     elif key != 'all_time':
#         grouped = grouped.groupby(key)
#     else:
#         grouped = grouped.groupby(lambda x: 'all_time')
    grouped = grouped.agg(['mean', 'std', 'count'])[sum_columns]
    
    
    grouped.index = ['all_time'] if key == 'all_time' else grouped.index.astype(str) if key == 'event_name_1' else grouped.index.astype(str) + '_week_prior' if key == 'event_name_1_week_prior' else key + '_' + grouped.index.astype(str)
    idx = pd.IndexSlice
    count_df = grouped.loc[:, idx[:, 'count']]
    count_df = count_df.loc[:, idx['units_sold', 'count']]
    count_df.index = count_df.index + '_day_count'

    grouped = grouped.loc[:, idx[:, ['mean', 'std']]]
    grouped.columns = map(lambda x: x[1] + '_' + x[0] + '_per_day', zip(grouped.columns.get_level_values(0), grouped.columns.get_level_values(1)))
    grouped = grouped.unstack()
    grouped.index = map(lambda x: x[1] + '_' + x[0], zip(grouped.index.get_level_values(0), grouped.index.get_level_values(1)))
    return pd.concat([grouped, count_df])

def build_summary_series(current_item_df):
    return_l = []
    return_l.append(get_total_sale_max_row(current_item_df))
    return_l.append(get_units_sold_max_row(current_item_df))
    return_l.append(get_availability_s(current_item_df))
    for key in ['wday', 'month', 'year', 'SNAP_allowed', 'all_time']:
        return_l.append(get_grouped_s(current_item_df, key))
#     all_time_sum_d = {}
#     print(return_l[-1])
#     for key in ['event_name_1', 'event_name_1_week_prior']:
#         return_l.append(get_grouped_special_event_s(current_item_df, key))
    return return_l

@timer
def get_all_item_s_list(num_items = 1):
    calendar_df = pd.read_csv('calendar.csv')    
    with open('sell_prices.csv', 'r') as sp, open('sales_train_validation.csv', 'r') as stv:
        sp_cols = sp.readline().strip().split(',')
        stv_cols = stv.readline().strip().split(',')
        all_item_s_list = []
        i_cnt = 0
        sp_current_i, sp_current_item = -1, None
        stv_current_i = 0
        
        while True:
            # get sp
            sp_next_i, sp_next_item, sp_current_item_df = get_all_item_sp(sp_current_i, sp_current_item, sp)
            sp_current_item_df = pd.DataFrame(sp_current_item_df, columns = sp_cols)
#             sp_current_item_df['wm_yr_wk'] = pd.to_numeric(sp_current_item_df['wm_yr_wk'])
#             sp_current_item_df['sell_price'] = pd.to_numeric(sp_current_item_df['sell_price'])
            
            # get stv
            stv_current_item = stv.readline().strip().split(',')
            if stv_current_item[4] != sp_current_item_df.loc[0, 'store_id'] or stv_current_item[1] != sp_current_item_df.loc[0, 'item_id']:
                print('Weird, STV and SP do not match - STV item_id: {}, SP item_id: {}\n\
                                                        STV store_id, {}, SP store_id: {}'.format(stv_current_item[1], sp_current_item_df.loc[0, 'item_id'], \
                                                                                                  stv_current_item[4], sp_current_item_df.loc[0, 'store_id']))
                return
            stv_current_item_df = pd.Series(stv_current_item, index = stv_cols)
            
            # build item df
            current_item_df = build_current_item_df(sp_current_item_df, stv_current_item_df, calendar_df)
            
            # compute summary
            series_list = []
            # '0 if ..' is to account for first item in sp table
            item_meta_s = pd.Series(stv_current_item[:6] + [0 if sp_current_i == -1 else sp_current_i, sp_next_i - 1, stv_current_i], index = stv_cols[:6] + ['sp_start_i', 'sp_end_i', 'stv_i'])
            series_list.append(item_meta_s)
            series_list.extend(build_summary_series(current_item_df))
            item_sum_series = pd.concat(series_list) # final product!
            all_item_s_list.append(pd.DataFrame(item_sum_series).T)
            
            # evaluate break conditions 
            # when no more items left
            if sp_next_item[0] == '': # no more items in sp
                break
            # when reached num_items
            i_cnt += 1
            if i_cnt == num_items:
                return all_item_s_list
            
            # then reset and continue with new item
            sp_current_i, sp_current_item = sp_next_i, sp_next_item
            stv_current_i += 1
        
        return all_item_s_list

def add_date_grid(ax, major, minor = None, major_fmt = None, minor_fmt = None):
    # https://matplotlib.org/3.3.0/gallery/text_labels_and_annotations/date.html
    years = mdates.YearLocator()
    years_fmt = mdates.DateFormatter('%Y-%m') # strftime() and strptime() Behavior
    quarters = mdates.MonthLocator(range(1, 13, 3))
    quarters_fmt = mdates.DateFormatter('%Y-%m')
    months =  mdates.MonthLocator()
    months_fmt = mdates.DateFormatter('%Y-%m')
    weeks = mdates.WeekdayLocator(byweekday = 0) # 0 == Monday https://dateutil.readthedocs.io/en/stable/rrule.html#module-dateutil.rrule
    weeks_fmt = mdates.DateFormatter('%Y-%m-%d %a')
    days = mdates.DayLocator()
    days_fmt = mdates.DateFormatter('%Y-%m-%d')
    auto = mdates.AutoDateLocator()
    auto_fmt = mdates.AutoDateFormatter(auto)
    
    locator_d = {
        'year': years, 
        'quarter': quarters, 
        'month': months, 
        'week': weeks, 
        'day': days, 
        'auto': auto
    }
    
    formatter_d = {
        'year': years_fmt, 
        'quarter': quarters_fmt, 
        'month': months_fmt, 
        'week': weeks_fmt, 
        'day': days_fmt, 
        'auto': auto_fmt
    }
    
    ax.xaxis.set_major_locator(locator_d[major])
    if major_fmt:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(major_fmt))
    else:
        ax.xaxis.set_major_formatter(formatter_d[major])
    if minor:
        ax.xaxis.set_minor_locator(locator_d[minor])
        if minor_fmt:
            ax.xaxis.set_minor_formatter(mdates.DateFormatter(minor_fmt))
        else:
            ax.xaxis.set_minor_formatter(formatter_d[minor])
    ax.grid(True, which = 'both')
    return ax

def print_adfuller_result(s, **kwargs):
    adfuller_result = adfuller(s, **kwargs)
    print('ADF test statistic: {}\n\
           P-value: {}\n\
           # lags used: {}\n\
           # observations used: {}\n\
           Critical values: {}\n\
           Maximized information criterion: {}'.format(*adfuller_result))
    return adfuller_result

def print_kpss_result(s, **kwargs):
    kpss_result = kpss(s, **kwargs)
    print('KPSS test statistic: {}\n\
           P-value: {}\n\
           # lags used: {}\n\
           Critical values: {}'.format(*kpss_result))
    return kpss_result

# ############################ FORWARD VALIDATION ############################ 
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

def walk_forward_validation_single_run(model, train, test):
    fitted = SARIMAX(train.values, order = (model.ar, model.d, model.ma), seasonal_order = (model.s_ar, model.s_d, model.s_ma, model.s_period), trend = 'c').fit()
    predicted_vals = fitted.predict(model.d, train.shape[0] - model.d + test.shape[0] - 1) # typ arg only exists for ARIMA, not SARIMAX model
    rmsse = get_rmsse(train, test, predicted_vals[-test.shape[0]:])
    return rmsse
    
def run_walk_forward_validation(model, y, test_split_iterator):
    iteration_results_d = {} # contains walk forward validation parameters and RMSSE, for each validation run
    # for each walk forward validation:
    for test_split in test_split_iterator:
        # run validation and record result
        train, test = time_series_train_test_split(y, test_split)
        rmsse = walk_forward_validation_single_run(model, train, test)
        iteration_results_d[test_split] = rmsse
    mean_rmsse = np.array(list(iteration_results_d.values())).mean()
    return mean_rmsse

def get_arima_model(model_params):
#     arima_sarima, ar, d, ma, s_ar, s_d, s_ma, s_period = model_params
    model = ARIMABaseModel(*model_params)
    return model
        
class ARIMABaseModel(object):
    def __init__(self, arima_sarima, ar, d, ma, s_ar, s_d, s_ma, s_period):
        if arima_sarima not in ('arima', 'sarima'):
            raise ValueError('arima_sarima can only be either (arima, sarima)')
        self.arima_sarima = arima_sarima
        self.ar = ar
        self.d = d
        self.ma = ma
        if arima_sarima == 'arima':
            self.s_ar = 0
            self.s_d = 0
            self.s_ma = 0
            self.s_period = 0
        else:
            self.s_ar = s_ar
            self.s_d = s_d
            self.s_ma = s_ma
            self.s_period = s_period
    
    def __str__(self):
        arima_sarima_repr = 'ARIMA' if self.arima_sarima == 'arima' else 'SARIMA'
        return '{}({},{},{}) x ({},{},{})_{}'.format(arima_sarima_repr, self.ar, self.d, self.ma, self.s_ar, self.s_d, self.s_ma, self.s_period)
        
def time_series_grid_search(y, test_split_iterator = None, specific_models = None, \
                            arima_sarima_l = None, ar_l = None, d_l = None, ma_l = None, \
                            s_ar_l = None, s_d_l = None, s_ma_l = None, s_period_l = None):
    # specific_models = [ARIMABaseModel]
    if specific_models == None:
        if arima_sarima_l == None or ar_l == None or d_l == None or ma_l == None or s_ar_l == None or s_d_l == None or s_ma_l == None or s_period_l == None:
            raise ValueError('If specific_models is None, then other arguments must be non-null')
        models = product(*[arima_sarima_l, ar_l, d_l, ma_l, s_ar_l, s_d_l, s_ma_l, s_period_l])
        models = map(lambda x: get_arima_model(x), models)
    else:
        if arima_sarima_l != None or ar_l != None or d_l != None or ma_l != None or s_ar_l != None or s_d_l != None or s_ma_l != None or s_period_l != None:
            raise ValueError('If specific_models is not null, then other arguments must be null')
        models = specific_models
    # things to record: model name, model object, y, walk forward validation parameters, RMSSE for each validation run
    results_d = {}
    for model in models:
#         print('Fitting model {}'.format(str(model)))
#         try: 
#             # run model
#             model_result = run_walk_forward_validation(model, y, test_split_iterator)
#         except: 
#             model_result = None
        model_result = run_walk_forward_validation(model, y, test_split_iterator)
        model_result_d = {
#             'store_id': _, 
#             'item_id': _, 
#             'data': y, 
            'model': model, 
            'result': model_result
        }
        results_d[str(model)] = model_result_d
    return results_d
