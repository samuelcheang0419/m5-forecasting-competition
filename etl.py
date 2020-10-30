import numpy as np
import pandas as pd
from utilities import timer, get_slope, get_train_validate_test_indices
from sklearn.base import BaseEstimator, TransformerMixin
import datetime

special_events_d = {
    'Ramadan starts': ['2011-08-01', '2012-07-20', '2013-07-09', '2014-06-29', '2015-06-18', '2016-06-07', '2017-05-27', '2018-05-16'], 
    'Eid al-Fitr': ['2011-08-31', '2012-08-19', '2013-08-08', '2014-07-29', '2015-07-18', '2016-07-06', '2017-06-24', '2018-06-14'], 
    'LaborDay': ['2011-09-05', '2012-09-03', '2013-09-02', '2014-09-01', '2015-09-07', '2016-09-05', '2017-09-04', '2018-09-03'], 
    'ColumbusDay': ['2011-10-10', '2012-10-08', '2013-10-14', '2014-10-13', '2015-10-12', '2016-10-10', '2017-10-09', '2018-10-08'], 
    'EidAlAdha': ['2011-11-07', '2012-10-26', '2013-10-15', '2014-10-04', '2015-09-24', '2016-09-13', '2017-09-02', '2018-08-22'], 
    'Halloween': ['2011-10-31', '2012-10-31', '2013-10-31', '2014-10-31', '2015-10-31', '2016-10-31', '2017-10-31', '2018-10-31'], 
    'VeteransDay': ['2011-11-11', '2012-11-11', '2013-11-11', '2014-11-11', '2015-11-11', '2016-11-11', '2017-11-11', '2018-11-11'], 
    'Thanksgiving': ['2011-11-24', '2012-11-22', '2013-11-28', '2014-11-27', '2015-11-26', '2016-11-24', '2017-11-23', '2018-11-22'], 
    'Chanukah End': ['2011-12-28', '2012-12-16', '2013-12-05', '2014-12-24', '2015-12-14', '2017-01-01', '2017-12-20', '2018-12-10'], 
    'Christmas': ['2011-12-25', '2012-12-25', '2013-12-25', '2014-12-25', '2015-12-25', '2016-12-25', '2017-12-25', '2018-12-25'], 
    'NewYear': ['2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01'], 
    'OrthodoxChristmas': ['2012-01-07', '2013-01-07', '2014-01-07', '2015-01-07', '2016-01-07', '2017-01-07', '2018-01-07', '2019-01-07'], 
    'MartinLutherKingDay': ['2012-01-16', '2013-01-21', '2014-01-20', '2015-01-19', '2016-01-18', '2017-01-16', '2018-01-15'], 
    'SuperBowl': ['2011-02-06', '2012-02-05', '2013-02-03', '2014-02-02', '2015-02-01', '2016-02-07', '2017-02-05', '2018-02-04'], 
    'ValentinesDay': ['2011-02-14', '2012-02-14', '2013-02-14', '2014-02-14', '2015-02-14', '2016-02-14', '2017-02-14', '2018-02-14'], 
    'PresidentsDay': ['2011-02-21', '2012-02-20', '2013-02-18', '2014-02-17', '2015-02-16', '2016-02-15', '2017-02-20', '2018-02-19'], 
    'LentStart': ['2011-03-09', '2012-02-22', '2013-02-13', '2014-03-05', '2015-02-18', '2016-02-10', '2017-03-01', '2018-02-14'], 
    'Purim End': ['2011-03-20', '2012-03-08', '2013-02-24', '2014-03-16', '2015-03-05', '2016-03-24', '2017-03-12', '2018-03-01'], 
    'StPatricksDay': ['2011-03-17', '2012-03-17', '2013-03-17', '2014-03-17', '2015-03-17', '2016-03-17', '2017-03-17', '2018-03-17'], 
    'Easter': ['2011-04-24', '2012-04-08', '2013-03-31', '2014-04-20', '2015-04-05', '2016-03-27', '2017-04-16', '2018-04-01'], 
    'Pesach End': ['2011-04-26', '2012-04-14', '2013-04-02', '2014-04-22', '2015-04-11', '2016-04-30', '2017-04-18', '2018-04-07'], 
    'Cinco De Mayo': ['2011-05-05', '2012-05-05', '2013-05-05', '2014-05-05', '2015-05-05', '2016-05-05', '2016-05-05', '2017-05-05', '2018-05-05'], 
    "Mother's day": ['2011-05-08', '2012-05-13', '2013-05-12', '2014-05-11', '2015-05-10', '2016-05-08', '2017-05-14', '2018-05-13'], 
    'MemorialDay': ['2011-05-30', '2012-05-28', '2013-05-27', '2014-05-26', '2015-05-25', '2016-05-30', '2017-05-29', '2018-05-28'], 
    'NBAFinalsStart': ['2011-05-31', '2012-06-12', '2013-06-06', '2014-06-05', '2015-06-04', '2016-06-02', '2017-06-01', '2018-05-31'], 
    'IndependenceDay': ['2011-07-04', '2012-07-04', '2013-07-04', '2014-07-04', '2015-07-04', '2016-07-04', '2017-07-04', '2018-07-04'], 
    'OrthodoxEaster': ['2011-04-24', '2012-04-15', '2013-05-05', '2014-04-20', '2015-04-12', '2016-05-01', '2017-04-16', '2018-04-08'], 
    "Father's day": ['2011-06-19', '2012-06-17', '2013-06-16', '2014-06-15', '2015-06-21', '2016-06-19', '2017-06-18', '2018-06-17']
}

### INDEX TABLES ###
def get_all_item_indices_df():
    sp = get_all_sp_indices()
    st = get_all_st_indices()
    sp_st = sp.merge(st, how = 'outer', left_index = True, right_index = True)
    return sp_st
    
def get_all_sp_indices(file_loc = 'sell_prices.csv'):
    sp_cols = ['store_id', 'item_id']
    sp = pd.read_csv(file_loc, header = 0, usecols = sp_cols)
    sp.reset_index(inplace = True)
    sp = sp.groupby(sp_cols)['index'].agg(['min', 'max'])
    sp.columns = 'sp_' + sp.columns + '_index'
    return sp

def get_all_st_indices(file_loc = 'sales_train_validation.csv'):
    sp_cols = ['store_id', 'item_id']
    st_cols = ['id', 'item_id', 'dept_id', 'cat_id','store_id', 'state_id']
    st = pd.read_csv(file_loc, header = 0, usecols = st_cols)
    st.reset_index(inplace = True)
    st = st.groupby(sp_cols)['index'].agg('min')
    st.rename('st_index', inplace = True)
    return st

### BUILD STARTING DF ###
def get_calendar_df():
    cal_dtype = {
        'date': 'str', 
        'wm_yr_wk': 'int16', 
        'weekday': 'category', 
        'wday': 'int16', 
        'month': 'int16', 
        'year': 'int16', 
        'd': 'str', 
        'event_name_1': 'category', 
        'event_type_1': 'category', 
        'event_name_2': 'category', 
        'event_type_2': 'category', 
        'snap_CA': 'int16', 
        'snap_TX': 'int16', 
        'snap_WI': 'int16'
    }
    cal_dates = ['date']
    calendar = pd.read_csv('calendar.csv', dtype = cal_dtype, parse_dates = cal_dates)
    return calendar

def get_sell_prices_df(from_l = None, from_csv_i = None):
    # from_l or from_csv_i must be not None
    if from_l == None and from_csv_i == None:
        raise ValueError('from_l or from_csv_i must be not None')
    
    sp_dtype = {
        'store_id': 'category', 
        'item_id': 'category', 
        'wm_yr_wk': 'int16', 
        'sell_price': 'float32'
    }
    sp_cols = ['store_id', 'item_id', 'wm_yr_wk', 'sell_price']
    file_loc = 'sell_prices.csv'
    
    if from_l == None:
        sp_start_i, sp_end_i = from_csv_i
        sp = pd.read_csv(file_loc, dtype = sp_dtype, header = 0, skiprows = range(1, sp_start_i + 1), nrows = sp_end_i - sp_start_i + 1) # + 1 because sp_end_i is inclusive
    else:
        sp = pd.DataFrame(from_l, columns = sp_cols, dtype = sp_dtype)
    return sp

def get_sales_train_validation_df(from_l = None, from_csv_i = None, first_day_int = 1, last_day_int = 1913): # sales_train_validation last day is d_1913
    # from_l or from_csv_i must be not None
    if from_l == None and from_csv_i == None:
        raise ValueError('from_l or from_csv_i must be not None')
    
    st_cat_cols = ['id', 'item_id', 'dept_id', 'cat_id','store_id', 'state_id']
    st_num_cols = list(map(lambda x: 'd_{}'.format(x), range(first_day_int, first_day_int + last_day_int)))
    st_dtype = {col: 'category' for col in st_cat_cols}
    st_dtype.update({col: 'int16' for col in st_num_cols})
    file_loc = 'sales_train_validation.csv'
    
    if from_l == None:
        st_start_i, st_end_i = from_csv_i
        st = pd.read_csv(file_loc, dtype = st_dtype, header = 0, skiprows = range(1, st_start_i + 1), nrows = st_end_i - st_start_i + 1)
        st = pd.melt(st, id_vars = st_cat_cols, value_vars = st_num_cols, var_name = 'd', value_name = 'units_sold')
    else:
        st = pd.DataFrame(from_l, columns = st_cat_cols + st_num_cols, dtype = st_dtype)
    return st

def build_starting_df_from_three_dfs(st, cal, sp):
    df = pd.merge(st, cal, on = 'd', copy = False)
    df = pd.merge(df, sp, on = ['store_id', 'item_id', 'wm_yr_wk'], copy = False)
    df.sort_values(['id', 'date'], inplace = True)
    df.reset_index(inplace = True, drop = True)
    # ideally would sort earlier, but merge has bugs
    # https://github.com/pandas-dev/pandas/pull/24521
    return df

@timer
def build_starting_df_specific_i(sp_start_i, sp_end_i, st_start_i, st_end_i, st_first_day_int = 1, st_last_day_int = 1913):
    calendar = get_calendar_df()
    sp = get_sell_prices_df(from_csv_i = (sp_start_i, sp_end_i))
    st = get_sales_train_validation_df(from_csv_i = (st_start_i, st_end_i), first_day_int = st_first_day_int, last_day_int = st_last_day_int)
    df = build_starting_df_from_three_dfs(st, calendar, sp)
    return df

### ADD FEATURES ###
@timer
def feature_SNAP_allowed(df):
    df['SNAP_allowed'] = pd.concat([((df['state_id'] == state_id) & (df['snap_{}'.format(state_id)])) for state_id in ['CA', 'TX', 'WI']], axis = 1).any(axis = 1)
    
@timer
def feature_special_events(df, events_d = special_events_d, variable_length_events_l = ['NBAFinalsEnd']):
    # def single_item_special_events(item_df):
    #     item_df_max_date = item_df['date'].max()

    #     for event in events_d.keys():
    #         if event in ['LentWeek2', 'NBAFinalsEnd']:
    #             continue
    #         item_df['is_event'] = ((item_df['event_name_1'] == event) | (item_df['event_name_2'] == event))
    #         item_df['date_only_event'] = np.where(item_df['is_event'] == True, item_df['date'].astype(int), np.nan) # np.where doesn't work well with date types
    #         item_df['nearest_future_event_date'] = item_df.loc[::-1, 'date_only_event'].expanding(1).min()[::-1]
    #         item_df['nearest_future_event_date'] = pd.to_datetime(item_df['nearest_future_event_date'], unit = 'ns')
    #         # get event dates beyond dataset
    #         nearest_future_event_date = list(filter(lambda x: pd.to_datetime(x) > item_df_max_date, events_d[event]))[0]
    #         null_future_event_date_mask = item_df['nearest_future_event_date'].isnull()
    #         item_df.loc[null_future_event_date_mask, 'nearest_future_event_date'] = pd.to_datetime(nearest_future_event_date)
    #         item_df['x_days_before_first_day_of_' + event.replace(' ', '').replace("'", '')] = (item_df['nearest_future_event_date'] - item_df['date']).dt.days
    #         # harder to do x_days_before_LAST_day as events like NBAFinalsEnd is unknown
    #         item_df.drop(['is_event', 'date_only_event', 'nearest_future_event_date'], axis = 1, inplace = True)

    #     for event in variable_length_events_l:
    #         item_df['is_' + event.replace(' ', '').replace("'", '')] = ((item_df['event_name_1'] == event) | (item_df['event_name_2'] == event))
            
    #     return item_df
            
    # special_event_df = df[['id', 'date', 'event_name_1', 'event_name_2']].groupby('id', observed = True).apply(single_item_special_events)
    # df[special_event_df.columns] = special_event_df
    for k, v in events_d.items():
        event_col_name = 'x_days_before_first_day_of_{}'.format(k.replace(' ', '').replace("'", ''))
        events_df = pd.DataFrame(data = map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'), v), columns = [event_col_name])
        df[event_col_name] = df.groupby(['id'], as_index = False)[['date']].apply(lambda x: pd.merge_asof(x, events_df, left_on = 'date', right_on = events_df.columns[0], direction = 'forward'))[event_col_name].reset_index(drop = True)
        df[event_col_name] = (df[event_col_name] - df['date']).dt.days

    for event in variable_length_events_l:
        df['is_' + event.replace(' ', '').replace("'", '')] = ((df['event_name_1'] == event) | (df['event_name_2'] == event))
    
@timer
def feature_total_sales(df):
    df['total_sales'] = df['units_sold'] * df['sell_price']
    
@timer
def feature_lag(df, d):
    # df['lag_' + str(d) + '_units_sold'] = df.groupby(['id'], observed = True)['units_sold'].shift(d)
    df['lag_' + str(d) + '_units_sold'] = df.groupby(['id'], observed = True).apply(lambda x: x.sort_values(['date'])['units_sold'].shift(d)).values
    
@timer
def feature_rolling_average_of_lag(df, rolling_d, lag_d):
    temp_col = 'TEMP_lag1'
    df[temp_col] = df.groupby(['id'], observed = True)['units_sold'].shift(lag_d)
    df['rolling_average_{}_of_lag_{}_units_sold'.format(rolling_d, lag_d)] = df.groupby(['id'], observed = True)[temp_col].rolling(rolling_d).mean().values
    df.drop([temp_col], axis = 1, inplace = True)
    
@timer
def feature_rolling_trend_of_lag(df, rolling_d, lag_d):
    temp_col = 'TEMP_lag1'
    df[temp_col] = df.groupby(['id'], observed = True)['units_sold'].shift(lag_d)
    df['rolling_trend_{}_of_lag_{}_units_sold'.format(rolling_d, lag_d)] = df.groupby(['id'], observed = True)[temp_col].rolling(rolling_d).apply(lambda x: get_slope(x)).values
    df.drop([temp_col], axis = 1, inplace = True)

### BUILD AGGREGATE TABLE ###
@timer
def build_agg_items_df(items_df):
    merge_lst = [get_overall_trend(items_df, 
                                   groupby_cols=['id', lambda x: 'all_time'], 
                                   agg_cols=['units_sold'])]
    builtin_aggs_col = ['wday', 'month', 'year', 'SNAP_allowed', lambda x: 'all_time']
    for col in builtin_aggs_col:
        merge_lst.append(get_builtin_aggs(items_df, 
                                         groupby_cols = ['id', col], 
                                         agg_cols = ['units_sold', 'sell_price'], 
                                         agg_funcs = ['mean', 'std']))
    return pd.concat(merge_lst, axis = 1)

def flatten_agg(df):
    for i, name in enumerate(df.index.names):
        if name != 'id' and name != None:
            df.index.set_levels(
                map(lambda x: '{}_{}'.format(name, x), df.index.levels[i]), 
                level = i, 
                inplace = True
            )
    df = df.unstack(level = list(map(lambda x: x[0], filter(lambda x: x[1] != 'id', enumerate(df.index.names))))) # level arg doesn't take iterators
    df.columns = map(lambda x: '_'.join(x), df.columns)
    return df
    
def get_builtin_aggs(items_df, groupby_cols, agg_cols, agg_funcs = ['mean', 'std', 'count']):
    return_df = items_df.groupby(groupby_cols)[agg_cols].agg(agg_funcs)
    return_df = flatten_agg(return_df)
    return return_df

def get_overall_trend(items_df, groupby_cols, agg_cols):
    return_df = items_df.groupby(groupby_cols)[agg_cols].apply(get_slope)
    return_df = pd.DataFrame(return_df, columns = pd.MultiIndex.from_product([agg_cols, ['trend']]))
    return_df = flatten_agg(return_df)
    return return_df

### DATASET OBJECT ###
class Dataset():
    def __init__(
        self, 
        st_exclude_last_n_days = None, 
        st_csv_i = None, 
        sp_csv_i = None, 
        initial_add_features_fn = None, 
        pipeline = None, 
        intermediate_from_file_loc = None, 
        final_from_file_loc = None, 
        intermediate_to_file_loc = None, 
        final_to_file_loc = None, 
        split_days = 28, 
        verbose = True
    ):
        self.st_exclude_last_n_days = st_exclude_last_n_days
        self.st_csv_i = st_csv_i
        self.sp_csv_i = sp_csv_i
        self.w_train = None
        self.w_validate = None
        self.w_test = None
        self.w_train_validate = None

        if verbose:
            print('[{}] Creating initial df..'.format(datetime.datetime.now()))
        if intermediate_from_file_loc:
            self.df = pd.read_pickle(intermediate_from_file_loc)
        else:
            self.df = self.build_starting_df_specific_i(st_exclude_last_n_days, st_csv_i, sp_csv_i)
        if verbose:
            print('[{}] Finish creating initial df..'.format(datetime.datetime.now()))
        
        if verbose:
            print('[{}] Adding initial features..'.format(datetime.datetime.now()))
        if initial_add_features_fn:
            initial_add_features_fn(self.df)
        if verbose:
            print('[{}] Finish adding initial features..'.format(datetime.datetime.now()))
        
        if intermediate_to_file_loc:
            self.df.to_pickle(intermediate_to_file_loc)
        
        if final_from_file_loc:
            final_from_file_name = final_from_file_loc.split('.')[0]
            self.X_train = pd.read_pickle('{}_X_train.pkl'.format(final_from_file_name))
            self.y_train = pd.read_pickle('{}_y_train.pkl'.format(final_from_file_name))
            self.total_sales_train = pd.read_pickle('{}_total_sales_train.pkl'.format(final_from_file_name))
            self.X_validate = pd.read_pickle('{}_X_validate.pkl'.format(final_from_file_name))
            self.y_validate = pd.read_pickle('{}_y_validate.pkl'.format(final_from_file_name))
            self.total_sales_validate = pd.read_pickle('{}_total_sales_validate.pkl'.format(final_from_file_name))
            self.X_test = pd.read_pickle('{}_X_test.pkl'.format(final_from_file_name))
            self.y_test = pd.read_pickle('{}_y_test.pkl'.format(final_from_file_name))
            self.total_sales_test = pd.read_pickle('{}_total_sales_test.pkl'.format(final_from_file_name))
            self.X_train_validate = pd.read_pickle('{}_X_train_validate.pkl'.format(final_from_file_name))
            self.y_train_validate = pd.read_pickle('{}_y_train_validate.pkl'.format(final_from_file_name))
            self.total_sales_train_validate = pd.read_pickle('{}_total_sales_train_validate.pkl'.format(final_from_file_name))
        elif initial_add_features_fn: # initial_add_features_fn builds 'total_sales' - change in future
            train_indices, validate_indices, test_indices = get_train_validate_test_indices(self.df, 28)
            self.X_train = self.df.iloc[train_indices, :]
            self.y_train = self.X_train[['id', 'date', 'units_sold']].copy(deep = True).set_index(['id', 'date'])['units_sold']
            self.total_sales_train = self.X_train[['id', 'date', 'total_sales']].copy(deep = True).set_index(['id', 'date'])['total_sales']
            self.X_validate = self.df.iloc[validate_indices, :]
            self.y_validate = self.X_validate[['id', 'date', 'units_sold']].copy(deep = True).set_index(['id', 'date'])['units_sold']
            self.total_sales_validate = self.X_validate[['id', 'date', 'total_sales']].copy(deep = True).set_index(['id', 'date'])['total_sales']
            self.X_test = self.df.iloc[test_indices, :]
            self.y_test = self.X_test[['id', 'date', 'units_sold']].copy(deep = True).set_index(['id', 'date'])['units_sold']
            self.total_sales_test = self.X_test[['id', 'date', 'total_sales']].copy(deep = True).set_index(['id', 'date'])['total_sales']
            self.X_train_validate = pd.concat([self.X_train, self.X_validate])
            self.y_train_validate = pd.concat([self.y_train, self.y_validate])
            self.total_sales_train_validate = pd.concat([self.total_sales_train, self.total_sales_validate])
        
            if verbose:
                print('[{}] Fit-transforming through pipeline..'.format(datetime.datetime.now()))
            if pipeline:
                self.X_train = pipeline.fit_transform(self.X_train)
                self.X_train = pd.DataFrame(data = self.X_train, columns = pipeline[0].get_feature_names(), index = self.y_train.index)
                self.X_validate = pipeline.transform(self.X_validate)
                self.X_validate = pd.DataFrame(data = self.X_validate, columns = pipeline[0].get_feature_names(), index = self.y_validate.index)
                self.X_test = pipeline.transform(self.X_test)
                self.X_test = pd.DataFrame(data = self.X_test, columns = pipeline[0].get_feature_names(), index = self.y_test.index)
                self.X_train_validate = pipeline.transform(self.X_train_validate)
                self.X_train_validate = pd.DataFrame(data = self.X_train_validate, columns = pipeline[0].get_feature_names(), index = self.y_train_validate.index)
            if verbose:
                print('[{}] Finish fit-transforming through pipeline..'.format(datetime.datetime.now()))
                
            if final_to_file_loc:
                final_to_file_name = final_to_file_loc.split('.')[0]
                # np.save('{}_X_train.npy'.format(final_to_file_name), self.X_train)
                self.X_train.to_pickle('{}_X_train.pkl'.format(final_to_file_name))
                self.y_train.to_pickle('{}_y_train.pkl'.format(final_to_file_name))
                self.total_sales_train.to_pickle('{}_total_sales_train.pkl'.format(final_to_file_name))
                # np.save('{}_X_validate.npy'.format(final_to_file_name), self.X_validate)
                self.X_validate.to_pickle('{}_X_validate.pkl'.format(final_to_file_name))
                self.y_validate.to_pickle('{}_y_validate.pkl'.format(final_to_file_name))
                self.total_sales_validate.to_pickle('{}_total_sales_validate.pkl'.format(final_to_file_name))
                # np.save('{}_X_test.npy'.format(final_to_file_name), self.X_test)
                self.X_test.to_pickle('{}_X_test.pkl'.format(final_to_file_name))
                self.y_test.to_pickle('{}_y_test.pkl'.format(final_to_file_name))
                self.total_sales_test.to_pickle('{}_total_sales_test.pkl'.format(final_to_file_name))
                # np.save('{}_X_train_validate.npy'.format(final_to_file_name), self.X_train_validate)
                self.X_train_validate.to_pickle('{}_X_train_validate.pkl'.format(final_to_file_name))
                self.y_train_validate.to_pickle('{}_y_train_validate.pkl'.format(final_to_file_name))
                self.total_sales_train_validate.to_pickle('{}_total_sales_train_validate.pkl'.format(final_to_file_name))
    
    def build_starting_df_specific_i(self, st_exclude_last_n_days, st_csv_i, sp_csv_i):
        calendar = self.get_calendar_df()
        sp = self.get_sell_prices_df(csv_i = sp_csv_i)
        st = self.get_sales_train_validation_df(st_exclude_last_n_days, csv_i = st_csv_i)
        return self.build_starting_df_from_three_dfs(st, calendar, sp)
        
    def get_calendar_df(self):
        cal_dtype = {
            'date': 'str', 
            'wm_yr_wk': 'int16', 
            'weekday': 'category', 
            'wday': 'int16', 
            'month': 'int16', 
            'year': 'int16', 
            'd': 'str', 
            'event_name_1': 'category', 
            'event_type_1': 'category', 
            'event_name_2': 'category', 
            'event_type_2': 'category', 
            'snap_CA': 'int16', 
            'snap_TX': 'int16', 
            'snap_WI': 'int16'
        }
        cal_dates = ['date']
        calendar = pd.read_csv('calendar.csv', dtype = cal_dtype, parse_dates = cal_dates)
        return calendar

    def get_sell_prices_df(self, csv_i = None):
        sp_dtype = {
            'store_id': 'category', 
            'item_id': 'category', 
            'wm_yr_wk': 'int16', 
            'sell_price': 'float32'
        }
        sp_cols = ['store_id', 'item_id', 'wm_yr_wk', 'sell_price']
        file_loc = 'sell_prices.csv'

        if csv_i != None:
            sp_start_i, sp_end_i = csv_i
            sp = pd.read_csv(file_loc, dtype = sp_dtype, header = 0, skiprows = range(1, sp_start_i + 1), nrows = sp_end_i - sp_start_i + 1) # + 1 because sp_end_i is inclusive
        else:
            sp = pd.read_csv(file_loc, dtype = sp_dtype, header = 0)
        return sp

    def get_sales_train_validation_df(self, exclude_last_n_days, csv_i = None):
        st_cat_cols = ['id', 'item_id', 'dept_id', 'cat_id','store_id', 'state_id']
        # sales_train_validation last day is d_1913
        st_num_cols = list(map(lambda x: 'd_{}'.format(x), range(1, 1914 - exclude_last_n_days)))
        st_dtype = {col: 'category' for col in st_cat_cols}
        st_dtype.update({col: 'int16' for col in st_num_cols})
        file_loc = 'sales_train_validation.csv'

        if csv_i != None:
            st_start_i, st_end_i = csv_i
            st = pd.read_csv(file_loc, dtype = st_dtype, header = 0, skiprows = range(1, st_start_i + 1), nrows = st_end_i - st_start_i + 1)
        else:
            st = pd.read_csv(file_loc, dtype = st_dtype, header = 0)
        st = pd.melt(st, id_vars = st_cat_cols, value_vars = st_num_cols, var_name = 'd', value_name = 'units_sold')
        return st
    
    def build_starting_df_from_three_dfs(self, st, cal, sp):
        df = pd.merge(st, cal, on = 'd', copy = False)
        df = pd.merge(df, sp, on = ['store_id', 'item_id', 'wm_yr_wk'], copy = False)
        df.sort_values(['id', 'date'], inplace = True)
        df.reset_index(inplace = True, drop = True)
        # ideally would sort earlier, but merge has bugs
        # https://github.com/pandas-dev/pandas/pull/24521
        return df

    def calculate_sample_weight(self, w_fn = None, to_file_loc = None, from_file_loc = None): # make w_fn a transformer class in the future
        if from_file_loc:
            from_file_name = from_file_loc.split('.')[0]
            self.w_train = pd.read_pickle('{}_w_train.pkl'.format(from_file_name))
            self.w_validate = pd.read_pickle('{}_w_validate.pkl'.format(from_file_name))
            self.w_test = pd.read_pickle('{}_w_test.pkl'.format(from_file_name))
            self.w_train_validate = pd.read_pickle('{}_w_train_validate.pkl'.format(from_file_name))
        else:
            self.w_train = w_fn(self.y_train, self.total_sales_train)
            self.w_validate = self.y_validate.reset_index().merge(
                    self.w_train.groupby(level = 'id', observed = True).min().reset_index(), 
                    how = 'left'
                ).drop(['units_sold'], axis = 1)\
                .set_index(['id', 'date'])\
                .fillna(0)\
                .iloc[:, 0]
                # null values mean item was not in train set - can fill as 0
            self.w_test = self.y_test.reset_index().merge(
                    self.w_train.groupby(level = 'id', observed = True).min().reset_index(), 
                    how = 'left'
                ).drop(['units_sold'], axis = 1)\
                .set_index(['id', 'date'])\
                .fillna(0)\
                .iloc[:, 0]
            # self.w_train_validate is used when using train_validate set as training set, so fit w_fn from scratch (rather than transform)
            self.w_train_validate = w_fn(self.y_train_validate, self.total_sales_train_validate)

        if to_file_loc:
            to_file_name = to_file_loc.split('.')[0]
            self.w_train.to_pickle('{}_w_train.pkl'.format(to_file_name))
            self.w_validate.to_pickle('{}_w_validate.pkl'.format(to_file_name))
            self.w_test.to_pickle('{}_w_test.pkl'.format(to_file_name))
            self.w_train_validate.to_pickle('{}_w_train_validate.pkl'.format(to_file_name))

### PIPELINE CONSTRUCT ###
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    
    def fit(self, df, y = None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
    def transform(self, df, y = None):
        return df[self.features]
    
    def get_feature_names(self):
        return self.features
    
class DayExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_col = 'date'):
        self.date_col = date_col
    
    def fit(self, df, y = None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
    def transform(self, df, y = None):
        return pd.to_datetime(df[self.date_col]).dt.day[:, np.newaxis]
    
    def get_feature_names(self):
        return ['day']
    
class FloatYearTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_col = 'date'):
        self.date_col = date_col
        
    def fit(self, df, y = None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
    def transform(self, df, y = None):
        return pd.to_datetime(df[self.date_col])\
                .map(lambda x: x.year + ((x.month-1)/12) + 
                     ((x.day-1)/x.daysinmonth/12)
                    )\
                [:, np.newaxis]
    
    def get_feature_names(self):
        return ['float_year']

class AggMetricsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y = None):
        merge_lst = [
            get_overall_trend(df, 
                              groupby_cols=['id', lambda x: 'all_time'], 
                              agg_cols=['units_sold'])
            ]
        builtin_aggs_col = ['wday', 'month', 'year', 'SNAP_allowed', lambda x: 'all_time']
        for col in builtin_aggs_col:
            merge_lst.append(
                get_builtin_aggs(df, 
                                 groupby_cols = ['id', col], 
                                 agg_cols = ['units_sold', 'sell_price'], 
                                 agg_funcs = ['mean', 'std'])
                )
        self.train_agg_df = pd.concat(merge_lst, axis = 1)
        return self
    
    def transform(self, df, y = None):
        return df.merge(
                self.train_agg_df, 
                how = 'left', 
                left_on = 'id', 
                right_index = True
            )[self.train_agg_df.columns]

    def get_feature_names(self):
        return self.train_agg_df.columns
