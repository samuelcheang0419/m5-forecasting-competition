import numpy as np
import pandas as pd
from utilities import timer, get_slope

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
    return df

@timer
def build_starting_df_specific_i(sp_start_i, sp_end_i, st_start_i, st_end_i, st_first_day_int = 1, st_last_day_int = 1913):
    calendar = get_calendar_df()
    sp = get_sell_prices_df(from_csv_i = (sp_start_i, sp_end_i))
    st = get_sales_train_validation_df(from_csv_i = (st_start_i, st_end_i), first_day_int = st_first_day_int, last_day_int = st_last_day_int)
    df = build_starting_df_from_three_dfs(st, calendar, sp)
    return df

### ADD FEATURES ###
def feature_SNAP_allowed(df):
    df['SNAP_allowed'] = pd.concat([((df['state_id'] == state_id) & (df['snap_{}'.format(state_id)])) for state_id in ['CA', 'TX', 'WI']], axis = 1).any(axis = 1)
    
def feature_special_events(df, events_d, variable_length_events_l):
    def single_item_special_events(item_df):
        item_df_max_date = item_df['date'].max()

        for event in events_d.keys():
            if event in ['LentWeek2', 'NBAFinalsEnd']:
                continue
            item_df['is_event'] = ((item_df['event_name_1'] == event) | (item_df['event_name_2'] == event))
            item_df['date_only_event'] = np.where(item_df['is_event'] == True, item_df['date'].astype(int), np.nan) # np.where doesn't work well with date types
            item_df['nearest_future_event_date'] = item_df.loc[::-1, 'date_only_event'].expanding(1).min()[::-1]
            item_df['nearest_future_event_date'] = pd.to_datetime(item_df['nearest_future_event_date'], unit = 'ns')
            # get event dates beyond dataset
            nearest_future_event_date = list(filter(lambda x: pd.to_datetime(x) > item_df_max_date, events_d[event]))[0]
            null_future_event_date_mask = item_df['nearest_future_event_date'].isnull()
            item_df.loc[null_future_event_date_mask, 'nearest_future_event_date'] = pd.to_datetime(nearest_future_event_date)
            item_df['x_days_before_first_day_of_' + event.replace(' ', '').replace("'", '')] = (item_df['nearest_future_event_date'] - item_df['date']).dt.days
            # harder to do x_days_before_LAST_day as events like NBAFinalsEnd is unknown
            item_df.drop(['is_event', 'date_only_event', 'nearest_future_event_date'], axis = 1, inplace = True)

        for event in variable_length_events_l:
            item_df['is_' + event.replace(' ', '').replace("'", '')] = ((item_df['event_name_1'] == event) | (item_df['event_name_2'] == event))
            
        return item_df
            
    special_event_df = df[['id', 'date', 'event_name_1', 'event_name_2']].groupby('id').apply(single_item_special_events)
    df[special_event_df.columns] = special_event_df
    
def feature_total_sales(df):
    df['total_sales'] = df['units_sold'] * df['sell_price']
    
def feature_lag(df, d):
    df['lag_' + str(d) + '_units_sold'] = df['units_sold'].shift(d)
    
def feature_rolling_average_of_lag(df, rolling_d, lag_d):
    temp_col = 'TEMP_lag1'
    df[temp_col] = df['units_sold'].shift(lag_d)
    df['rolling_average_{}_of_lag_{}_units_sold'.format(rolling_d, lag_d)] = df[temp_col].rolling(rolling_d).mean()
    df.drop([temp_col], axis = 1, inplace = True)
    
def feature_rolling_trend_of_lag(df, rolling_d, lag_d):
    temp_col = 'TEMP_lag1'
    df[temp_col] = df['units_sold'].shift(lag_d)
    df['rolling_trend_{}_of_lag_{}_units_sold'.format(rolling_d, lag_d)] = df[temp_col].rolling(rolling_d).apply(lambda x: get_slope(x))
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
    