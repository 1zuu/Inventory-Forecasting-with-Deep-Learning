import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from pandas.core.common import SettingWithCopyWarning
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from variables import *

def get_data():
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)

    return train_data, test_data

def date_features(df, train=False):

    if train:
        df = df[input_columns + output_columns]
    else:
        df = df[['id'] + input_columns]

    df['date'] = pd.to_datetime(df['datetime'])
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['day'] = df.date.dt.day
    df['dayofyear'] = df.date.dt.dayofyear
    df['dayofweek'] = df.date.dt.dayofweek
    df['weekofyear'] = df.date.dt.isocalendar().week
    df['day^year'] = np.log((np.log(df['dayofyear'] + 1)) ** (df['year'] - 2000))
    
    df.drop(['date', 'datetime'], axis=1, inplace=True)
    return df

def average_data(train_data):
    train_data['daily_avg']  = train_data.groupby(['item','store','dayofweek'])['sales'].transform('mean')
    train_data['monthly_avg'] = train_data.groupby(['item','store','month'])['sales'].transform('mean')
    train_data = train_data.dropna()

    daily_avg = train_data.groupby(['item','store','dayofweek'])['sales'].mean().reset_index()
    monthly_avg = train_data.groupby(['item','store','month'])['sales'].mean().reset_index()
    return daily_avg, monthly_avg

def merge_averages(df, df_avg, columns,column_avg):
    
    df =pd.merge(df, df_avg, how='left', on=None, left_on=columns, right_on=columns,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False)
    
    df = df.rename(
                columns={
                    'sales':column_avg
                        }
                  )

    return df

def add_features(train_data, test_data):
    daily_avg, monthly_avg = average_data(train_data)
    
    test_data = merge_averages(test_data, daily_avg,['item','store','dayofweek'],'daily_avg')
    test_data = merge_averages(test_data, monthly_avg,['item','store','month'],'monthly_avg')

    rolling_10 = train_data.groupby(['item'])['sales'].rolling(10).mean().reset_index().drop('level_1', axis=1)
    train_data['rolling_mean'] = rolling_10['sales'] 

    rolling_last90 = train_data.groupby(['item','store'])['rolling_mean'].tail(90).copy()
    test_data['rolling_mean'] = rolling_last90.reset_index().drop('index', axis=1)

    train_data['rolling_mean'] = train_data.groupby(['item'])['rolling_mean'].shift(90) 
    return train_data, test_data

def normalize_data(train_data, test_data):
    for df in [train_data, test_data]:
        df.drop(['dayofyear', 'weekofyear','daily_avg','day','month','item','store',],axis=1, inplace=True)

    test_data = test_data.dropna()    
    train_data = train_data.dropna() 
    test_data.sort_values(by=['id'], inplace=True)

    sales, ids = train_data['sales'].values, test_data['id'].values
    del train_data['sales'], test_data['id']

    scalar = StandardScaler()
    scalar.fit(train_data)

    Xtest = scalar.transform(test_data)
    X, Y = scalar.transform(train_data), sales

    Y = Y.reshape(-1,1)
    minmax_scaler = MinMaxScaler()
    Y = minmax_scaler.fit_transform(Y)
    Y = Y.reshape(-1,)

    if not os.path.exists(model_weights):
        X, Y = shuffle(X, Y)

    return X, Y, Xtest, minmax_scaler

def load_Data():
    train_data, test_data = get_data()

    test_data = date_features(test_data)
    train_data = date_features(train_data, True)

    train_data, test_data = add_features(train_data, test_data)
    X, Y, Xtest, minmax_scaler = normalize_data(train_data, test_data)
    return X, Y, Xtest, minmax_scaler


def process_sample_input(sample_input):
        # datetime = sample_input['datetime']
        # store = sample_input['store']
        # item = sample_input['datetime']

    df = pd.read_csv(test_csv_path)
    store = df['store'].values
    item = df['item'].values

    print(len(set(zip(store, item))))

process_sample_input(sample_input)