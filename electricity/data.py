# data.py
from torch.utils.data import Dataset, DataLoader
import torch

import numpy as np
import pandas as pd
import os
from datetime import date

from gluonts.time_feature import (
    MinuteOfHour, 
    HourOfDay, 
    DayOfWeek, 
    DayOfMonth, 
    DayOfYear, 
    MonthOfYear, 
    WeekOfYear)

class ElectricityDataSet(Dataset):
    '''
    Load and prepare the electricity data set.
    https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
    Aggregating to hourly values.

    Parameters:
        file_path           : file path to the hourly aggregated file
        start_date          : start of the wanted dataset as YYYY-MM-DD 
        end_date            : end of the wanted dataset as YYYY-MM-DD 
        conv=False,
        include_covariates  : including time covariates from gluonts or not
    '''
    def __init__(
        self, 
        file_path, 
        start_date='2012-01-01', # YYYY-MM-DD 
        end_date='2014-05-26',   # YYYY-MM-DD
        include_time_covariates=False,
        predict_ahead=1,
        h_batch=0
        ):

        # Check dates
        self.start_date = date.fromisoformat(start_date)
        self.end_date = date.fromisoformat(end_date)
        assert self.start_date < self.end_date
        assert self.start_date >= date.fromisoformat('2011-01-01')
        assert self.end_date <= date.fromisoformat('2015-01-01')
        self.daterange = pd.date_range(
            start=start_date, end=end_date, freq='H')

        self.include_time_covariates = include_time_covariates
        self.predict_ahead = predict_ahead
        self.h_batch = h_batch

        df, dates = self.get_time_range_df(
            file_path, start_date=start_date, end_date=end_date)
        X = torch.tensor(df.values)
        X = torch.transpose(X, 0, 1)
        self.num_ts = X.shape[0]
        self.length_ts = X.shape[1]

        X.resize_(self.num_ts, 1, self.length_ts)
        Y = torch.zeros(self.num_ts, 1, self.length_ts)
        pad_end = torch.zeros(self.num_ts,1,self.predict_ahead).double()
        self.Y = Y.copy_(torch.cat((X[:,:,self.predict_ahead:], pad_end), 2)).to(dtype=torch.float32)

        if self.include_time_covariates:
            Z, num_covariates = self.get_time_covariates(dates)
            Z = Z.repeat(self.num_ts, 1, 1)
            X = torch.cat((X, Z), 1)
        self.X = X.to(dtype=torch.float32)
        X = X.to(dtype=torch.float32)

        print("Dimension of X : ", self.X.shape)
        print("Dimension of Y : ", self.Y.shape)
        
    def __len__(self):
        return self.num_ts

    def __getitem__(self, idx):
        if self.h_batch == 0:
            return self.X[idx], self.Y[idx]
        else:
            j = np.random.randint(
                0, self.length_ts-self.h_batch-self.predict_ahead)
            return self.X[idx,:,j:j+self.h_batch], self.Y[idx,:,j:j+self.h_batch]

    def get_time_range_df(self, file_path, start_date, end_date):
        df_hourly = pd.read_csv(file_path, sep=';', low_memory=False)
        # Cut off at start and end date
        df_hourly.index = df_hourly['date'] 
        df_hourly = df_hourly.loc[str(start_date):str(end_date)]
        df = df_hourly.reset_index(drop=True)
        dates = df['date']
        df.drop('date', axis=1, inplace=True)
        return df, dates

    def get_time_covariates(self, dates):
        '''
        We use 7 time-covariates, which includes minute of
        the hour, hour of the day, day of the week, day of the month, 
        day of the year, month of the year, week of the year, all 
        normalized in a range [−0.5, 0.5], which is a subset of the 
        time-covariates used by default in the GluonTS library. 
        - From the paper.
        '''
        time_index = pd.DatetimeIndex(dates)
        time_index = pd.DatetimeIndex(time_index)
        Z = np.matrix([
            MinuteOfHour().__call__(time_index),
            HourOfDay().__call__(time_index),
            DayOfWeek().__call__(time_index), 
            DayOfMonth().__call__(time_index), 
            DayOfYear().__call__(time_index),
            MonthOfYear().__call__(time_index),
            WeekOfYear().__call__(time_index)
            ])
        Z = torch.from_numpy(Z)
        num_covariates = Z.shape[0]
        return Z, num_covariates

if __name__ == "__main__":
    # Electricity dataset
    print("Electricity dataset: ")

    dataset = ElectricityDataSet(
    'electricity/data/LD2011_2014_hourly.txt', 
    include_time_covariates=True,
    start_date='2013-03-03',
    end_date='2014-02-03',
    predict_ahead=3,
    h_batch=0)

    loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    dataiter = iter(loader)
    x, y = dataiter.next()

    #print('Samples : ', x)
    print('Shape of samples : ', x.shape)
    #print('Labels : ', y)
    print('Shape of labels : ', y.shape)
    print('Length of dataset: ', dataset.__len__())
    print("Type x : ", x.dtype)
    print("Type y : ", y.dtype)
    print(x[0, 0, -5:])
    print(y[0, 0, -5:])