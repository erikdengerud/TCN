# data.py
""" 
Dataset classes.
* Random dataset
* Electricity dataset
""" 
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
    '''
    def __init__(
        self, 
        file_path, 
        start_date='2012-01-01', # YYYY-MM-DD 
        end_date='2014-05-26',   # YYYY-MM-DD
        conv=False,
        include_covariates=False
        ):

        # Check dates
        self.start_date = date.fromisoformat(start_date)
        self.end_date = date.fromisoformat(end_date)
        assert self.start_date < self.end_date
        assert self.start_date >= date.fromisoformat('2011-01-01')
        assert self.end_date <= date.fromisoformat('2015-01-01')
        
        self.daterange = pd.date_range(
            start=start_date, end=end_date, freq='H')

        self.conv = conv
        self.include_covariates = include_covariates

        self.Y = None
        self.Z = 0
        self.num_covariates = 0

        if os.path.isfile(
            '.\data\LD2011_2014_aggr_hourly_from_{}_to_{}.txt'.format(
                    self.start_date, self.end_date
                    )
            ): 
            df = pd.read_csv(
                '.\data\LD2011_2014_aggr_hourly_from_{}_to_{}.txt'.format(
                    self.start_date, self.end_date
                    )
                , sep=';', low_memory=False)
            print('Aggregated file already exist.') 

        else:
            df_hourly = pd.read_csv(
                '.\data\LD2011_2014_hourly.txt', sep=';', low_memory=False)
            print(df_hourly.head())
            # Cut off at start and end date
            df_hourly.index = df_hourly['date'] 
            df_hourly = df_hourly.loc[str(self.start_date):str(self.end_date)]
            df = df_hourly.reset_index(drop=True)
            df_hourly.drop('date', axis=1, inplace=True)
            df.to_csv('.\data\LD2011_2014_aggr_hourly_from_{}_to_{}.txt'.format(
                    self.start_date, self.end_date
                    ))
            print(df.head())
            print('Created new file')

        if self.include_covariates:
            '''
            We use 7 time-covariates, which includes minute of
            the hour, hour of the day, day of the week, day of the month, 
            day of the year, month of the year, week of the year, all 
            normalized in a range [−0.5, 0.5], which is a subset of the 
            time-covariates used by default in the GluonTS library. 
            - From the paper.
            '''
            time_index = pd.DatetimeIndex(self.daterange)
            Z = np.matrix([
                MinuteOfHour().__call__(time_index),
                HourOfDay().__call__(time_index),
                DayOfWeek().__call__(time_index), 
                DayOfMonth().__call__(time_index), 
                DayOfYear().__call__(time_index),
                MonthOfYear().__call__(time_index),
                WeekOfYear().__call__(time_index)
                ])
            self.Z = torch.from_numpy(Z)
            self.num_covariates = self.Z.shape[0]

        # Convert to torch matrix
        tens = torch.tensor(df_hourly.values)
        #tens = torch.tensor(df_hourly.values)
        print('Dimensions of the tensor is : ', tens.shape)
        # Transpose to get a n x t matrix.
        self.Y = torch.transpose(tens, 0, 1)
        print('Dimensions of Y is : ', self.Y.shape)
        
    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.conv:
            # Return the series
            Y = self.Y[idx, :].view(1, -1)
            # Return the covariates. The number of channels is the number 
            # of covariates.
            if self.num_covariates > 0:
                Z = self.Z.view(self.num_covariates, -1)
            else:
                Z = self.Z
            return Y, idx, Z
        else:
            return torch.from_numpy(self.Y[idx, :]), idx, self.Z

class AddTwoDataSet(Dataset):
    """ 
    Adding dataset as described in https://arxiv.org/abs/1803.01271.
    Implementation based on:
    https://github.com/locuslab/TCN/blob/master/TCN/adding_problem/utils.py
    """
    def __init__(self, N, seq_length):
        self.N = N
        self.seq_length = seq_length

        X_rand = torch.rand(N, 1, seq_length)
        X_add = torch.zeros(N, 1, seq_length)
        self.Y = torch.zeros(N, 1)

        for i in range(N):
            ones = np.random.choice(seq_length, size=2, replace=False)
            X_add[i, 0, ones[0]] = 1
            X_add[i, 0, ones[1]] = 1
            self.Y[i, 0] = X_rand[i, 0, ones[0]] + X_rand[i, 0, ones[1]]

        self.X = torch.cat((X_rand, X_add), dim=1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx, 0, :], self.Y[idx, 0]

if __name__ == "__main__":
    electricity = False
    add = True

    if electricity:
        # Electricity dataset
        print("Electricity dataset: ")
        dataset = ElectricityDataSet(
        '.\data\LD2011_2014_hourly.txt', 
        conv=True, 
        include_covariates=True,
        start_date='2013-03-03',
        end_date='2014-02-03')
        loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
        dataiter = iter(loader)
        samples, indices , covariates = dataiter.next()
        print('Samples : ', samples)
        print('Indices : ', indices)
        print('Covariates : ', covariates)
        print('Shape of samples : ', samples.shape)
        print('Length of dataset: ', dataset.__len__())
        print('Shape of input: ', samples.shape)
        print('Input length?: ', samples.shape[2])

    if add:
        # Add two dataset
        print("Add Two dataset: ")
        dataset = AddTwoDataSet(N=1000, seq_length=64)
        loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
        dataiter = iter(loader)
        samples, labels = dataiter.next()
        print('Samples : ', samples)
        print('Labels : ', labels)
        print('Shape of samples : ', samples.shape)
        print('Length of dataset: ', dataset.__len__())
        print('Shape of input: ', samples.shape)