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

        self.include_covariates = include_covariates

        self.Y = None
        self.Z = 0
        self.num_covariates = 0

        if os.path.isfile(
            '.\electricity\data\LD2011_2014_aggr_hourly_from_{}_to_{}.txt'.format(
                    self.start_date, self.end_date
                    )
            ): 
            df = pd.read_csv(
                '.\ectricity\data\LD2011_2014_aggr_hourly_from_{}_to_{}.txt'.format(
                    self.start_date, self.end_date
                    )
                , sep=';', low_memory=False)
            print('Aggregated file already exist.') 

        else:
            df_hourly = pd.read_csv(
                file_path, sep=';', low_memory=False)
            print(df_hourly.head())
            # Cut off at start and end date
            df_hourly.index = df_hourly['date'] 
            df_hourly = df_hourly.loc[str(self.start_date):str(self.end_date)]
            df = df_hourly.reset_index(drop=True)
            df_hourly.drop('date', axis=1, inplace=True)
            df.to_csv('.\electricity\data\LD2011_2014_aggr_hourly_from_{}_to_{}.txt'.format(
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

if __name__ == "__main__":
    # Electricity dataset
    print("Electricity dataset: ")
    dataset = ElectricityDataSet(
    '.\electricity\data\LD2011_2014_hourly.txt', 
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
