# data.py
from torch.utils.data import Dataset, DataLoader
import torch

import numpy as np
import pandas as pd
import os
import sys
from datetime import date, timedelta
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

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
        h_batch=0,
        one_hot_id=False
        ):

        # Check dates
        self.start_date = date.fromisoformat(start_date)
        self.end_date = date.fromisoformat(end_date)
        assert self.start_date < self.end_date
        assert self.start_date >= date.fromisoformat('2011-01-01')
        assert self.end_date <= date.fromisoformat('2015-01-01')
        self.daterange = pd.date_range(
            start=start_date, end=end_date, freq='H')

        self.one_hot_id = one_hot_id
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

        if self.one_hot_id:
            ids = [[i] for i in range(self.num_ts)]
            self.enc = OneHotEncoder(handle_unknown='ignore')
            self.enc.fit(ids)

        self.X = X.to(dtype=torch.float32)
        #X = X.to(dtype=torch.float32)

        print("Dimension of X : ", self.X.shape)
        print("Dimension of Y : ", self.Y.shape)
        
    def __len__(self):
        return self.num_ts

    def __getitem__(self, idx):
        if self.h_batch == 0:
            X = self.X[idx]
            Y = self.Y[idx]
            
            if self.one_hot_id:
                if isinstance(idx, (list, np.ndarray)):
                    idx_enc = [[d] for d in idx]
                else:
                    idx_enc = [idx]
                """ Could be stored as E """
                encoded = torch.from_numpy(self.enc.transform([idx_enc]).toarray())
                encoded = encoded.repeat(self.length_ts, 1)
                encoded = torch.transpose(encoded, 0, 1)
                encoded = encoded.float()

                X = torch.cat((X, encoded), 0)
                
            return X, Y

        else:
            j = np.random.randint(
                0, self.length_ts-self.h_batch-self.predict_ahead)
                
            X = self.X[idx,:,j:j+self.h_batch]
            Y = self.Y[idx,:,j:j+self.h_batch]

            if self.one_hot_id:
                if isinstance(idx, (list, np.ndarray)):
                    idx_enc = [[d] for d in idx]
                else:
                    idx_enc = [idx]
                
                encoded = torch.from_numpy(self.enc.transform([idx_enc]).toarray())
                encoded = encoded.repeat(self.h_batch,1)
                encoded = torch.transpose(encoded, 0, 1)
                encoded = encoded.float()

                X = torch.cat((X, encoded), 0)

            return X, Y

    def get_time_range_df(self, file_path, start_date, end_date):
        mat = np.load(file_path)
        df = pd.DataFrame(mat.T)
        # create dates
        dates_index = pd.date_range(start='2012/01/01', periods=mat.shape[1], freq='H')
        # Cut off at start and end date
        df.index = dates_index
        df = df.loc[str(start_date):str(end_date)]
        dates = df.index
        df = df.reset_index(drop=True)

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

    def plot_examples(self, ids=[], n=3, length_plot=48, save_path='electricity/figures/ts_examples.pdf', logy=True):
        if ids:
            time_series = []
            for i in ids:
                start_point = np.random.randint(0, int((self.length_ts-length_plot)/24))*24
                s = self.X[i, 0, start_point:start_point+length_plot].numpy()
                time_series.append(np.transpose(s))
        else:
            # Choose n randomly selected series and a random start point
            examples_ids = np.random.choice(370, size=n, replace=False)
            start_point = np.random.randint(0, int((self.length_ts-length_plot)/24))*24
            time_series = []
            for example_id in examples_ids:
                s = self.X[example_id, 0, start_point:start_point+length_plot].numpy()
                time_series.append(np.transpose(s))
    
        # Create df
        df = pd.DataFrame(time_series).T

        # Get datetime start
        start_date = self.start_date + timedelta(hours=start_point)
        t_range = timedelta(hours=length_plot)
        end_date = start_date + t_range
        start_date = start_date.isoformat()
        end_date = end_date.isoformat()

        d_range = pd.date_range(start=start_date, end=end_date, freq='H')[:-1]
        df.index = d_range

        df.plot(subplots=True, figsize=(10,5), logy=logy)
        plt.savefig(save_path)
        plt.show()

        return 0



if __name__ == "__main__":
    # Electricity dataset
    print("Electricity dataset: ")
    np.random.seed(1729)
    dataset = ElectricityDataSet(
    'electricity_dglo_data/data/electricity.npy', 
    include_time_covariates=True,
    start_date='2014-06-01',
    end_date='2014-12-18',
    predict_ahead=3,
    h_batch=0,
    one_hot_id=True)

    #dataset.plot_examples(ids=[16, 22, 26], n=3, logy=False)

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
    

