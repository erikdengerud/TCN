# data.py

from datetime import date, timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

sys.path.append("./")
sys.path.append("../../")
from typing import Tuple, List

import torch
import torch.tensor as Tensor
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

from utils.time import (
    MinuteOfHour,
    HourOfDay,
    DayOfWeek,
    DayOfMonth,
    DayOfYear,
    MonthOfYear,
    WeekOfYear,
)

from sklearn.preprocessing import OneHotEncoder


class ElectricityDataSet(Dataset):
    """
    Load and prepare the electricity data set.
    The data is downloaded using the script data/download_data.sh which is copied from the
    DeepGLO paper. 
    """

    def __init__(
        self,
        file_path: str,
        start_date: str = "2012-01-01",  # YYYY-MM-DD
        end_date: str = "2014-05-26",  # YYYY-MM-DD
        include_time_covariates: bool = False,
        predict_ahead: int = 1,
        h_batch: int = 0,  # 0 gives the whole time series
        one_hot_id: bool = False,
        receptive_field: int = 385,
        scale: bool = False,
    ) -> None:
        """ Dates """
        # Check dates
        self.start_date = date.fromisoformat(start_date)
        self.end_date = date.fromisoformat(end_date)
        assert self.start_date < self.end_date
        assert self.start_date >= date.fromisoformat("2011-01-01")
        assert self.end_date <= date.fromisoformat("2015-01-01")
        # The date range of the dataset wanted
        self.daterange = pd.date_range(start=start_date, end=end_date, freq="H")

        """ Input parameters """
        self.one_hot_id = one_hot_id
        self.include_time_covariates = include_time_covariates
        self.predict_ahead = predict_ahead
        self.h_batch = h_batch
        self.receptive_field = receptive_field
        if scale:
            self.scaler = MinMaxScaler()

        """ Creating the dataset """
        # Extract the time series from the file and store as tensor X
        df, dates = self.get_time_range_df(
            file_path, start_date=start_date, end_date=end_date
        )
        if scale:
            values = self.scaler.fit_transform(df_values)
        else:
            values = df.values
        X = torch.tensor(scaled_values)
        X = torch.transpose(X, 0, 1)
        self.num_ts = X.shape[0]
        self.length_ts = X.shape[1]

        # Creating the labels Y by shifting the time series by predict_ahead
        X = torch.unsqueeze(X, 1)
        Y = torch.zeros(self.num_ts, 1, self.length_ts)
        pad_end = torch.zeros(self.num_ts, 1, self.predict_ahead).double()
        self.Y = Y.copy_(torch.cat((X[:, :, self.predict_ahead :], pad_end), 2)).to(
            dtype=torch.float32
        )

        # Including time covariates
        if self.include_time_covariates:
            Z, num_covariates = self.get_time_covariates(dates)
            Z = Z.repeat(self.num_ts, 1, 1)
            X = torch.cat((X, Z), 1)

        # Using the IDs of the time series as one-hot encoded covariates. The matrix is
        # too big to be stored in memory if we use on-hot encoding, so we specify the
        # encoding here and concatenate the encoding to the time series in the __getitem__
        # method.
        if self.one_hot_id:
            ids = [[i] for i in range(self.num_ts)]
            self.enc = OneHotEncoder(handle_unknown="ignore")
            self.enc.fit(ids)

        self.X = X.to(dtype=torch.float32)

        print("Dimension of X : ", self.X.shape)
        print("Dimension of Y : ", self.Y.shape)

    def __len__(self) -> int:
        if self.h_batch == 0:
            return self.num_ts
        else:
            return int((self.num_ts * self.length_ts) / self.h_batch)

    def __getitem__(self, idx: int) -> List[Tensor]:
        """ 
        Returns a batch of the dataset (X and Y), the ids of the batch and the time series
        id of the batch.  
        One hot encoded IDs are also added here.
        """
        # We get a random index in the range [0, int(num_ts * lenght_ts) / h_batch)].
        # Then we create the sample for that index. The sample is the next h_batch and
        # the previous receptive_field. The Y is only the values of length h_batch after
        # the index. This means that the length of X and Y differ.
        if self.h_batch == 0:
            X = self.X[idx]
            Y = self.Y[idx]

            if self.one_hot_id:
                if isinstance(idx, (list, np.ndarray)):
                    idx_enc = [[d] for d in idx]
                else:
                    idx_enc = [idx]
                """ Could store all encodings as a matrix E ?"""
                encoded = torch.from_numpy(self.enc.transform([idx_enc]).toarray())
                encoded = encoded.repeat(self.length_ts, 1)
                encoded = encoded.float()
                encoded = torch.transpose(encoded, 0, 1)

                X = torch.cat((X, encoded), 0)
            return X, Y, idx, idx

        else:
            row, column = self.get_row_column(idx)
            # Handling some edge cases just to be sure
            if column < self.receptive_field:
                column = self.receptive_field
            if column + self.h_batch > self.length_ts:
                column = self.length_ts - self.h_batch

            X = self.X[row, :, column - self.receptive_field : column + self.h_batch]
            Y = self.Y[row, :, column : column + self.h_batch]
            if self.one_hot_id:
                row_enc = [row]
                encoded = torch.from_numpy(self.enc.transform([row_enc]).toarray())
                encoded = encoded.repeat(self.h_batch + self.receptive_field, 1)
                encoded = encoded.float()
                encoded = torch.transpose(encoded, 0, 1)

                X = torch.cat((X, encoded), 0)
            return X, Y, idx, row

    def get_row_column(self, idx: int) -> List[int]:
        """ Gets row and column based on idx, num_ts and length_ts """
        row = 0
        column = idx * self.h_batch
        while column > self.length_ts:
            column -= self.length_ts
            row += 1
        return row, column

    def get_time_range_df(
        self, file_path: str, start_date: str, end_date: str
    ) -> List[pd.Series]:
        """ 
        Reads the file with teh time series and crops based on the start and end date 
        of the datasset. 
        """
        mat = np.load(file_path)
        df = pd.DataFrame(mat.T)
        # Create the  date range
        dates_index = pd.date_range(start="2012/01/01", periods=mat.shape[1], freq="H")
        # Cut off at start and end date
        df.index = dates_index
        df = df.loc[str(start_date) : str(end_date)]
        dates = df.index
        df = df.reset_index(drop=True)
        return df, dates

    def get_time_covariates(self, dates: pd.Series) -> Tuple[Tensor, int]:
        """ Creating time covariates normalized to the range [-0.5, 0.5] using GluonTS. """
        time_index = pd.DatetimeIndex(dates)
        time_index = pd.DatetimeIndex(time_index)
        Z = np.matrix(
            [
                MinuteOfHour(time_index),
                HourOfDay(time_index),
                DayOfWeek(time_index),
                DayOfMonth(time_index),
                DayOfYear(time_index),
                MonthOfYear(time_index),
                WeekOfYear(time_index),
            ]
        )
        Z = torch.from_numpy(Z)
        num_covariates = Z.shape[0]
        return Z, num_covariates

    def plot_examples(
        self,
        ids: List = [],
        n: int = 3,
        length_plot: int = 168,
        save_path: str = "electricity/figures/ts_examples.pdf",
        logy: bool = True,
    ) -> None:
        if ids:
            time_series = []
            start_point = (
                np.random.randint(0, int((self.length_ts - length_plot) / 24)) * 24
            )
            for i in ids:
                s = self.X[i, 0, start_point : start_point + length_plot].numpy()
                time_series.append(np.transpose(s))
        else:
            # Choose n randomly selected series and a random start point
            examples_ids = np.random.choice(370, size=n, replace=False)
            start_point = (
                np.random.randint(0, int((self.length_ts - length_plot) / 24)) * 24
            )
            time_series = []
            for example_id in examples_ids:
                s = self.X[
                    example_id, 0, start_point : start_point + length_plot
                ].numpy()
                time_series.append(np.transpose(s))

        df = pd.DataFrame(time_series).T

        # Get datetime start to use on the x axis
        start_date = self.start_date + timedelta(hours=start_point)
        t_range = timedelta(hours=length_plot)
        end_date = start_date + t_range
        start_date = start_date.isoformat()
        end_date = end_date.isoformat()

        d_range = pd.date_range(start=start_date, end=end_date, freq="H")[:-1]
        df.index = d_range

        df.plot(subplots=True, figsize=(10, 2), logy=logy)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    # Electricity dataset
    print("Electricity dataset: ")
    np.random.seed(1729)
    dataset = ElectricityDataSet(
        "electricity/data/electricity.npy",
        include_time_covariates=True,
        start_date="2012-06-01",
        end_date="2014-12-18",
        predict_ahead=3,
        h_batch=256,
        one_hot_id=True,
    )

    dataset.plot_examples(ids=[43], n=5, logy=False, length_plot=168)

    data_loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    dataiter = iter(data_loader)
    x, y, idx, row = dataiter.next()
    data = dataiter.next()
    print(type(data))
    # print(data)
    print("idx", idx)
    # print('Samples : ', x)
    print("Shape of samples : ", x.shape)
    # print('Labels : ', y)
    print("Shape of labels : ", y.shape)
    print("Length of dataset: ", dataset.__len__())
    print("Type x : ", x.dtype)
    print("Type y : ", y.dtype)
    print(x[0, 0, -5:])
    print(y[0, 0, -5:])
    print(dataset.__len__())

    print("Electricity dataset test 2: ")
    dataset = ElectricityDataSet(
        "electricity/data/electricity.npy",
        include_time_covariates=True,
        start_date="2012-06-01",
        end_date="2014-12-18",
        predict_ahead=3,
        h_batch=0,
        one_hot_id=True,
    )

    # dataset.plot_examples(ids=[16, 22, 26], n=3, logy=False, length_plot=168)

    data_loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    dataiter = iter(data_loader)
    x, y, idx, idx_row = dataiter.next()
    data = dataiter.next()
    print(type(data))
    # print(data)
    print("idx", idx)
    # print('Samples : ', x)
    print("Shape of samples : ", x.shape)
    # print('Labels : ', y)
    print("Shape of labels : ", y.shape)
    print("Length of dataset: ", dataset.__len__())
    print("Type x : ", x.dtype)
    print("Type y : ", y.dtype)
    print(x[0, 0, -5:])
    print(y[0, 0, -5:])
    print(dataset.__len__())

    print("Electricity dataset test 3: ")
    dataset = ElectricityDataSet(
        "representations/representation_matrices/electricity_train_scaled.npy",
        include_time_covariates=True,
        start_date="2012-06-01",
        end_date="2014-12-18",
        predict_ahead=3,
        h_batch=0,
        one_hot_id=True,
    )

    dataset.plot_examples(ids=[16, 22, 26], n=3, logy=False, length_plot=168)
