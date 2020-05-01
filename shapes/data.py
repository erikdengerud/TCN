# data.py

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

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


class ShapeDataset(Dataset):
    """
    A dataset of shapes with different frequencies and noise.
    """

    def __init__(
        self,
        N: int = 50,
        t: int = 100,
        balance: bool = False,
        h_batch: int = 100,
        predict_ahead: int = 1,
        receptive_field: int = 20,
        mean: float = 0,
        var: float = 1.0,
    ) -> None:

        self.N = N
        self.t = t
        self.balance = balance
        self.h_batch = h_batch
        self.predict_ahead = predict_ahead
        self.receptive_field = receptive_field

        """ Creating the dataset """
        df_ts, df_descriptive = self.create_shape_series(
            N, t, balance=balance, mean=mean, var=var
        )
        X = torch.tensor(df_ts.values)
        X = torch.unsqueeze(X, 1)
        self.X = X
        self.df_descriptive = df_descriptive
        # Creating the labels Y by shifting the time series by predict_ahead
        print(X.shape)
        Y = torch.zeros(self.N, 1, self.t)
        print(Y.shape)
        pad_end = torch.zeros(self.N, 1, self.predict_ahead).double()
        print(pad_end.shape)
        self.Y = Y.copy_(torch.cat((X[:, :, self.predict_ahead :], pad_end), 2)).to(
            dtype=torch.float32
        )

        print("Dimension of X : ", self.X.shape)
        print("Dimension of Y : ", self.Y.shape)

    def __len__(self) -> int:
        if self.h_batch == 0:
            return self.N
        else:
            return int((self.N * self.t) / self.h_batch)

    def __getitem__(self, idx: int) -> List[Tensor]:
        """ 
        Returns a batch of the dataset (X and Y), the ids of the batch and the time series
        id of the batch.  
        """
        # We get a random index in the range [0, int(num_ts * lenght_ts) / h_batch)].
        # Then we create the sample for that index. The sample is the next h_batch and
        # the previous receptive_field. The Y is only the values of length h_batch after
        # the index. This means that the length of X and Y differ.
        if self.h_batch == 0:
            X = self.X[idx]
            Y = self.Y[idx]

            return X, Y, idx, idx

        else:
            row, column = self.get_row_column(idx)
            # Handling some edge cases just to be sure
            if column < self.receptive_field:
                column = self.receptive_field
            if column + self.h_batch > self.t:
                column = self.t - self.h_batch

            X = self.X[row, :, column - self.receptive_field : column + self.h_batch]
            Y = self.Y[row, :, column : column + self.h_batch]
            return X, Y, idx, row

    def get_row_column(self, idx: int) -> List[int]:
        """ Gets row and column based on idx, num_ts and length_ts """
        row = 0
        column = idx * self.h_batch
        while column > self.t:
            column -= self.t
            row += 1
        return row, column

    def create_shape_series(
        self, N: int, t: int, balance: bool, mean: float = 0, var: float = 1.0
    ):
        ts = []
        descriptive = []
        for i in range(N):
            # Randomly choose shape, frequency and noise
            shape = np.random.choice(["sine", "square", "triangle"])
            noise = np.random.choice(["iid", "matern", None])
            period = np.random.choice([2, 10, 20])
            s = self.shape_series(
                shape, length=t, period=period, noise=noise, mean=mean, var=var
            )
            ts.append(s)
            descriptive.append([shape, noise, str(period)])

        df_ts = pd.DataFrame(ts)
        df_descriptive = pd.DataFrame(descriptive)
        df_descriptive.columns = ["shape", "noise", "period"]
        return df_ts, df_descriptive

    def shape_series(
        self, shape, length=100, period=10, mean=0, var=1, noise=None, noise_var=1
    ) -> List[float]:
        """ Creates a shape series and adds noise to it. """
        assert shape in ["sine", "triangle", "square"]
        assert noise in [None, "iid", "matern"]
        if shape == "sine":
            s = self.sine(length, period, mean, var)
        elif shape == "square":
            s = self.square(length, period, mean, var)
        elif shape == "triangle":
            s = self.triangle(length, period, mean, var)
        if noise == "iid":
            s += self.iid_noise(s)
        elif noise == "matern":
            s += self.matern_noise(s)
        return s

    def sine(self, t, p, mean=0, var=1, start=1):
        """ Creating a sine signal with period p. """
        x = np.linspace(1, t, t)
        s = np.sin(1 / p * x * 2 * np.pi)
        s = s * var + mean
        assert len(s) == t
        return s

    def triangle(self, t, p, mean=0, var=1, start=1):
        """ Creating a triangle signal with period p. """
        y = np.array([(1 - 2 / p * x) for x in range(p)])
        s = np.tile(
            np.concatenate((start * y, -start * y), axis=None), t // (2 * p) + 1
        )[:t]
        s = s * var + mean
        assert len(s) == t
        return s

    def square(self, t, p, mean=0, var=1, start=1):
        """ Creating a square signal that alternates every p step. """
        s = np.tile(
            np.concatenate((start * np.ones(p), -start * np.ones(p)), axis=None),
            t // (2 * p) + 1,
        )[:t]
        s = s * var + mean
        assert len(s) == t
        return s

    def iid_noise(self, x, var=0.2):
        noise = var * np.random.randn(len(x))
        x += noise
        return x

    def matern_noise(self, s, var=1, nu=1.5):
        """ nu determines the smoothness. smaller is less smooth """
        gp = GaussianProcessRegressor(
            kernel=var * Matern(length_scale=10, length_scale_bounds=(1e-5, 1e5), nu=nu)
        )
        x = np.linspace(1, len(s), len(s))
        sample = gp.sample_y(x.reshape(-1, 1), 1, random_state=None).flatten()

        return s + sample

    def plot_examples(
        self,
        ids: List = [],
        n: int = 3,
        save_path: str = "shapes/figures/ts_examples.pdf",
        length_plot: int = 20,
    ) -> None:
        if ids:
            time_series = []
            start_point = np.random.randint(0, int(self.t - length_plot))
            for i in ids:
                s = self.X[i, 0, start_point : start_point + length_plot].numpy()
                time_series.append(np.transpose(s))
        else:
            # Choose n randomly selected series and a random start point
            examples_ids = np.random.choice(self.N, size=n, replace=False)
            start_point = np.random.randint(0, int(self.t - length_plot))
            time_series = []
            for example_id in examples_ids:
                s = self.X[
                    example_id, 0, start_point : start_point + length_plot
                ].numpy()
                time_series.append(np.transpose(s))

        df = pd.DataFrame(time_series).T

        df.plot(subplots=True, figsize=(10, 5))
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    print("Shape dataset")
    np.random.seed(1729)
    dataset = ShapeDataset(N=50, t=100, balance=False, predict_ahead=1, h_batch=100,)

    dataset.plot_examples(ids=[], n=5, length_plot=50)

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
    print(dataset.df_descriptive.head())
