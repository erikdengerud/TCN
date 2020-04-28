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
    ) -> None:

        self.N = N
        self.t = t
        self.balance = balance
        self.h_batch = h_batch
        self.predict_ahead = predict_ahead
        self.receptive_field = receptive_field

        """ Creating the dataset """
        df = self.create_shape_series(N, t, balance)
        X = torch.tensor(df.values)
        X = torch.unsqueeze(X, 1)
        self.X = X

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

    def create_shape_series(self, N, t, balance):
        for i in range(N):
            # Randomly choose shape, frequency and noise

        X = np.random.rand(N, t)
        df = pd.DataFrame(X)
        return df

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

    dataset.plot_examples(ids=[], n=5, length_plot=20)

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
