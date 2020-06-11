# data.py

from datetime import date, timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sys

sys.path.append("./")
sys.path.append("../../")
from typing import Tuple, List

import torch
import torch.tensor as Tensor
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder

from clustering.cluster import cluster_dataset


class ElectricityDataSet(Dataset):
    """
    Load and prepare the electricity data set.
    The data is downloaded using the script data/download_data.sh which is copied from the
    DeepGLO paper. 
    """

    def __init__(
        self,
        file_path: str,
        data_scale: bool = True,
        data_scaler=None,
        start_date: str = "2012-01-01",  # yyyy-mm-dd
        end_date: str = "2014-05-26",  # yyyy-mm-dd
        predict_ahead: int = 1,
        h_batch: int = 0,  # 0 gives the whole time series
        receptive_field: int = 385,
        cluster_covariate: bool = False,
        random_covariate: bool = False,
        zero_covariate: bool = False,
        cluster_dict_path: str = None,
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
        assert cluster_covariate + zero_covariate + random_covariate <= 1
        self.cluster_covariate = cluster_covariate
        self.zero_covariate = zero_covariate
        self.random_covariate = random_covariate
        self.predict_ahead = predict_ahead
        self.h_batch = h_batch
        self.receptive_field = receptive_field
        if data_scaler is not None:
            self.data_scaler = data_scaler
        else:
            # self.data_scaler = RobustScaler()
            self.data_scaler = StandardScaler()
        self.data_scale = data_scale

        """ Creating the dataset """
        # Extract the time series from the file and store as tensor X
        df, dates = self.get_time_range_df(
            file_path, start_date=start_date, end_date=end_date
        )
        if self.data_scale:
            try:
                values = self.data_scaler.transform(df.values)
            except:
                values = self.data_scaler.fit_transform(df.values)
        else:
            values = df.values

        X = torch.tensor(values)
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
        self.X = X.to(dtype=torch.float32)

        """ Clustering """
        if cluster_covariate:
            if cluster_dict_path:
                try:
                    with open(cluster_dict_path, "rb") as handle:
                        self.cluster_dict = pickle.load(handle)
                except Exception as e:
                    print(e)
                self.prototypes = self.prototypes_from_cluster_dict(
                    self.X, self.cluster_dict
                )
            else:
                print("No clustering path given.")

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
        """
        # We get a random index in the range [0, int(num_ts * lenght_ts) / h_batch)].
        # Then we create the sample for that index. The sample is the next h_batch and
        # the previous receptive_field. The Y is only the values of length h_batch after
        # the index. This means that the length of X and Y differ.
        if self.h_batch == 0:
            X = self.X[idx]
            Y = self.Y[idx]

            if self.cluster_covariate:
                # find cluster and prototype
                # We only append X since appending Y would give look ahead bias
                c = self.cluster_dict[idx]
                prototype = self.prototypes[c]
                prototype = torch.from_numpy(prototype)

                # add prototype to X
                prototype = prototype.view(1, -1).to(dtype=torch.float32)
                X = torch.cat((X, prototype), 0)

            if self.zero_covariate:
                zero = torch.from_numpy(np.zeros(X.shape[1]))
                zero = zero.view(1, -1).to(dtype=torch.float32)
                X = torch.cat((X, zero), 0)

            if self.random_covariate:
                r = torch.randn(X.shape[1], dtype=torch.float32)
                r = r.view(1, -1)
                X = torch.cat((X, r), 0)

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

            if self.cluster_covariate:
                # find cluster and prototype
                # We only append X since appending Y would give look ahead bias
                c = self.cluster_dict[row]
                prototype = self.prototypes[c]
                prototype = torch.from_numpy(prototype)
                prototype = prototype[
                    column - self.receptive_field : column + self.h_batch
                ]
                # add prototype to X
                prototype = prototype.view(1, -1).to(dtype=torch.float32)
                X = torch.cat((X, prototype), 0)

            if self.zero_covariate:
                zero = torch.from_numpy(np.zeros(self.receptive_field + self.h_batch))
                zero = zero.view(1, -1).to(dtype=torch.float32)
                X = torch.cat((X, zero), 0)

            if self.random_covariate:
                r = torch.randn(
                    self.receptive_field + self.h_batch, dtype=torch.float32
                )
                r = r.view(1, -1)
                X = torch.cat((X, r), 0)

            return X, Y, idx, row

    def prototypes_from_cluster_dict(self, X, cluster_dict):
        """ 
        Calculates prototypes given X and a cluster dict.
        Uses the scaler to scale before taking the mean
        """
        prototypes = {}
        X_scaled = self.data_scaler.transform(X.squeeze().detach().numpy().T).T
        for c in set(cluster_dict.values()):
            # p = np.where()
            p = np.mean(
                X_scaled[[cluster_dict[i] == c for i in range(X_scaled.shape[0])]],
                axis=0,
            )
            prototypes[c] = p

        return prototypes

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

        size = 2 * 4.77
        ax = df.plot(
            subplots=True,
            figsize=(size, 1.5),
            logy=logy,
            color="black",
            legend=False,
            linewidth=1,
            rot=0,
        )
        for i in range(ax.shape[0]):
            x1, x2 = ax[i].get_xlim()
            ax[i].set_xlim((x1, x2 + 1))
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    print("Electricity Dataset")
    dataset = ElectricityDataSet(
        file_path="electricity/data/electricity.npy",
        data_scale=True,
        data_scaler=None,
        start_date="2012-01-01",  # yyyy-mm-dd
        end_date="2014-05-26",  # yyyy-mm-dd
        predict_ahead=1,
        h_batch=0,  # 0 gives the whole time series
        receptive_field=385,
        cluster_covariate=False,
        random_covariate=True,
        zero_covariate=False,
        cluster_dict_path="test_cluster_dict.pkl",
    )
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
    # print(*[dataset.cluster_dict[i] for i in idx.detach().numpy()])
    print(x[:, :, -5:])
    print(y[:, :, -5:])
    print(dataset.__len__())
    print(np.sum(x[:, 1, :].detach().numpy()))
    print(torch.sum(torch.randn(4, 21000)).item())
    """
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

    #dataset.plot_examples(ids=[16, 22, 26], n=3, logy=False, length_plot=168)

    from matplotlib import rc

    rc("text", usetex=True)

    mystyle = {
        "axes.spines.left": True,
        "axes.spines.right": False,
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.grid": False,
        "xtick.bottom": True,
        "ytick.left": True,
    }

    print("Electricity dataset test 4: ")
    dataset = ElectricityDataSet(
        "electricity/data/electricity.npy",
        include_time_covariates=False,
        start_date="2012-06-01",
        end_date="2014-12-18",
        predict_ahead=3,
        h_batch=256,
        one_hot_id=False,
        receptive_field=385,
        data_scale=False,
        cluster_covariate=False,
        random_covariate=False,
        representation="pca",
        similarity="euclidean",
        num_clusters=10,
        num_components=None,
        algorithm="KMeans",
    )
    with plt.style.context(mystyle):
        dataset.plot_examples(
            ids=[316],
            n=3,
            logy=False,
            length_plot=120,
            save_path="electricity_example_316.pdf",
        )
        dataset.plot_examples(
            ids=[16],
            n=3,
            logy=False,
            length_plot=120,
            save_path="electricity_example_16.pdf",
        )
        dataset.plot_examples(
            ids=[176],
            n=3,
            logy=False,
            length_plot=120,
            save_path="electricity_example_176.pdf",
        )

    data_loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    dataiter = iter(data_loader)
    x, y, idx, idx_row = dataiter.next()
    print(x.shape)
    print(y.shape)
    """
