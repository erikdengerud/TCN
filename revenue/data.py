# data.py

from datetime import date, timedelta

# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

sys.path.append("./")
sys.path.append("../../")
from typing import Tuple, List, Dict

import torch
import torch.tensor as Tensor
from torch.utils.data import Dataset, DataLoader

# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import StandardScaler, RobustScaler


class RevenueDataset(Dataset):
    """
    Dataset on total quarterly revenue.
    """

    def __init__(
        self,
        file_path: str,
        meta_path: str,
        data_scale: bool = True,
        data_scaler=None,
        start_date: str = "2007-01-01",  # YYYY-MM-DD
        end_date: str = "2017-01-01",  # YYYY-MM-DD
        predict_ahead: int = 1,
        h_batch: int = 0,  # 0 gives the whole time series
        receptive_field: int = 20,
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
        assert self.start_date >= date.fromisoformat("2007-01-01")
        assert self.end_date <= date.fromisoformat("2020-01-01")

        """ Input parameters """
        self.predict_ahead = predict_ahead
        self.h_batch = h_batch
        self.receptive_field = receptive_field
        assert cluster_covariate + random_covariate + zero_covariate <= 1
        self.cluster_covariate = cluster_covariate
        self.zero_covariate = zero_covariate
        self.random_covariate = random_covariate
        self.data_scale = data_scale
        if data_scaler is None:
            self.data_scaler = StandardScaler()
        else:
            self.data_scaler = data_scaler

        """ Creating the dataset """
        # Extract the time series from the file and store as tensor X
        df, dates = self.get_df(file_path, start_date=start_date, end_date=end_date)
        self.companies = df.columns
        if self.data_scale:
            try:
                values = self.data_scaler.transform(df.values)
            except:
                values = self.data_scaler.fit_transform(df.values)
        else:
            values = df.values
        X = torch.tensor(values)
        X = torch.transpose(X, 0, 1)
        self.dates = dates
        self.id_companies_dict = {i: df.columns[i] for i in range(len(df.columns))}
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

        self.comp_sect_dict, self.num_sect, self.sect_id_dict = self.get_meta(meta_path)

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
        One hot encoded IDs are also added here.
        """
        # We get a random index in the range [0, int(num_ts * lenght_ts) / h_batch)].
        # Then we create the sample for that index. The sample is the next h_batch and
        # the previous receptive_field. The Y is only the values of length h_batch after
        # the index. This means that the length of X and Y differ.
        if self.h_batch == 0:
            X = self.X[idx]
            Y = self.Y[idx]

            # Using only one of the sectors if the company belongs to more
            sect = self.sect_id_dict[
                self.comp_sect_dict[self.id_companies_dict[idx]][0]
            ]

            if self.cluster_covariate:
                c = self.cluster_dict[idx]
                prototype = self.prototype[c]
                prototype = torch.from_numpy(prototype)
                prototype.view(1, -1).to(dtype=torch.float32)
                X = torch.cat((X, prototype), 0)

            if self.zero_covariate:
                zero = torch.from_numpy(np.zeros(X.shape[1]))
                zero = zero.view(1, -1).to(dtype=torch.float32)
                X = torch.cat((X, zero), 0)

            if self.random_covariate:
                r = torch.randn(X.shape[1], dtype=torch.float32)
                r = r.view(1, -1)
                X = torch.cat((X, r), 0)

            return X, Y, idx, idx, sect

        else:
            # print("before")
            row, column = self.get_row_column(idx)
            if column < self.receptive_field:
                column = self.receptive_field
            if column + self.h_batch > self.length_ts:
                column = self.length_ts - self.h_batch
            if column == self.length_ts:
                column -= 1

            X = self.X[row, :, column - self.receptive_field : column + self.h_batch]
            Y = self.Y[row, :, column : column + self.h_batch]

            sect = self.sect_id_dict[
                self.comp_sect_dict[self.id_companies_dict[row]][0]
            ]

            if self.cluster_covariate:
                c = self.cluster_dict[row]
                prototype = self.prototypes[c]
                prototype = torch.from_numpy(prototype)
                prototype = prototype[
                    column - self.receptive_field : column + self.h_batch
                ]
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

            return X, Y, idx, row, sect

    def prototypes_from_cluster_dict(self, X, cluster_dict):
        """ 
        Calculates prototypes given X and a cluster dict.
        Uses the scaler to scale before taking the mean
        """
        prototypes = {}
        X_scaled = self.data_scaler.transform(X.squeeze().detach().numpy().T).T
        for c in set(cluster_dict.values()):
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

    def get_df(self, file_path: str, start_date: str, end_date: str) -> List[pd.Series]:
        """ 
        Reads the file with the time series and crops based on the start and end date 
        of the datasset. 
        """
        df = pd.read_csv(file_path, index_col=0)
        df = df.fillna(0)
        # Cut off at start and end date
        df = df.loc[str(start_date) : str(end_date)]
        dates = df.index
        return df, dates

    def get_meta(self, meta_path) -> Dict:
        df = pd.read_csv(meta_path, index_col=0)
        sectors = df["sector"].to_numpy()
        comp_sect_dict = {}
        for i, comp in enumerate(df["company"].to_numpy()):
            if comp not in comp_sect_dict.keys():
                comp_sect_dict[comp] = [sectors[i]]
            else:
                comp_sect_dict[comp].append(sectors[i])
        # d = df.set_index("company").T.to_dict("records")
        unique_sectors = list(set(sectors))
        sect_id_dict = {s: i for i, s in enumerate(unique_sectors)}
        return comp_sect_dict, len(set(sectors)), sect_id_dict

    def plot_examples(
        self,
        ids: List = [],
        n: int = 3,
        length_plot: int = 16,
        save_path: str = "revenue/figures/ts_examples.pdf",
    ) -> None:
        if ids:
            time_series = []
            start_point = np.random.randint(
                0, max(int(self.length_ts - length_plot), 1)
            )
            for i in ids:
                s = self.X[i, 0, start_point : start_point + length_plot].numpy()
                time_series.append(np.transpose(s))
            examples_ids = ids
        else:
            # Choose n randomly selected series and a random start point
            examples_ids = np.random.choice(self.num_ts, size=n, replace=False)
            start_point = np.random.randint(
                0, max(int(self.length_ts - length_plot), 1)
            )
            time_series = []
            for example_id in examples_ids:
                s = self.X[
                    example_id,
                    0,
                    start_point : start_point + min(length_plot, self.length_ts),
                ].numpy()
                time_series.append(np.transpose(s))

        df = pd.DataFrame(time_series).T

        # Get datetime start to use on the x axis
        date_list = self.dates.tolist()
        df.index = pd.to_datetime(date_list[start_point : start_point + length_plot])
        titles = [self.companies[i] for i in examples_ids]
        titles = [title.replace("&", "\&") for title in titles]

        size = 2 * 4.77
        ax = df.plot(
            subplots=True,
            figsize=(size, 1.5),
            # title=titles,
            legend=False,
            linewidth=1,
            rot=0,
            color="black",
        )
        for i in range(ax.shape[0]):
            x1, x2 = ax[i].get_xlim()
            ax[i].set_xlim((x1, x2 + 1))
        # plt.legend()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    # revenue dataset

    print("Revenue dataset")
    dataset = RevenueDataset(
        file_path="revenue/data/processed_companies.csv",
        meta_path="revenue/data/comp_sect_meta.csv",
        data_scale=True,
        start_date="2007-01-01",
        end_date="2017-01-01",
        cluster_covariate=False,
        random_covariate=False,
        zero_covariate=False,
    )
    data_loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=False)
    dataiter = iter(data_loader)
    x, y, idx, idx_row, id_sect = dataiter.next()
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
    print(torch.sum(torch.randn(4, 40)).item())
    """
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

    with plt.style.context(mystyle):
        dataset.plot_examples(
            ids=[316], save_path="revenue_example_316.pdf", length_plot=16,
        )
        dataset.plot_examples(
            ids=[16], save_path="revenue_example_16.pdf", length_plot=16,
        )
        dataset.plot_examples(
            ids=[176], save_path="revenue_example_176.pdf", length_plot=16,
        )

    print("Revenue dataset.")
    dataset = RevenueDataset(
        file_path="revenue/data/processed_companies.csv",
        meta_path="revenue/data/comp_sect_meta.csv",
        start_date="2007-01-01",
        end_date="2017-01-01",
        predict_ahead=1,
        h_batch=0,
        receptive_field=20,
    )

    print(dataset.length_ts)
    dataset.plot_examples(ids=[], n=5, length_plot=40)

    data_loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    dataiter = iter(data_loader)
    x, y, idx, row, sect = dataiter.next()
    data = dataiter.next()
    print(type(data))
    # print(data)
    print("idx", idx)
    print("sect: ", sect)
    d = dataset.sect_id_dict
    print(d)
    d = {v: k for k, v in d.items()}
    print([d[s] for s in sect.detach().numpy().tolist()])
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

    # print(dataset.companies_id_dict)
    print(dataset.num_sect)
    print(dataset.sect_id_dict)
    """
