# model.py
import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import Tuple, List

sys.path.append("")
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.tensor as Tensor

from TCN.tcn import TemporalConvolutionalNetwork
from TCN.layers import DilatedCausalConv

from utils.metrics import MAPE, SMAPE, WAPE


class TCN(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        residual_blocks_channel_size: List[int],
        dilations: None = None,
        kernel_size: int = 3,
        bias: bool = True,
        dropout: float = 0.5,
        stride: int = 1,
        leveledinit: bool = False,
        type_res_blocks: str = "erik",
        num_embeddings: int = 370,
        embedding_dim: int = 3,
        embed: str = None,
        clustering_dict: dict = None,
    ) -> None:
        """
        A TCN for the electricity dataset. An additional layer is added to the TCN to get 
        the correct number of output channels. The residual_blocks_channel_size parameter
        does therefore not have to end with the out_channel size.
        """
        super(TCN, self).__init__()
        in_channels = in_channels + embedding_dim if embed == "pre" else in_channels
        self.tcn = TemporalConvolutionalNetwork(
            num_layers,
            in_channels,
            out_channels,
            residual_blocks_channel_size,
            dilations,
            kernel_size,
            bias,
            dropout,
            stride,
            leveledinit,
            type_res_blocks,
        )
        self.conv1d = DilatedCausalConv(
            in_channels=residual_blocks_channel_size[-1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
        )

        """ Embeddings """
        self.embed = embed
        if embed == "pre":  # this doesn't converge
            self.embedding = nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=embedding_dim
            )
        elif embed == "post":
            self.embedding = nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=embedding_dim
            )
            # We add a layer after we concat the embeddings to the output to
            # get some learning of the embeddings.
            self.conv1demb = DilatedCausalConv(
                in_channels=residual_blocks_channel_size[-1] + embedding_dim,
                out_channels=residual_blocks_channel_size[-1],
                kernel_size=kernel_size,
                bias=bias,
            )
        self.clustering_dict = clustering_dict

        self.init_weights(leveledinit, kernel_size, bias)
        self.lookback = 1 + 2 * (kernel_size - 1) * 2 ** (num_layers - 1)

    def init_weights(self, leveledinit: bool, kernel_size: int, bias: bool) -> None:
        """ 
        Init the weights in the last layer. The rest is initialized in the residual block. 
        From the DeepGLO paper.
        """
        if leveledinit:
            nn.init.normal_(self.conv1d.weight, std=1e-3)
            nn.init.normal_(self.conv1d.bias, std=1e-6)
            with torch.no_grad():
                self.conv1d.weight[:, 0, :] += 1.0 / kernel_size
        else:
            nn.init.xavier_uniform_(self.conv1d.weight)

        if self.embed in ("pre", "post"):
            nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x: Tensor, emb_id: Tensor) -> Tensor:
        if self.embed == "pre":
            emb = self.embedding(emb_id)
            emb = torch.unsqueeze(emb, 2)
            emb = emb.repeat(1, 1, x.shape[2])
            x = torch.cat((x, emb), 1)

        out = self.tcn(x)

        if self.embed == "post":
            emb = self.embedding(emb_id)
            emb = torch.unsqueeze(emb, 2)
            emb = emb.repeat(1, 1, x.shape[2])
            out = torch.cat((out, emb), 1)
            out = self.conv1demb(out)

        out = self.conv1d(out)  # to get right dimensions
        return out

    def rolling_prediction(
        self, x: Tensor, emb_id: Tensor, tau: int = 24, num_windows: int = 7
    ) -> List[Tensor]:
        """
        Rolling prediction. Returns MAPE, SMAPE, WAPE on the predictions.
        x is the matrix of the test set. Needs the covariates.
        """
        net_lookback = self.lookback
        predictions_list = []
        real_values = []
        # Divide x into the rolling windows
        for i in range(num_windows):
            t_i = net_lookback + i * tau
            x_prev_window = x[:, :, :t_i]
            x_cov_curr_window = x[:, 1:, t_i : (t_i + tau)]
            assert x_cov_curr_window.shape[2] == tau
            # Multi step prediction of current window
            _, preds = self.multi_step_prediction(
                x_prev_window, x_cov_curr_window, tau, emb_id
            )
            predictions_list.append(preds)
            real_values.append(x[:, 0, t_i : (t_i + tau)])
            assert preds.shape == x[:, 0, t_i : (t_i + tau)].shape

        predictions = torch.cat(predictions_list, 1)
        real_values = torch.cat(real_values, 1)
        return predictions, real_values

    def multi_step_prediction(
        self, x_prev: Tensor, x_cov_curr: Tensor, num_steps: int, emb_id: Tensor
    ) -> List[Tensor]:
        """ x_cov should be the covariates for the next num_steps """
        for i in range(num_steps):
            x_next = self.forward(x_prev, emb_id)[:, :, -1]
            # Add covariates
            if self.clustering_dict is None:
                x_next = torch.cat((x_next, x_cov_curr[:, :, i]), 1)
                x_next = x_next.unsqueeze(2)
            else:  # calculate prototypes
                prototypes_next = self.calculate_prototypes(x_next, emb_id)
                x_next = torch.cat((x_next, prototypes_next), 1)
                x_next = x_next.unsqueeze(2)

            # Add back onto x
            x_prev = torch.cat((x_prev, x_next), 2)
        # Return predicted x with covariates and just the predictions
        return x_prev[:, :, -num_steps:], x_prev[:, 0, -num_steps:]

    def calculate_prototypes(self, x_next, emb_id):
        """ Calculates the prototypes values at the time step """
        x_next_numpy = x_next.detach().numpy()
        emb_id_numpy = emb_id.detach().numpy()
        x_next_cov = np.zeros(x_next.shape[0])
        for c in set(self.clustering_dict.values()):
            # p = np.where()
            p = np.mean(
                x_next_numpy[[self.clustering_dict[emb] == c for emb in emb_id_numpy]]
            )
            x_next_cov[[self.clustering_dict[emb] == c for emb in emb_id_numpy]] = p

        prototypes_next = torch.from_numpy(x_next_cov).to(dtype=torch.float32)
        prototypes_next = prototypes_next.unsqueeze(1)

        return prototypes_next  # n_samples x 1


if __name__ == "__main__":
    from data import ElectricityDataSet
    from torch.utils.data import DataLoader

    # Electricity dataset
    print("Electricity dataset: ")
    np.random.seed(1729)
    dataset = ElectricityDataSet(
        "electricity/data/electricity.npy",
        include_time_covariates=False,
        start_date="2014-06-01",
        end_date="2014-12-18",
        scale=True,
        scaler=None,
        predict_ahead=1,
        h_batch=0,  # 0 gives the whole time series
        one_hot_id=False,
        receptive_field=385,
        cluster_covariate=True,
        random_covariate=False,
        representation="pca",
        similarity="euclidean",
        num_clusters=10,
        num_components=None,
        algorithm="KMeans",
    )
    data_loader = DataLoader(
        dataset, batch_size=len(dataset), num_workers=0, shuffle=True
    )
    dataiter = iter(data_loader)
    x, y, idx, idx_row = dataiter.next()

    tcn = TCN(
        num_layers=5,
        in_channels=2,
        out_channels=1,
        kernel_size=3,
        residual_blocks_channel_size=[16, 16, 16, 16, 16],
        bias=True,
        dropout=0.5,
        stride=1,
        leveledinit=False,
        embed=None,
        clustering_dict=dataset.cluster_dict,
    )

    pytorch_total_params = sum(p.numel() for p in tcn.parameters() if p.requires_grad)
    print(f"Number of learnable parameters : {pytorch_total_params}")

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            x, y, idx = data[0], data[1], data[2]
            print(x.shape)
            print(y.shape)
            print("i=", i)
            preds, real = tcn.rolling_prediction(x, tau=24, num_windows=7, emb_id=idx)
            print(preds.shape)
            print(real.shape)
            if i == 0:
                break
