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

# from metrics import MAPE, SMAPE, WAPE


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
    ) -> None:
        """
        A TCN for the electricity dataset. An additional layer is added to the TCN to get 
        the correct number of output channels. The residual_blocks_channel_size parameter
        does therefore not have to end with the out_channel size.
        """
        super(TCN, self).__init__()
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
        )
        self.conv1d = DilatedCausalConv(
            in_channels=residual_blocks_channel_size[-1],
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
        )
        self.init_weights(leveledinit, kernel_size, bias)
        self.lookback = 1 + 2 * (kernel_size - 1) * 2 ** (num_layers - 1)

    def init_weights(self, leveledinit: bool, kernel_size: int, bias: bool) -> None:
        if leveledinit:
            with torch.no_grad():
                self.conv1d.weight.copy_(torch.tensor(1.0 / kernel_size))
                if bias:
                    self.conv1d.bias.copy_(torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        out = self.tcn(x)
        out = self.conv1d(out)
        return out

    def rolling_prediction(
        self, x: Tensor, tau: int = 24, num_windows: int = 7
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
            _, preds = self.multi_step_prediction(x_prev_window, x_cov_curr_window, tau)
            predictions_list.append(preds)
            real_values.append(x[:, 0, t_i : (t_i + tau)])
            assert preds.shape == x[:, 0, t_i : (t_i + tau)].shape

        predictions = torch.cat(predictions_list, 1)
        real_values = torch.cat(real_values, 1)
        return predictions, real_values

    def multi_step_prediction(
        self, x_prev: Tensor, x_cov_curr: Tensor, num_steps: int
    ) -> List[Tensor]:
        """ x_cov should be the covariates for the next num_steps """
        for i in range(num_steps):
            x_next = self.forward(x_prev)[:, :, -1]
            # Add covariates
            x_next = torch.cat((x_next, x_cov_curr[:, :, i]), 1)
            x_next = x_next.unsqueeze(2)
            # Add back onto x
            x_prev = torch.cat((x_prev, x_next), 2)
        # Return predicted x with covariates and just the predictions
        return x_prev[:, :, -num_steps:], x_prev[:, 0, -num_steps:]


if __name__ == "__main__":
    from data import ElectricityDataSet
    from torch.utils.data import DataLoader

    # Electricity dataset
    print("Electricity dataset: ")
    np.random.seed(1729)
    dataset = ElectricityDataSet(
        "electricity_dglo_data/data/electricity.npy",
        include_time_covariates=True,
        start_date="2014-06-01",
        end_date="2014-12-18",
        predict_ahead=3,
        h_batch=0,
        one_hot_id=False,
    )
    data_loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    dataiter = iter(data_loader)
    x, y, idx = dataiter.next()

    tcn = TCN(
        num_layers=5,
        in_channels=8,
        out_channels=1,
        kernel_size=3,
        residual_blocks_channel_size=[16, 16, 16, 16, 16],
        bias=True,
        dropout=0.5,
        stride=1,
        leveledinit=False,
    )

    pytorch_total_params = sum(p.numel() for p in tcn.parameters() if p.requires_grad)
    print(f"Number of learnable parameters : {pytorch_total_params}")

    for i, data in enumerate(data_loader):
        x, y = data[0], data[1]
        print(x.shape)
        print(y.shape)
        print(i)
        preds, real = tcn.rolling_prediction(x, tau=24, num_windows=7)
        print(preds.shape)
        print(real.shape)
        if i == 0:
            break
