# model.py
import torch.nn as nn
import torch.tensor as Tensor

import sys

sys.path.append("")
sys.path.append("../../")

from TCN.tcn import TemporalConvolutionalNetwork


class TCN(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        residual_blocks_channel_size: List[int],
        dilations: List[int] = None,
        kernel_size: int = 3,
        bias: bool = True,
        dropout: float = 0.5,
        stride: int = 1,
        leveledinit: bool = False,
    ) -> None:
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
        self.linear = nn.Linear(residual_blocks_channel_size[-1], out_channels)
        self.init_weights()

    def init_weights(self) -> None:
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x: Tensor) -> Tensor:
        out = self.tcn(x)
        out = self.linear(out[:, :, -1])
        return out
