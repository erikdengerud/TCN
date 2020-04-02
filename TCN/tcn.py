# tcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.tensor as Tensor

import sys
from typing import List

sys.path.append("")
sys.path.append("../../")

from TCN.layers import ResidualBlock, ResidualBlockChomp


class TemporalConvolutionalNetwork(nn.Module):
    """
    A TCN consisting of Residual blocks. 
    """

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
        type_res_blocks: str = "deepglo",
    ) -> None:

        super(TemporalConvolutionalNetwork, self).__init__()

        if dilations is None:
            dilations = [2 ** i for i in range(num_layers)]

        assert num_layers == len(dilations)
        assert num_layers == len(residual_blocks_channel_size)
        assert dropout <= 1 and dropout >= 0
        assert stride > 0

        res_blocks = []
        # Initial convolution to get correct num in channels
        if type_res_blocks == "deepglo":
            first_block = ResidualBlockChomp(
                in_channels=in_channels,
                out_channels=residual_blocks_channel_size[0],
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * dilations[0],
                stride=stride,
                dilation=dilations[0],
                bias=bias,
                dropout=dropout,
                leveledinit=leveledinit,
            )
            res_blocks += [first_block]

            for i in range(1, num_layers):
                block = ResidualBlockChomp(
                    in_channels=residual_blocks_channel_size[i - 1],
                    out_channels=residual_blocks_channel_size[i],
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) * dilations[i],
                    stride=stride,
                    dilation=dilations[i],
                    bias=bias,
                    dropout=dropout,
                    leveledinit=leveledinit,
                )
                res_blocks += [block]

        else:
            first_block = ResidualBlock(
                in_channels=in_channels,
                out_channels=residual_blocks_channel_size[0],
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilations[0],
                bias=bias,
                dropout=dropout,
                leveledinit=leveledinit,
            )
            res_blocks += [first_block]

            for i in range(1, num_layers):
                block = ResidualBlock(
                    in_channels=residual_blocks_channel_size[i - 1],
                    out_channels=residual_blocks_channel_size[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilations[i],
                    bias=bias,
                    dropout=dropout,
                    leveledinit=leveledinit,
                )
                res_blocks += [block]
        self.net = nn.Sequential(*res_blocks)

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        return out


if __name__ == "__main__":
    import sys

    sys.path.append("")
    from adding_problem.data import AddTwoDataSet
    from torch.utils.data import DataLoader

    # Add two dataset
    print("Add Two dataset: ")
    dataset = AddTwoDataSet(N=1000, seq_length=64)
    loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    dataiter = iter(loader)
    samples, labels = dataiter.next()

    tcn = TemporalConvolutionalNetwork(
        num_layers=3,
        in_channels=2,
        out_channels=1,
        kernel_size=3,
        residual_blocks_channel_size=[16, 16, 1],
        bias=True,
        dropout=0.5,
        stride=1,
        leveledinit=False,
        type_res_blocks="deepglo",
    )

    # print(tcn(samples))
    pytorch_total_params = sum(p.numel() for p in tcn.parameters() if p.requires_grad)
    print(f"Number of learnable parameters : {pytorch_total_params}")
