# tcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from layers import ResidualBlock

class TCN(nn.Module):
    """
    num_layers                      : int
    dilations                       : list of dilations for each layer
    in_channels                     : num channels into the network
    out_channels                    : num channels out of the network
    residual_blocks_channel_size    : list of channel sizes for each block
    bias                            : bias
    dropout                         : dropout in the dropout layers 
                                      of the residual block
    stride                          : stride of the filters
    """
    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        residual_blocks_channel_size,
        dilations=None,
        kernel_size=3,
        bias=True,
        dropout=0.5,
        stride=1,
        leveledinit=False):

        super(TCN, self).__init__()

        if dilations is None:
            dilations = [2**i for i in range(num_layers)]

        assert(num_layers == len(dilations))
        assert(num_layers == len(residual_blocks_channel_size))
        assert(dropout <= 1 and dropout >= 0)
        assert(type(stride) is int and stride > 0)

        res_blocks = []
        # Initial convolution to get correct num channels
        init_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=residual_blocks_channel_size[0],
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilations[0],
            bias=bias,
            dropout=dropout,
            leveledinit=leveledinit)
        res_blocks += [init_block]

        for i in range(1, num_layers):
            block = ResidualBlock(
                in_channels=residual_blocks_channel_size[i-1],
                out_channels=residual_blocks_channel_size[i],
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilations[i],
                bias=bias,
                dropout=dropout,
                leveledinit=leveledinit)
            res_blocks += [block]
        self.net = nn.Sequential(*res_blocks)
        self.linear = nn.Linear(residual_blocks_channel_size[-1], out_channels)
        self.conv1d = nn.Conv1d(
            in_channels=residual_blocks_channel_size[-1],
            out_channels=out_channels, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        #out = self.linear(out[:, :, -1])
        #self.linear(out)
        out = self.conv1d(out)
        return out



if __name__ == "__main__":

    from adding_problem.data import AddTwoDataSet
    from torch.utils.data import DataLoader
    # Add two dataset
    print("Add Two dataset: ")
    dataset = AddTwoDataSet(N=1000, seq_length=64)
    loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    dataiter = iter(loader)
    samples, labels = dataiter.next()

    tcn = TCN(
        num_layers=3,
        in_channels=2,
        out_channels=1,
        kernel_size=3,
        residual_blocks_channel_size=[16, 16, 1],
        bias=True,
        dropout=0.5,
        stride=1,
        leveledinit=False)
    print(tcn(samples))
    pytorch_total_params = sum(
    p.numel() for p in tcn.parameters() if p.requires_grad)
    print(f"Number of learnable parameters : {pytorch_total_params}")
    

