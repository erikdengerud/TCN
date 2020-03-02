# layers.py
"""
Layers used in the TCN.
* Causal Convolution
* Temporal block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# Causal convolution
class DilatedCausalConv(nn.Conv1d):
    """ https://github.com/pytorch/pytorch/issues/1333 """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(DilatedCausalConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super(DilatedCausalConv, self).forward(x)

# Temporal block
class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        dropout=0.5):

        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dcc1 = weight_norm(DilatedCausalConv(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=True))
        self.drop1 = nn.Dropout(dropout)
        self.dcc2 = weight_norm(DilatedCausalConv(
            in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=True))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.dcc1(x)
        out = F.relu(out)
        out = self.drop1(out)
        out = self.dcc2(out)
        out = F.relu(out)
        out = self.drop2(out)
        # If we don't have the same shape we have to make sure the residuals
        # get the same shape as out has gotten by going through the layers.
        if out.shape != residual.shape: 
            residual = nn.Conv1d(
                in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)(residual)
        out += residual
        return out

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
        stride=1
    ):
        super(TCN, self).__init__()

        if dilations is None:
            dilations = [2**i for i in range(num_layers)]

        assert(num_layers == len(dilations))
        assert(num_layers == len(residual_blocks_channel_size))
        assert(out_channels == residual_blocks_channel_size[-1])
        assert(dropout <= 1 and dropout >= 0)
        assert(type(stride) is int and stride > 0)

        self.res_blocks = []
        # Initial convolution to get correct num channels
        init_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=residual_blocks_channel_size[0],
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilations[0],
            bias=bias,
            dropout=dropout)
        self.res_blocks.append(init_block)

        for i in range(1, num_layers):
            block = ResidualBlock(
                in_channels=residual_blocks_channel_size[i-1],
                out_channels=residual_blocks_channel_size[i],
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilations[i],
                bias=bias,
                dropout=dropout)
            self.res_blocks.append(block)

    def forward(self, x):
        for block in self.res_blocks:
            x = block(x)
        return x



if __name__ == "__main__":
    causal_test = False
    block_test = False
    tcn_test = True

    from data import AddTwoDataSet
    from torch.utils.data import DataLoader
    # Add two dataset
    print("Add Two dataset: ")
    dataset = AddTwoDataSet(N=1000, seq_length=64)
    loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    dataiter = iter(loader)
    samples, labels = dataiter.next()

    if causal_test:
        cv1 = DilatedCausalConv(
            in_channels=2, out_channels=1, kernel_size=3, stride=1, dilation=1)
        print(cv1.forward(samples))
    
    if block_test:
        block = ResidualBlock(in_channels=2, out_channels=2, kernel_size=3, stride=1, dilation=1, dropout=0.5)
        print(block.forward(samples))

    if tcn_test:
        tcn = TCN(
            num_layers=3,
            in_channels=2,
            out_channels=1,
            kernel_size=3,
            residual_blocks_channel_size=[16, 16, 1],
            bias=True,
            dropout=0.5,
            stride=1)
        print(tcn.forward(samples))
    

