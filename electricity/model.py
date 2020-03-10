# model.py

import torch.nn as nn

import sys
sys.path.append('')
sys.path.append("../../")

from TCN.tcn import TemporalConvolutionalNetwork
from TCN.layers import DilatedCausalConv

class TCN(nn.Module):
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
            leveledinit
        )
        self.conv1d = DilatedCausalConv(
            in_channels=residual_blocks_channel_size[-1], 
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias)
        self.init_weights(leveledinit, kernel_size, bias)
    
    def init_weights(self, leveledinit, kernel_size, bias):
        if leveledinit:
            with torch.no_grad():
                self.conv1d.weight.copy_(torch.tensor(1.0/kernel_size))
                if bias:
                    self.conv1d.bias.copy_(torch.tensor(0.0))

    def forward(self, x):
        out = self.tcn(x)
        out = self.conv1d(out)
        return out

