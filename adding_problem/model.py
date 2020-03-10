# model.py

import torch.nn as nn

import sys

sys.path.append('')
sys.path.append("../../")

from TCN.tcn import TemporalConvolutionalNetwork


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
        self.linear = nn.Linear(residual_blocks_channel_size[-1], out_channels)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        out = self.tcn(x)
        out = self.linear(out[:,:,-1])
        return out

