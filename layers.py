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

        self.dcc1 = weight_norm(DilatedCausalConv(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=True))
        self.drop1 = nn.Dropout(dropout)
        self.dcc2 = weight_norm(DilatedCausalConv(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=True))
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
                in_channels=self.in_channels, out_channels=self.out_channel, kernel_size=1)(residual)
        out += residual
        return out

class TCN(nn.Module):
    def __init__(
        self,
    ):
        super(TCN, self).__init__()



    def forward(self, x):
        pass



if __name__ == "__main__":
    causal_test = False
    block_test = True

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
    

