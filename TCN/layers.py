# layers.py
"""
Layers used in the TCN.
* Causal Convolution
* Temporal block
"""
import sys

sys.path.append("")
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.tensor as Tensor

# Causal convolution
class DilatedCausalConv(nn.Conv1d):
    """ https://github.com/pytorch/pytorch/issues/1333 """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super(DilatedCausalConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        We add self.left_padding on the left side of the last dimension and 0 on the right.
        We add nothing on the second to last (first) dimension.
        """
        x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super(DilatedCausalConv, self).forward(x)


# Temporal block
class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        dropout: float = 0.5,
        leveledinit: bool = False,
    ) -> None:

        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dcc1 = weight_norm(
            DilatedCausalConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=True,
            )
        )
        self.drop1 = nn.Dropout(dropout)
        self.dcc2 = weight_norm(
            DilatedCausalConv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=True,
            )
        )
        self.drop2 = nn.Dropout(dropout)

        # If we don't have the same shape we have to make sure the residuals
        # get the same shape as out has gotten by going through the layers.
        if in_channels != out_channels:
            self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.res_conv = None

        # Init weights
        self.init_weights(leveledinit, kernel_size, bias)

    def init_weights(self, leveledinit: bool, kernel_size: int, bias: bool) -> None:
        """
        We refer to 
        https://github.com/rajatsen91/deepglo/blob/54e0644d764f1ead65d4203b72c8634e2f6ea25e/DeepGLO/LocalModel.py#L34
        """
        if leveledinit:
            nn.init.normal_(self.dcc1.weight, std=1e-3)
            nn.init.normal_(self.dcc2.weight, std=1e-3)
            nn.init.normal_(self.dcc1.bias, std=1e-6)
            nn.init.normal_(self.dcc2.bias, std=1e-6)
            self.dcc1.weight[:, 0, :] += 1.0 / kernel_size
            self.dcc2.weight += 1.0 / kernel_size
        else:
            nn.init.xavier_uniform_(self.dcc1.weight)
            nn.init.xavier_uniform_(self.dcc2.weight)

    def forward(self, x: Tensor) -> Tensor:
        """ 
        Refer to https://github.com/locuslab/TCN and their paper for the residual convolution. 
        """
        out = self.dcc1(x)
        out = F.relu(out)
        out = self.drop1(out)
        out = self.dcc2(out)
        out = F.relu(out)
        out = self.drop2(out)

        # If we don't have the same shape we have to make sure the residuals
        # get the same shape as out has gotten by going through the layers.
        if self.res_conv is None:
            residual = x
        else:
            residual = self.res_conv(x)

        out += residual
        return out


if __name__ == "__main__":
    causal_test = True
    block_test = True

    from adding_problem.data import AddTwoDataSet
    from torch.utils.data import DataLoader

    # Add two dataset
    print("Add Two dataset: ")
    dataset = AddTwoDataSet(N=1000, seq_length=64)
    loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    dataiter = iter(loader)
    samples, labels = dataiter.next()

    if causal_test:
        print("-----DilatedCausalConv test-----")
        cv1 = DilatedCausalConv(
            in_channels=2, out_channels=1, kernel_size=3, stride=1, dilation=1
        )
        print(
            f"Length of input  : {len(samples)}\nLength of output : {len(cv1(samples))}"
        )
        pytorch_total_params = sum(
            p.numel() for p in cv1.parameters() if p.requires_grad
        )
        print(f"Number of learnable parameters : {pytorch_total_params}")
        print(cv1.weight[:, 0])

    if block_test:
        print("-----ResidualBlock test-----")
        block = ResidualBlock(
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            stride=1,
            dilation=1,
            dropout=0.5,
        )
        print(
            f"Length of input  : {len(samples)}\nLength of output : {len(block(samples))}"
        )
        pytorch_total_params = sum(
            p.numel() for p in block.parameters() if p.requires_grad
        )
        print(f"Number of learnable parameters : {pytorch_total_params}")
        print(block.dcc1.weight)
