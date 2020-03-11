# model.py

import torch
import torch.nn as nn

import sys
sys.path.append('')
sys.path.append("../../")

from TCN.tcn import TemporalConvolutionalNetwork
from TCN.layers import DilatedCausalConv

from utils.metrics import MAPE, SMAPE, WAPE

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
        self.lookback = 1 + 2*(kernel_size-1)*2**(num_layers-1)
    
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

    def rolling_prediction(self, x, tau=24, num_windows=7):
        """
        Rolling prediction. Returns MAPE, SMAPE, WAPE on the predictions.
        x is the matrix of the test set. Needs the covariates.
        """
        #print('Rolling predictions')
        net_lookback = self.lookback
        predictions_list = []
        real_values = []
        # divide x into the rolling windows
        for i in range(num_windows):
            t_i = net_lookback + i*tau
            x_window = x[:,:,:t_i]
            x_cov_window = x[:,1:,t_i:(t_i+tau)]
            assert(x_cov_window.shape[2]==tau)
            # multi step prediction of that window
            _, preds = self.multi_step_prediction(x_window, x_cov_window, tau)
            predictions_list.append(preds)
            real_values.append(x[:,0,t_i:(t_i+tau)])
            assert(preds.shape == x[:,0,t_i:(t_i+tau)].shape)

        predictions = torch.cat(predictions_list, 1)
        real_values = torch.cat(real_values, 1)
        return predictions, real_values
 
    def multi_step_prediction(self, x, x_cov, num_steps):
        """ x_cov should be the covariates for the next num_steps """
        for i in range(num_steps):
            x_next = self.forward(x)[:,:,-1]
            # add covariates
            x_next = torch.cat((x_next, x_cov[:,:,i]),1)
            x_next = x_next.unsqueeze(2)
            # Add back onto x
            x = torch.cat((x, x_next), 2)
        # Return predicted x with covariates and just the predictions
        return x[:,:,-num_steps:], x[:,0,-num_steps:]

if __name__ == "__main__":
    import torch
    import sys
    sys.path.append('')
    from data import ElectricityDataSet
    from torch.utils.data import DataLoader

    # Add two dataset
    print("Electricity dataset: ")
    dataset = ElectricityDataSet(
        'electricity/data/LD2011_2014_hourly.txt', 
        start_date='2012-01-01',
        end_date='2013-01-01',
        h_batch=0,
        include_time_covariates=True)
    loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    dataiter = iter(loader)
    samples, labels = dataiter.next()

    tcn = TCN(
        num_layers=5,
        in_channels=8,
        out_channels=1,
        kernel_size=3,
        residual_blocks_channel_size=[16, 16, 16, 16, 16],
        bias=True,
        dropout=0.5,
        stride=1,
        leveledinit=False)

    pytorch_total_params = sum(
    p.numel() for p in tcn.parameters() if p.requires_grad)
    print(f"Number of learnable parameters : {pytorch_total_params}")

    for i, data in enumerate(loader):
        x, y = data[0], data[1]
        print(x.shape)
        print(y.shape)
        print(i)
        preds, real = tcn.rolling_prediction(x, tau=24, num_windows=7)
        print(preds.shape)
        print(real.shape)
        if i == 0:
            break

