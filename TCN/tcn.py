# tcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import sys
sys.path.append('')
sys.path.append('../../')
from TCN.layers import ResidualBlock

class TemporalConvolutionalNetwork(nn.Module):
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

        super(TemporalConvolutionalNetwork, self).__init__()

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
        #self.lookback = 1 + 2*(kernel_size-1)*2**(num_layers-1)

    def forward(self, x):
        out = self.net(x)
        return out

    '''
    def rolling_prediction(self, x, tau=24, num_windows=7):
        """
        Rolling prediction. Returns MAPE, SMAPE, WAPE on the predictions.
        x is the matrix of the test set. Needs the covariates.
        """
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

        # calculate metrics
        real_values = real_values.cpu()
        predictions = predictions.cpu()
        mape = MAPE(real_values, predictions)
        smape = SMAPE(real_values, predictions)
        wape = WAPE(real_values, predictions)
        return mape, smape, wape

    def multi_step_prediction(self, x, x_cov, num_steps):
        """ x_cov should be the covariates for the next num_steps """
        for i in range(num_steps):
            x_next = self.forward(x)[:,:,-1].view(-1,1,1)
            # add covariates
            x_next = torch.cat((x_next, x_cov[:,:,i].view(-1,1,1)), 1)
            x = torch.cat((x, x_next), 2)
        # Return predicted x with covariates and just the predictions
        return x[:,:,-num_steps:], x[:,0,-num_steps:]
    '''

if __name__ == "__main__":
    import sys
    sys.path.append('')
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
        leveledinit=False)

    #print(tcn(samples))
    pytorch_total_params = sum(
    p.numel() for p in tcn.parameters() if p.requires_grad)
    print(f"Number of learnable parameters : {pytorch_total_params}")


    '''
    # Multi step predictions
    cov = torch.rand(4,1,1000)
    x, preds = tcn.multi_step_prediction(samples, cov, 10)
    #print(x)
    print(preds)
    print(x.shape)
    print(preds.shape)
    
    test = torch.rand(4, 2, 1000)
    print(tcn.rolling_prediction(test, tau=24, num_windows=7))
    '''


    

