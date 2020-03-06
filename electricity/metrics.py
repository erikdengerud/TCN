# metrics.py


def WAPE(Y, Y_hat):
    """ Weighted Absolute Percent Error """
    numerator = abs(Y-Y_hat).sum()
    denominator = abs(Y).sum()
    return (numerator / denominator).item()

def MAPE(Y, Y_hat):
    nz = np.where(abs(Y)>0)
    Y = Y[nz]
    Y_hat = Y_hat[nz]
    numerator = abs(Y-Y_hat)
    denominator = abs(Y)
    divided = torch.div(numerator, denominator)
    tot = divided.sum()
    z = Y.nonzero().shape[0]
    return tot.div(z).item()

def SMAPE(Y, Y_hat):
    nz = np.where(abs(Y)>0)
    Y = Y[nz]
    Y_hat = Y_hat[nz]
    numerator = 2*abs(Y-Y_hat)
    denominator = abs(Y+Y_hat)
    divided = torch.div(numerator, denominator)
    tot = divided.sum()
    z = Y.nonzero().shape[0]
    return tot.div(z).item()


if __name__ == "__main__":
    import torch
    import numpy as np
    torch.manual_seed(1729)

    Y = torch.rand(370, 1, 500)
    Y_hat = torch.rand(370, 1, 500)
    print("*"*10 + "WAPE" + "*"*10)
    print(WAPE(Y, Y_hat))
    print("*"*10 + "MAPE" + "*"*10)
    print(MAPE(Y, Y_hat))
    print("*"*10 + "SMAPE" + "*"*10)
    print(SMAPE(Y, Y_hat))