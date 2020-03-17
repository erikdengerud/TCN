# metrics.py
import numpy as np
import torch

def WAPE(Y, Y_hat):
    """ Weighted Absolute Percent Error """
    Y = Y.numpy()
    Y_hat = Y_hat.numpy()
    return np.mean(np.abs(Y_hat - Y)) / np.mean(np.abs(Y_hat))

def MAPE(Y, Y_hat):
    Y = Y.numpy()
    Y_hat = Y_hat.numpy()
    nz = np.where(Y_hat > 0)
    Pz = Y[nz]
    Az = Y_hat[nz]

    return np.mean(np.abs(Az - Pz) / np.abs(Az))

def SMAPE(Y, Y_hat):
    Y = Y.numpy()
    Y_hat = Y_hat.numpy()
    nz = np.where(Y_hat > 0)
    Pz = Y[nz]
    Az = Y_hat[nz]

    return np.mean(2 * np.abs(Az - Pz) / (np.abs(Az) + np.abs(Pz)))


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