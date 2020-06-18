# metrics.py
import numpy as np
import torch


def WAPE(Y, Y_hat):
    """ Weighted Absolute Percent Error """
    Y = Y.numpy()
    Y_hat = Y_hat.numpy()
    return np.mean(np.abs(Y - Y_hat)) / np.mean(np.abs(Y))


def MAPE(Y, Y_hat):
    Y = Y.numpy()
    Y_hat = Y_hat.numpy()
    nz = np.where(Y > 0)
    Pz = Y_hat[nz]
    Az = Y[nz]

    return np.mean(np.abs(Az - Pz) / np.abs(Az))


def SMAPE(Y, Y_hat):
    Y = Y.numpy()
    Y_hat = Y_hat.numpy()
    nz = np.where(Y > 0)
    Pz = Y_hat[nz]
    Az = Y[nz]

    return np.mean(2 * np.abs(Az - Pz) / (np.abs(Az) + np.abs(Pz)))


def MAE(Y, Y_hat):
    Y = Y.numpy()
    Y_hat = Y_hat.numpy()
    return np.abs(Y_hat - Y).mean()


def RMSE(Y, Y_hat):
    Y = Y.numpy()
    Y_hat = Y_hat.numpy()
    return np.sqrt(((Y_hat - Y) ** 2).mean())
    # real_values_tensor, predictions_tensor


if __name__ == "__main__":
    import torch
    import numpy as np

    torch.manual_seed(1729)

    Y = torch.rand(370, 1, 500)
    Y_hat = torch.rand(370, 1, 500)
    print("*" * 10 + "WAPE" + "*" * 10)
    print(WAPE(Y, Y_hat))
    print("*" * 10 + "MAPE" + "*" * 10)
    print(MAPE(Y, Y_hat))
    print("*" * 10 + "SMAPE" + "*" * 10)
    print(SMAPE(Y, Y_hat))
