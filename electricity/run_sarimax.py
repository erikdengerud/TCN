import sys

sys.path.append("")
sys.path.append("../../")
import numpy as np
import pandas as pd
import torch
from data import ElectricityDataSet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.optimize import brute
import matplotlib.pyplot as plt
import warnings
import time
import datetime
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


def WAPE(Y, Y_hat):
    return np.mean(np.abs(Y - Y_hat)) / np.mean(np.abs(Y))


def MAPE(Y, Y_hat):
    nz = np.where(Y > 0)
    Pz = Y_hat[nz]
    Az = Y[nz]
    return np.mean(np.abs(Az - Pz) / np.abs(Az))


def SMAPE(Y, Y_hat):
    nz = np.where(Y > 0)
    Pz = Y_hat[nz]
    Az = Y[nz]
    return np.mean(2 * np.abs(Az - Pz) / (np.abs(Az) + np.abs(Pz)))


def MAE(Y, Y_hat):
    return np.abs(Y_hat - Y).mean()


def RMSE(Y, Y_hat):
    return np.sqrt(((Y_hat - Y) ** 2).mean())


def objfunc(order, endog, exog):
    print(order)
    pdq = order[:3]
    PDQ = order[3:]
    PDQs = np.append(PDQ, 24)
    fit = SARIMAX(endog=endog, exog=exog, order=pdq, seasonal_order=PDQs).fit(
        disp=False, maxiter=20
    )
    return fit.aic


if __name__ == "__main__":
    t0 = time.time()
    print("Dataset")
    ds = ElectricityDataSet(
        file_path="electricity/data/electricity.npy",
        data_scale=False,
        data_scaler=None,
        start_date="2014-01-01",  # yyyy-mm-dd
        end_date="2014-12-16",  # yyyy-mm-dd
        cluster_dict_path="prototypes/cluster_dicts/electricity_pca_scaled_nc_10_euclidean_Spectral_clustering_nc_10.pkl",
        prototypes_file_path="prototypes/prototypes_matrices/electricity_pca_scaled__nc_10_euclidean_Spectral_clustering_nc_10.npy",
        cluster_covariate=True,
    )
    ds_test = ElectricityDataSet(
        file_path="electricity/data/electricity.npy",
        data_scale=False,
        data_scaler=None,
        start_date="2014-12-17",  # yyyy-mm-dd
        end_date="2014-12-23",  # yyyy-mm-dd
        cluster_dict_path="prototypes/cluster_dicts/electricity_pca_scaled_nc_10_euclidean_Spectral_clustering_nc_10.pkl",
        prototypes_file_path="prototypes/prototypes_matrices/electricity_pca_scaled__nc_10_euclidean_Spectral_clustering_nc_10.npy",
        cluster_covariate=True,
    )

    print("Fitting SARIMA models:")
    grid = (
        slice(1, 2, 1),
        slice(1, 2, 1),
        slice(1, 2, 1),
        slice(0, 2, 1),
        slice(1, 2, 1),
        slice(0, 2, 1),
    )
    """
    grid = (
        slice(1, 4, 1),
        slice(1, 3, 1),
        slice(1, 4, 1),
        slice(0, 2, 1),
        slice(1, 2, 1),
        slice(0, 2, 1),
    )
    """

    # predictions =
    dl_train = DataLoader(dataset=ds, batch_size=370, shuffle=False)
    x_train, y_train, idx, idx_row = iter(dl_train).next()
    print(x_train.shape)
    dl_test = DataLoader(dataset=ds_test, batch_size=370, shuffle=False)
    x_train, y_train, idx, idx_row = iter(dl_train).next()
    x_test, y_test, idx, idx_row = iter(dl_test).next()
    X_train_endog = x_train.detach().cpu()[:, 0, :].numpy()
    X_train_exog = x_train.detach().cpu()[:, 1, :].numpy()
    X_test_endog = x_test.detach().cpu()[:, 0, :].numpy()
    X_test_exog = x_test.detach().cpu()[:, 1, :].numpy()
    print(X_train_endog.shape)
    print(X_train_exog.shape)
    print(X_test_endog.shape)
    print(X_test_exog.shape)
    d_list = []
    order_list = []
    all_predictions = []
    for ts in range(X_train_endog.shape[0]):
        endog = X_train_endog[ts, -200:]
        exog = X_train_exog[ts, -200:]
        print(f"Fitting {ts+1:3} of {X_train_endog.shape[0]}")
        order = brute(objfunc, grid, args=(endog, exog), finish=None)
        order_list.append(
            {
                "name": ts,
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "P": order[3],
                "D": order[4],
                "Q": order[5],
                "s": 24,
            }
        )
        pdq = order[:3]
        PDQ = order[3:]
        PDQs = np.append(PDQ, 24)
        mod = SARIMAX(endog=endog, exog=exog, order=pdq, seasonal_order=PDQs)
        mod_fit = mod.fit(maxiter=20, disp=False)
        d = dict(zip(mod.param_names, mod_fit.params))
        d["id"] = ts
        d_list.append(d)

        # rolling forecast
        hist_endog = endog
        hist_exog = exog
        predictions = []
        for i in range(7):
            model = SARIMAX(
                endog=hist_endog, exog=hist_exog, order=pdq, seasonal_order=PDQs
            ).fit(maxiter=20, disp=False)
            if -(7 - i - 1) * 24 == 0:
                next_exog = X_test_exog[ts, -(7 - i) * 24 :].flatten()
            else:
                next_exog = X_test_exog[ts, -(7 - i) * 24 : -(7 - i - 1) * 24].flatten()
            pred = model.forecast(steps=24, exog=next_exog)
            predictions = np.append(predictions, pred)
            if -(7 - i + 1) * 24 == 0:
                real = X_test_endog[ts, -(7 - i) * 24 :].flatten()
            else:
                real = X_test_endog[ts, -(7 - i) * 24 : -(7 - i - 1) * 24].flatten()
            hist_endog = np.append(hist_endog, real)
            hist_exog = np.append(hist_exog, next_exog)

        all_predictions.append(predictions)
        # plt.plot(predictions)
        # plt.plot(hist_endog[-24 * 7 :])
        # plt.legend(["preds", "orig"])
        # plt.show()

    df = pd.DataFrame(d_list)
    df.to_csv("representations/representation_matrices/electricity_sarimax.csv")

    df = pd.DataFrame(order_list)
    df.to_csv("representations/representation_matrices/electricity_sarimax_order.csv")

    predictions = np.stack(all_predictions, axis=0)
    actual = X_test_endog[:, -24 * 7 :]

    wape = WAPE(actual, predictions)
    mape = MAPE(actual, predictions)
    smape = SMAPE(actual, predictions)
    mae = MAE(actual, predictions)
    rmse = RMSE(actual, predictions)

    print(f"WAPE = {wape:.3f}")
    print(f"MAPE = {mape:.3f}")
    print(f"SMAPE = {smape:.3f}")
    print(f"MAE = {mae:.3f}")
    print(f"RMSE = {rmse:.3f}")
    # calculate metrics all

    # save all the stuff
    with open("sarimax_electricity.txt", "w") as f:
        f.write(f"WAPE = {wape:.3f}")
        f.write("\n")
        f.write(f"MAPE = {mape:.3f}")
        f.write("\n")
        f.write(f"SMAPE = {smape:.3f}")
        f.write("\n")
        f.write(f"MAE = {mae:.3f}")
        f.write("\n")
        f.write(f"RMSE = {rmse:.3f}")
    print(f"Total time: {datetime.timedelta(seconds=round((time.time() - t0), 1))} ")
