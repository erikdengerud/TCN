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


def objfunc(order, endog):
    print(order)
    pdq = order[:3]
    PDQ = order[3:]
    PDQs = np.append(PDQ, 24)
    fit = SARIMAX(endog=endog, order=pdq, seasonal_order=PDQs).fit(
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
    )
    ds_test = ElectricityDataSet(
        file_path="electricity/data/electricity.npy",
        data_scale=False,
        data_scaler=None,
        start_date="2014-12-17",  # yyyy-mm-dd
        end_date="2014-12-23",  # yyyy-mm-dd
    )

    print("Fitting SARIMA models:")
    grid = (
        slice(1, 4, 1),
        slice(1, 3, 1),
        slice(1, 4, 1),
        slice(0, 2, 1),
        slice(1, 2, 1),
        slice(0, 2, 1),
    )

    # predictions =
    X = ds.X.detach().cpu().squeeze(1).numpy()
    X_test = ds_test.X.detach().cpu().squeeze(1).numpy()
    d_list = []
    all_predictions = []
    for ts in range(X.shape[0])[:1]:
        endog = X[ts, -200:]
        print(f"Fitting {ts+1:3} of {X.shape[0]}")
        order = brute(objfunc, grid, args=(endog,), finish=None)
        pdq = order[:3]
        PDQ = order[3:]
        PDQs = np.append(PDQ, 24)
        mod = SARIMAX(endog=endog, order=pdq, seasonal_order=PDQs)
        mod_fit = mod.fit(maxiter=20, disp=False)
        d = dict(zip(mod.param_names, mod_fit.params))
        d["id"] = ts
        d_list.append(d)

        # rolling forecast
        hist = endog
        predictions = []
        for i in range(7):
            model = SARIMAX(endog=hist, order=pdq, seasonal_order=PDQs).fit(
                maxiter=20, disp=False
            )
            pred = model.forecast(steps=24)
            predictions = np.append(predictions, pred)
            real = X_test[ts, -(7 - i) * 24 : -(7 - i + 1) * 24].flatten()
            hist = np.append(hist, real)
        all_predictions.append(predictions)
        # plt.plot(predictions)
        # plt.plot(hist[-24 * 7 :])
        # plt.legend(["preds", "orig"])
        # plt.show()

    df = pd.DataFrame(d_list)
    df.to_csv("representations/representation_matrices/electricity_sarima.csv")

    predictions = np.stack(all_predictions, axis=0)
    print(predictions.shape)
    actual = X_test[:1, -24 * 7 :]
    print(actual.shape)

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
    with open("sarima.txt", "w") as f:
        f.write(f"WAPE = {wape:.3f}")
        f.write("\n")
        f.write(f"MAPE = {mape:.3f}")
        f.write("\n")
        f.write(f"SMAPE = {smape:.3f}")
        f.write("\n")
        f.write(f"MAE = {mae:.3f}")
        f.write("\n")
        f.write(f"RMSE = {rmse:.3f}")
    print(
        f"One time series takes {datetime.timedelta(seconds=round((time.time() - t0), 1))} "
    )
