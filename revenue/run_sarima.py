import sys

sys.path.append("")
sys.path.append("../../")
import numpy as np
import pandas as pd
from revenue.data import RevenueDataset
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.optimize import brute
import matplotlib.pyplot as plt
import warnings
import time
import datetime
import torch

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
    # print(order)
    pdq = order[:3]
    PDQ = order[3:]
    PDQs = np.append(PDQ, 4)
    fit = SARIMAX(endog=endog, order=pdq, seasonal_order=PDQs).fit(
        disp=False, maxiter=20
    )
    return fit.aic


if __name__ == "__main__":
    t0 = time.time()
    print("Dataset")
    ds = RevenueDataset(
        file_path="revenue/data/processed_companies.csv",
        meta_path="revenue/data/comp_sect_meta.csv",
        data_scale=False,
        data_scaler=None,
        start_date="2012-01-01",  # yyyy-mm-dd
        end_date="2017-01-01",  # yyyy-mm-dd
    )
    ds_test = RevenueDataset(
        file_path="revenue/data/processed_companies.csv",
        meta_path="revenue/data/comp_sect_meta.csv",
        data_scale=False,
        data_scaler=None,
        start_date="2017-03-01",  # yyyy-mm-dd
        end_date="2018-12-31",  # yyyy-mm-dd
    )

    print("Fitting SARIMA models:")
    grid = (
        slice(1, 4, 1),
        slice(1, 2, 1),
        slice(1, 4, 1),
        slice(0, 2, 1),
        slice(1, 2, 1),
        slice(0, 2, 1),
    )

    # predictions =
    X = ds.X.detach().cpu().squeeze(1).numpy()
    X_test = ds_test.X.detach().cpu().squeeze(1).numpy()
    d_list = []
    order_list = []
    all_predictions = []
    for ts in range(X.shape[0]):
        endog = X[ts, :]
        print(f"Fitting {ts+1:3} of {X.shape[0]}")
        try:
            order = brute(objfunc, grid, args=(endog,), finish=None)
        except Exception as e:
            print(e)
            order = np.array(
                [1, 0, 0, 0, 0, 0]
            )  # Using this as a backup if stuff fails.
        order_list.append(
            {
                "name": ts,
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "P": order[3],
                "D": order[4],
                "Q": order[5],
                "s": 4,
            }
        )
        pdq = order[:3]
        PDQ = order[3:]
        PDQs = np.append(PDQ, 4)
        mod = SARIMAX(endog=endog, order=pdq, seasonal_order=PDQs)
        mod_fit = mod.fit(maxiter=20, disp=False)
        d = dict(zip(mod.param_names, mod_fit.params))
        d["id"] = ts
        d_list.append(d)

        # rolling forecast
        hist = endog
        predictions = []
        for i in range(2):
            model = SARIMAX(endog=hist, order=pdq, seasonal_order=PDQs).fit(
                maxiter=20, disp=False
            )
            pred = model.forecast(steps=4)
            predictions = np.append(predictions, pred)
            if -(2 - i - 1) * 4 == 0:
                real = X_test[ts, -(2 - i) * 4 :].flatten()
            else:
                real = X_test[ts, -(2 - i) * 4 : -(2 - i - 1) * 4].flatten()
            hist = np.append(hist, real)
        all_predictions.append(predictions)
        # plt.plot((ds.dates.tolist() + ds_test.dates.tolist()), hist)
        # plt.plot(ds_test.dates.tolist(), predictions)
        # plt.xticks(rotate=45)
        # plt.legend(["orig", "pred"])
        # plt.show()

    df = pd.DataFrame(d_list)
    df.to_csv("representations/representation_matrices/revenue_sarima.csv")

    df = pd.DataFrame(order_list)
    df.to_csv("representations/representation_matrices/revenue_sarima_order.csv")

    predictions = np.stack(all_predictions, axis=0)
    actual = X_test[:, -4 * 2 :]

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
    with open("sarima_revenue.txt", "w") as f:
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
