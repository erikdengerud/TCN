import numpy as np
import pandas as pd
from revenue.data import RevenueDataset
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import warnings
import time
import datetime

warnings.filterwarnings("ignore")
from tqdm import tqdm


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

df_order = pd.read_csv(
    "Z:\TCN_clone\TCN\\representations\\representation_matrices\\revenue_sarima_order.csv",
    index_col=0,
)


X = ds.X.detach().cpu().squeeze(1).numpy()
X_test = ds_test.X.detach().cpu().squeeze(1).numpy()
d_list = []
order_list = []
all_predictions = []

t0 = time.time()
for index, row in tqdm(df_order.iterrows(), total=df_order.shape[0]):
    endog = X[index, :]
    pdq = row[1:4].to_numpy()
    PDQs = row[4:].to_numpy()
    hist = endog
    predictions = []
    for i in range(2):
        try:
            model = SARIMAX(endog=hist, order=pdq, seasonal_order=PDQs).fit(
                maxiter=20, disp=False
            )
            pred = model.forecast(steps=4)
        except:
            if i == 0:
                pred = np.repeat(X[index, -1], 4)
            else:
                pred = np.repeat(X_test[index, 4], 4)

        predictions = np.append(predictions, pred)
        if -(2 - i - 1) * 4 == 0:
            real = X_test[index, -(2 - i) * 4 :].flatten()
        else:
            real = X_test[index, -(2 - i) * 24 : -(2 - i - 1) * 4].flatten()
        hist = np.append(hist, real)
    all_predictions.append(predictions)

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
