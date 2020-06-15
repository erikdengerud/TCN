import numpy as np
from data import ElectricityDataSet

from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.optimize import brute

from utils.metrics import MAPE, SMAPE, WAPE, MAE, RMSE

from multiprocessing import Pool

# read dataset

# loop over and fit auti_arima
# Rolling validations
# Save parameters


def objfunc(order, endog):
    print(order)
    pdq = order[:3]
    PDQ = order[3:]
    s = 24
    PDQs = np.append(PDQ, s)
    fit = SARIMAX(endog=endog, order=pdq, seasonal_order=PDQs).fit()
    print(fit.aic)
    return fit.aic


if __name__ == "__main__":
    print("Dataset")
    ds = ElectricityDataSet(
        file_path="electricity/data/electricity.npy",
        data_scale=False,
        data_scaler=None,
        start_date="2014-01-01",  # yyyy-mm-dd
        end_date="2014-16-12",  # yyyy-mm-dd
        predict_ahead=1,
        h_batch=0,  # 0 gives the whole time series
        receptive_field=385,
        cluster_covariate=False,
        random_covariate=False,
        zero_covariate=False,
        cluster_dict_path="test_cluster_dict.pkl",
    )
    print("Fitting SARIMA models:")
    grid = (
        slice(1, 5, 1),
        slice(1, 2, 1),
        slice(1, 5, 1),
        slice(1, 2, 1),
        slice(0, 1, 1),
        slice(1, 2, 1),
    )

    # predictions =
    X = ds.X.detach().cpu().numpy()
    for ts in range(X.shape[0])[:1]:
        pool = Pool(processes=4)
        endog = X[ts, -200:].flatten()
        print(endog)
        print(len(endog))
        print("Fitting")
        res = brute(objfunc, grid, args=(endog,), finish=None)
        print(res)
        # predict rolling windows
        # fit the sarima model
        pdq = order[:3]
        PDQ = order[3:]
        s = 24
        PDQs = np.append(PDQ, s)
        print(PDQs)
        mod = SARIMAX(endog=X[ts, -200:], order=pdq, seasonal_order=PDQ)
        preds, real = rolling_predictions(X, model, num_windows, tau)
        # save predictions
        # save model params
    # Calculate metrics
    # mape = MAPE()

    # Save parameter representation -> saving all models basically
