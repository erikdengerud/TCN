import numpy as np
from electricity.data import ElectricityDataSet

from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.optimize import brute

from utils.metrics import MAPE, SMAPE, WAPE, MAE, RMSE
# read dataset

# loop over and fit auti_arima
# Rolling validations
# Save parameters

ds = ElectricityDataSet(
    file_path="electricity/data/electricity.npy",
    data_scale=False,
    data_scaler=None,
    start_date="2012-01-01",  # yyyy-mm-dd
    end_date="2014-05-26",  # yyyy-mm-dd
    predict_ahead=1,
    h_batch=0,  # 0 gives the whole time series
    receptive_field=385,
    cluster_covariate=False,
    random_covariate=False,
    zero_covariate=False,
    cluster_dict_path="test_cluster_dict.pkl",
)

def objfunc(order, endog):
    pdq = order[:3]
    PDQ = order[3:]
    s = 24
    PDQs = PDQ + tuple(s)
    fit = SARIMAX(endog=endog, order=pdq, seasonal_order=PDQs).fit()
    return fit.aic()


grid = (slice(1, 3, 1), slice(1, 3, 1), slice(1, 3, 1))

predictions = 
for ts in range(X.shape[0]):
    endog = X[ts,-200:]
    res = brute(objfunc, grid, args=ts, finish=None)
    print(res)
    # predict rolling windows
    # fit the sarima model
    pdq = order[:3]
    PDQ = order[3:]
    s = 24
    PDQs = PDQ + tuple(s)
    mod = SARIMAX(endog=X[ts,-200:], order=pdq, seasonal_order=)
    preds, real = rolling_predictions(X, model, num_windows, tau)
    # save predictions
    # save model params
# Calculate metrics
mape = MAPE()
    


# Save parameter representation -> saving all models basically
