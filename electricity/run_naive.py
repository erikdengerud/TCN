import sys

sys.path.append("")
sys.path.append("../../")

import numpy as np
import torch

from electricity.data import ElectricityDataSet

from utils.metrics import WAPE, MAPE, SMAPE, MAE, RMSE


if __name__ == "__main__":
    import multiprocessing

    print(multiprocessing.cpu_count())
    # dataset
    scale = True
    ds_train = ElectricityDataSet(
        file_path="electricity/data/electricity.npy",
        data_scale=True,
        data_scaler=None,
        start_date="2012-01-01",  # yyyy-mm-dd
        end_date="2014-12-26",  # yyyy-mm-dd
        predict_ahead=1,
        h_batch=0,  # 0 gives the whole time series
        receptive_field=0,
        cluster_covariate=False,
        random_covariate=False,
        zero_covariate=False,
        cluster_dict_path="test_cluster_dict.pkl",
    )

    ds_test = ElectricityDataSet(
        file_path="electricity/data/electricity.npy",
        data_scale=False,
        data_scaler=None,
        start_date="2014-12-16",  # yyyy-mm-dd
        end_date="2014-12-23",  # yyyy-mm-dd
        predict_ahead=1,
        h_batch=0,  # 0 gives the whole time series
        receptive_field=0,
        cluster_covariate=False,
        random_covariate=False,
        zero_covariate=False,
        cluster_dict_path="test_cluster_dict.pkl",
    )

    # preds
    X = ds_test.X.detach().cpu().squeeze(1)
    print(X.shape)
    if scale:
        preds = X[:, -24 * (7 + 1) : -24]
        preds_scale = ds_train.data_scaler.transform(preds.numpy().T).T
        preds_rescaled = ds_train.data_scaler.inverse_transform(preds_scale.T).T
        preds = torch.from_numpy(preds_rescaled)
    else:
        preds = X[:, -24 * (7 + 1) : -24]
    real = X[:, -24 * 7 :]
    print(preds.shape)
    print(real.shape)
    # real

    # metrics
    wape = WAPE(real, preds)
    mape = MAPE(real, preds)
    smape = SMAPE(real, preds)
    mae = MAE(real, preds)
    rmse = RMSE(real, preds)

    print(f"WAPE = {wape:.3f}")
    print(f"MAPE = {mape:.3f}")
    print(f"SMAPE = {smape:.3f}")
    print(f"MAE = {mae:.3f}")
    print(f"RMSE = {rmse:.3f}")
