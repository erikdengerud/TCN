import sys

sys.path.append("")
sys.path.append("../../")

import numpy as np
import torch

from revenue.data import RevenueDataset

from utils.metrics import WAPE, MAPE, SMAPE, MAE, RMSE


if __name__ == "__main__":
    import multiprocessing

    print(multiprocessing.cpu_count())
    # dataset
    scale = True
    ds_train = RevenueDataset(
        file_path="revenue/data/processed_companies.csv",
        meta_path="revenue/data/comp_sect_meta.csv",
        data_scale=False,
        data_scaler=None,
        start_date="2012-01-01",
        end_date="2017-01-01",
    )

    ds_test = RevenueDataset(
        file_path="revenue/data/processed_companies.csv",
        meta_path="revenue/data/comp_sect_meta.csv",
        data_scale=False,
        data_scaler=None,
        start_date="2012-01-01",
        end_date="2017-01-01",
    )

    # preds
    X = ds_test.X.detach().cpu().squeeze(1)
    print(X.shape)
    preds = X[:, -4 * (2 + 1) : -4]
    real = X[:, -4 * 2 :]
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
