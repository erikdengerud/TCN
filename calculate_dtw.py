import sys

sys.path.append("")
sys.path.append("../../")
import numpy as np
import pandas as pd
import argparse
import torch
import pickle
from tqdm import tqdm
import os

from dtaidistance import distance_matrix_fast
from revenue.data import RevenueDataset

# datasets
ds_train_unscaled = RevenueDataset(
    file_path="revenue/data/processed_companies.csv",
    meta_path="revenue/data/comp_sect_meta.csv",
    data_scale=False,
    data_scaler=None,
    start_date="2007-01-01",
    end_date="2017-01-01",
    receptive_field=0,
)
ds_train_scaled = RevenueDataset(
    file_path="revenue/data/processed_companies.csv",
    meta_path="revenue/data/comp_sect_meta.csv",
    data_scale=True,
    data_scaler=None,
    start_date="2007-01-01",
    end_date="2017-01-01",
    receptive_field=0,
)
ds_test_unscaled = RevenueDataset(
    file_path="revenue/data/processed_companies.csv",
    meta_path="revenue/data/comp_sect_meta.csv",
    data_scale=False,
    data_scaler=ds_train_scaled.data_scaler,
    start_date="2007-01-01",
    end_date="2017-01-01",
    receptive_field=19,
)
ds_test_scaled = RevenueDataset(
    file_path="revenue/data/processed_companies.csv",
    meta_path="revenue/data/comp_sect_meta.csv",
    data_scale=True,
    data_scaler=ds_train_scaled.data_scaler,
    start_date="2007-01-01",
    end_date="2017-01-01",
    receptive_field=19,
)
X_train_scaled = ds_train_scaled.X.squeeze(1).detach().cpu().numpy()
X_train_unscaled = ds_train_unscaled.X.squeeze(1).detach().cpu().numpy()
X_test_scaled = ds_test_scaled.X.squeeze(1).detach().cpu().numpy()
X_test_unscaled = ds_test_unscaled.X.squeeze(1).detach().cpu().numpy()

S = distance_matrix_fast(X_train_scaled)
S[np.isinf(S)] = 0
S = S + S.T
np.save("similarities/similarity_matrices/dtw_revenue_scaled.npy", S)

print("Done dtw scaled.")
S = distance_matrix_fast(X_train_scaled)
S[np.isinf(S)] = 0
S = S + S.T
np.save("similarities/similarity_matrices/dtw_revenue_unscaled.npy", S)
print("Done dtw unscaled.")
