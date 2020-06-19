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

from sklearn.cluster import *

from representations.representations import calculate_representation
from similarities.similarities import calculate_similarity_matrix


parser = argparse.ArgumentParser()
parser.add_argument("--df_path", type=str)
parser.add_argument("--dataset", type=str, default="electricity")
parser.add_argument("--model_path", type=str)
args = parser.parse_args()

# TODO: Check if the prototypes already exists


if args.dataset == "electricity":
    from electricity.data import ElectricityDataSet
    from electricity.model import TCN
elif args.dataset == "revenue":
    from revenue.data import RevenueDataset
    from revenue.model import TCN
else:
    raise RuntimeError("No valid dataset specified.")


# read in arguments from df
try:
    df = pd.read_csv(args.df_path).dropna(how="all")
except Exception as e:
    print(e)
    print("Not able to read df.")

assert set(
    [
        "representation",
        "scaled_representation",
        "num_components",
        "algorithm",
        "similarity",
        "num_clusters",
    ]
).issubset(set(df.columns))
assert set(df.representation).issubset(set(["sarima", "pca", "embedded_id", "raw"]))
assert set(df.similarity).issubset(set(["euclidean", "correlation", "dtw"]))
assert set(df.algorithm).issubset(
    set(["KMeans", "Agglomerative", "Spectral clustering"])
)
df = df.astype({"representation":str, "scaled_representation":bool, "num_components":int, "algorithm":str, "similarity":str, "num_clusters":int})

# datasets
if args.dataset == "electricity":
    ds_train_unscaled = ElectricityDataSet(
        file_path="electricity/data/electricity.npy",
        data_scale=False,
        data_scaler=None,
        start_date="2012-01-01",
        end_date="2014-12-16",
        receptive_field=0,
    )
    ds_train_scaled = ElectricityDataSet(
        file_path="electricity/data/electricity.npy",
        data_scale=True,
        data_scaler=None,
        start_date="2012-01-01",
        end_date="2014-12-16",
        receptive_field=0,
    )
    ds_test_unscaled = ElectricityDataSet(
        file_path="electricity/data/electricity.npy",
        data_scale=False,
        data_scaler=ds_train_scaled.data_scaler,
        receptive_field=385,
        start_date="2014-12-16",
        end_date="2014-12-23",
    )
    ds_test_scaled = ElectricityDataSet(
        file_path="electricity/data/electricity.npy",
        data_scale=True,
        data_scaler=ds_train_scaled.data_scaler,
        receptive_field=385,
        start_date="2014-12-16",
        end_date="2014-12-23",
    )
    # model
    try:
        model_args = pickle.load(open(f"{args.model_path}__args.pkl", "rb"))
    except Exception as e:
        print(e)
        print("No valid model parameter dict.")

    try:
        model = TCN(
            num_layers=model_args.num_layers,
            in_channels=1,
            out_channels=1,
            residual_blocks_channel_size=[model_args.res_block_size]
            * model_args.num_layers,
            kernel_size=model_args.kernel_size,
            bias=model_args.bias,
            dropout=model_args.dropout,
            stride=1,
            leveledinit=model_args.leveledinit,
            embedding_dim=model_args.embedding_dim,
            embed=model_args.embed,
        )
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        model.eval()
    except Exception as e:
        print(e)
        print("Not able to read model.")

elif args.dataset == "revenue":
    pass
    print("Not yet implemented. Coma back in a few.")
else:
    raise RuntimeError("Not able to read in dataset.")

# representation, similarity, cluster
X_train_scaled = ds_train_scaled.X.squeeze(1).detach().cpu().numpy()
X_train_unscaled = ds_train_unscaled.X.squeeze(1).detach().cpu().numpy()
X_test_scaled = ds_test_scaled.X.squeeze(1).detach().cpu().numpy()
X_test_unscaled = ds_test_unscaled.X.squeeze(1).detach().cpu().numpy()

for i, row in tqdm(df.iterrows(), total=df.shape[0], ascii=True, desc="Rows in df"):
    prots_path = (
        f"prototypes/prototypes_matrices/{args.dataset}_"
        f"{df.representation[i]}_"
        f"{'scaled_' if df.scaled_representation[i] else 'unscaled_'}_"
        f"nc_{df.num_components[i]}_"
        f"{df.similarity[i]}_"
        f"{df.algorithm[i]}_"
        f"nc_{df.num_clusters[i]}.npy"
    )

    dict_path = (
        f"prototypes/cluster_dicts/{args.dataset}_"
        f"{df.representation[i]}_"
        f"{'scaled_' if df.scaled_representation[i] else 'unscaled_'}"
        f"nc_{df.num_components[i]}_"
        f"{df.similarity[i]}_"
        f"{df.algorithm[i]}_"
        f"nc_{df.num_clusters[i]}.pkl"
    )

    if os.path.isfile(prots_path):
        print("Prototypes file exists!")
    if os.path.isfile(dict_path):
        print("Dict file exists!")

    """ Algorithm """
    if df.algorithm[i] in ("KMeans"):
        method = KMeans(n_clusters=df.num_clusters[i])
        dist_or_sim_or_feat = "feat"
    elif df.algorithm[i] in ("Agglomerative"):
        method = AgglomerativeClustering(
            n_clusters=df.num_clusters[i], affinity="precomputed", linkage="average"
        )
        dist_or_sim_or_feat = "dist"
    elif df.algorithm[i] in ("Spectral clustering"):
        method = SpectralClustering(
            n_clusters=df.num_clusters[i], affinity="precomputed"
        )
        dist_or_sim_or_feat = "sim"

    if df.scaled_representation[i]:
        rep = calculate_representation(
            X_train_scaled,
            representation=df.representation[i],
            num_components=df.num_components[i],
            dataset=args.dataset
        )
    else:
        rep = calculate_representation(
            X_train_unscaled,
            representation=df.representation[i],
            num_components=df.num_components[i],
            dataset=args.dataset
        )

    """ Representation """
    if dist_or_sim_or_feat == "feat":
        D = rep
    else:
        D = calculate_similarity_matrix(
            rep, metric=df.similarity[i], dist_or_sim=dist_or_sim_or_feat
        )

    """ Clustering """
    clusters = method.fit_predict(D)
    cluster_dist = {
        c: len(clusters[np.where(clusters == c)]) for c in range(len(set(clusters)))
    }
    for k, v in cluster_dist.items():
        print(f"{k:2} : {v}")
    cluster_dict = {i: clusters[i] for i in range(len(clusters))}

    """ Calculate prototypes """
    train_prototypes = {}
    test_prototypes = {}
    for c in set(clusters):
        # get all ts from Y that are in cluster c
        ids = np.where(clusters == c)[0]
        # take mean
        ts_train = X_train_scaled[ids]
        ts_test = X_test_scaled[ids]
        prot_train = np.mean(ts_train, axis=0)
        prot_test = np.mean(ts_test, axis=0)
        # store in dict
        train_prototypes[c] = prot_train
        test_prototypes[c] = prot_test

    full_pred_prototypes = []
    for k in tqdm(
        train_prototypes.keys(), total=len(train_prototypes), desc="Prototypes"
    ):
        p_train = torch.from_numpy(train_prototypes[k]).view(1, 1, -1)
        p_test = torch.from_numpy(test_prototypes[k]).view(1, 1, -1)
        full_prototype = torch.cat((p_train, p_test[:, :, -24 * 7 :]), 2)
        tau = 24 if args.dataset == "electricity" else 4
        num_windows = 7 if args.dataset == "electricity" else 2
        with torch.no_grad():
            pred_prototype, _ = model.rolling_prediction(
                full_prototype, emb_id=k, tau=tau, num_windows=num_windows
            )
        full_pred_prototypes.append(torch.cat((p_train.squeeze(1), pred_prototype), 1))

    full_pred_prototypes = torch.cat(full_pred_prototypes, 0)

    """ Save """
    prots_np = full_pred_prototypes.detach().cpu().numpy()
    np.save(prots_path, prots_np)

    with open(dict_path, "wb") as handle:
        pickle.dump(cluster_dict, handle)

print("Done")
