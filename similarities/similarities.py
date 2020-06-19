"""
General similarity calculation.
"""
from sklearn.metrics import pairwise_distances
from dtaidistance.dtw import distance_matrix_fast
import numpy as np


def calculate_similarity_matrix(
    X: np.array, metric: str = "euclidean", dist_or_sim: str = "sim"
) -> np.array:
    if metric == "dtw":
        S = np.load("similarities/similarity_matrices/dtw.npy")
        #S = distance_matrix_fast(X[:, -500:])
        #S[np.isinf(S)] = 0
        #S = S + S.T
    else:
        S = pairwise_distances(X, metric=metric)

    if dist_or_sim == "sim":
        # RBF kernel on the measure, should possibly scale too
        S = np.exp(-(S ** 2) / (2 * np.var(S)))

    return S


if __name__ == "__main__":
    import time
    import sys

    sys.path.append("")
    sys.path.append("../../")

    from electricity.data import ElectricityDataSet

    ds = ElectricityDataSet(
        file_path="electricity/data/electricity.npy",
        data_scaler=None,
        data_scale=True,
        start_date="2012-01-01",
        end_date="2014-12-16",
    )
    X = ds.X.squeeze(1).detach().cpu().double().numpy()

    print(X.shape)
    X = X[:, -500:]
    print(X.shape)
    print(X)

    S = distance_matrix_fast(X)
    S[np.isinf(S)] = 0
    S = S + S.T
    np.save("similarities/similarity_matrices/dtw.npy", S)
    print(S.shape)
    print(S)
    """
    X = np.load("representations/representation_matrices/electricity_train_scaled.npy")
    print(X)
    t1 = time.time()
    S = calculate_similarity_matrix(X, metric="euclidean")
    print(S)
    np.save(
        "similarities/similarity_matrices/electricity_train_scaled_euclidean.npy", S
    )

    t2 = time.time()
    S = calculate_similarity_matrix(X, metric="dtw")
    np.save("similarities/similarity_matrices/electricity_train_scaled_dtw.npy", S)
    print(S)
    t3 = time.time()
    S = calculate_similarity_matrix(X, metric="correlation")
    np.save("similarity_matrices/electricity_train_scaled_correlation.npy", S)
    print(S)
    t4 = time.time()
    print(f"Euclidean: {t2-t1:.2f} sec")
    print(f"dtw: {t3-t2:.2f} sec")
    print(f"Correlation: {t4-t3:.2f} sec")
    """
