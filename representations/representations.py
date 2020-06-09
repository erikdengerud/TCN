"""
Creating representations of time series.
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pmdarima as pm


def calculate_representation(Y, representation, num_components, **kwargs):
    print("Num components ", num_components)
    if representation == "pca":
        Y_rep = calculate_pca(Y, num_components=num_components)
    elif representation == "tsne":
        Y_rep = calculate_tsne(Y, num_components=num_components)
    elif representation == "sarima":
        Y_rep = calculate_sarima(Y, num_components=num_components, **kwargs)
    else:
        print("No such representation available")
        Y_rep = None
    return Y_rep


def calculate_pca(Y, num_components):
    print("Num components ", num_components)
    pca = PCA(n_components=num_components)
    Y_pca = pca.fit_transform(Y)
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_)}")
    return Y_pca


def calculate_sarima(Y, s):
    if Y.shape[1] > 200:
        Y = Y[:, -200:]
    params_list = []
    print(Y.shape)
    print(Y.shape[0])
    print(Y[1].shape)
    for i in range(Y.shape[0]):
        if i > 10:
            break
        print(i)
        train = Y[i][~np.isnan(Y[i])].copy()
        print("fitting model")
        modl = pm.auto_arima(
            train,
            start_p=1,
            start_q=1,
            start_P=1,
            start_Q=1,
            max_p=5,
            max_q=5,
            max_P=5,
            max_Q=5,
            seasonal=True,
            stepwise=True,
            suppress_warnings=False,
            D=1,
            max_D=3,
            m=s,
            error_action="ignore",
        )
        print("done fit")

        d = modl.to_dict()
        params = list(d["params"][1:-1])  # not intercept and variance. list to pop
        order = d["order"]
        seasonal_order = d["seasonal_order"]
        arma = {
            "ar": order[0],
            "ma": order[2],
            "AR": seasonal_order[0],
            "MA": seasonal_order[2],
        }
        p = {}
        for term, num in arma.items():
            for i in range(num):
                p["".join([term, str(i + 1)])] = params.pop()

        params_list.append(p)

    df = pd.DataFrame(params_list)
    print(df.head())
    ret = df.values
    return ret


def calculate_tsne(Y, num_components):
    # Do pca first if there are more than 30 components
    if Y.shape[1] > 30:
        pca = PCA(n_components=30)
        Y = pca.fit_transform(Y)
    tsne = TSNE(n_components=num_components, method="exact")
    Y_tsne = tsne.fit_transform(Y)
    return Y_tsne


def calculate_ae(Y, num_components):
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # electricity sarima
    Y = np.load("representations/representation_matrices/electricity_train_raw.npy")
    print(calculate_pca(Y, 10))
    X = calculate_sarima(Y, 24)
    np.save("representations/representation_matrices/electricity_train_sarima.npy")

    """

    # electricity pca
    Y = np.load("representations/representation_matrices/electricity_train_scaled.npy")
    Y_raw = np.load("representations/representation_matrices/electricity_train_raw.npy")
    print(Y.shape)

    mean = np.mean(Y_raw, axis=1)
    mean_75 = np.quantile(mean, 0.75)
    print(f"75% quantile is {mean_75}")
    mean = np.where(mean < mean_75, mean, mean_75)
    cm = plt.cm.get_cmap("RdYlBu")

    # PCA
    Y_pca = calculate_pca(Y, 20)
    print(Y_pca.shape)

    sc = plt.scatter(Y_pca[:, 0], Y[:, 1], c=mean, cmap=cm)
    plt.colorbar(sc)
    plt.show()

    # TSNE
    Y_tsne = calculate_tsne(Y, 2)

    sc = plt.scatter(Y_tsne[:, 0], Y[:, 1], c=mean, cmap=cm)
    plt.colorbar(sc)
    plt.show()

    # revenue pca
    Y = np.load("representations/representation_matrices/revenue_train_scaled.npy")
    Y_raw = np.load("representations/representation_matrices/revenue_train_raw.npy")
    Y[np.isnan(Y)] = 0
    Y_raw[np.isnan(Y_raw)] = 0
    print(Y.shape)

    mean = np.mean(Y_raw, axis=1)
    mean_75 = np.quantile(mean, 0.75)
    mean_25 = np.quantile(mean, 0.25)
    print(f"75% quantile is {mean_75}")
    mean = np.where(mean < mean_75, mean, mean_75)
    mean = np.where(mean > mean_25, mean, mean_25)

    # PCA
    Y_pca = calculate_pca(Y, 2)
    print(Y_pca.shape)

    sc = plt.scatter(Y_pca[:, 0], Y[:, 1], c=mean, cmap=cm)
    plt.colorbar(sc)
    plt.show()

    # TSNE
    Y_tsne = calculate_tsne(Y, 2)

    sc = plt.scatter(Y_tsne[:, 0], Y[:, 1], c=mean, cmap=cm)
    plt.colorbar(sc)
    plt.show()
    """
