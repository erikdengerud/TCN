"""
Creating representations of time series.
"""

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pmdarima as pm


def calculate_representation(
    Y, representation, num_components, dataset, sector_or_id="id"
):
    print("Num components ", num_components)
    if representation == "pca":
        Y_rep = calculate_pca(Y, num_components=num_components)
    elif representation == "tsne":
        Y_rep = calculate_tsne(Y, num_components=num_components)
    elif representation == "sarima":
        Y_rep = calculate_sarima(dataset=dataset)
    elif representation == "embedded_id":
        Y_rep = calculate_embedded_id(
            dataset=dataset, num_components=num_components, sector_or_id=sector_or_id
        )
    else:
        print("No such representation available, Rep = raw.")
        Y_rep = Y
    return Y_rep


def calculate_pca(Y, num_components):
    print("Num components ", num_components)
    pca = PCA(n_components=num_components)
    Y_pca = pca.fit_transform(Y)
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_)}")
    return Y_pca


def calculate_sarima(dataset):
    """ Get the pre fitted parameters """
    try:
        if dataset == "electricity":
            df = pd.read_csv(
                "representations/representation_matrices/electricity_sarima.csv",
                index_col=0,
            ).fillna(0)
        elif dataset == "revenue":
            df = pd.read_csv(
                "representations/representation_matrices/revenue_sarima.csv",
                index_col=0,
            ).fillna(0)
    except Exception as e:
        print(e)
    df = df.set_index("id")
    df = df.drop(columns=["sigma2"])
    rep = df.values
    return rep


def calculate_embedded_id(dataset, num_components, sector_or_id="id"):
    """ Precomputed embeddings """
    try:
        if dataset == "electricity":
            Y = np.load(
                f"representations/representation_matrices/electricity_scaled_embedded_id_nc_{num_components}.npy"
            )
        elif dataset == "revenue":
            Y = np.load(
                f"representations/representation_matrices/revenue_embedded_{sector_or_id}_nc_{num_components}.npy"
            )
    except Exception as e:
        print(e)
    return Y


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
