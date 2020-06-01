"""
Cluster similarity matrices.
Write clustering and the prototypes to the clusters folder.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sys

sys.path.append("./")
sys.path.append("../../")

from similarities.similarities import calculate_similarity_matrix
from representations.representations import calculate_representation
from sklearn.cluster import *


def cluster_similarity_matrix(
    D: np.array, algorithm: str, num_clusters: int
) -> np.array:
    assert min(D.shape) > 0

    if algorithm == "Affinity propogation":
        method = AffinityPropagation(affinity="precomputed", verbose=True)
    elif algorithm == "Spectral clustering":
        method = SpectralClustering(n_clusters=num_clusters, affinity="precomputed")
    elif algorithm == "Mean-shift":
        method = MeanShift()
    elif algorithm == "DBSCAN":
        method = DBSCAN(metric="precomputed")
    elif algorithm == "Agglomerative":
        method = AgglomerativeClustering(
            n_clusters=num_clusters, affinity="precomputed", linkage="average"
        )
    elif algorithm == "OPTICS":
        method = OPTICS(metric="precomputed")
    elif algorithm == "KMeans":
        method = KMeans(n_clusters=num_clusters)
    else:
        raise RuntimeError("No valid algorithm")

    clusters = method.fit_predict(D)

    num_clusters = len(set(clusters))

    return clusters, num_clusters


def evaluate_clustering(clusters, similarity_matrix):
    pass


def create_prototypes(clusters, Y, num_examples: int = 3):
    prototypes = {}
    examples = {}
    sizes = {}
    for c in set(clusters):
        # get all ts from Y that are in cluster c
        ids = np.where(clusters == c)[0]
        try:
            plot_ids = np.random.choice(ids, num_examples, replace=False)
        except:
            plot_ids = np.random.choice(ids, num_examples)

        # take mean
        ts = Y[ids]
        ex_ts = Y[plot_ids]
        prot = np.mean(ts, axis=0)
        # store in dict
        prototypes[c] = prot
        examples[c] = ex_ts
        sizes[c] = len(ids)

    return prototypes, examples, sizes


def plot_clustering(
    prototypes,
    examples,
    sizes,
    algorithm,
    representation,
    similarity,
    length: int = 168,
):
    len_series = min([len(p) for p in prototypes.values()])
    plot_index = np.random.randint(0, len_series - length)
    for c in prototypes.keys():
        if c > 10:
            break
        d = {}
        d[f"prototype cluster {c} : size={sizes[c]:3}"] = prototypes[c][
            plot_index : plot_index + length
        ]
        for i, ex in enumerate(examples[c]):
            d[f"example {i}"] = ex[plot_index : plot_index + length]
        df = pd.DataFrame(d)
        df.plot(
            figsize=(10, 5),
            title=list(df.columns),
            subplots=True,
            legend=False,
            ylim=(0, 1),
        )
        plt.tight_layout()
    plt.show()


def cluster_ts(
    dataset: str,
    representation: str,
    similarity: str,
    algorithm: str,
    num_clusters: int = 10,
    num_components: int = None,
    plot: bool = False,
    **kwargs,
) -> None:

    if algorithm in ("KMeans", "Mean-shift"):
        dist_or_sim_or_feat = "feat"
    elif algorithm in ("Agglomerative", "DBSCAN", "OPTICS"):
        dist_or_sim_or_feat = "dist"
    elif algorithm in ("Affinity propogation", "Spectral clustering"):
        dist_or_sim_or_feat = "sim"
    else:
        raise RuntimeError("No valid algorithm specified.")

    if num_components is None:
        num_components = num_clusters

    # read similarity matrix if it exists, compute if not
    raw_path = f"representations/representation_matrices/{dataset}_train_scaled.npy"
    rep_path = f"representations/representation_matrices/{dataset}_train_{representation}_nc_{num_components}.npy"
    sim_path = f"similarities/similarity_matrices/{dataset}_train_{representation}_nc_{num_components}_{similarity}_{dist_or_sim_or_feat}.npy"

    Y = np.load(raw_path)
    try:
        X = np.load(rep_path)
    except:
        X = calculate_representation(
            Y, representation=representation, num_components=num_components
        )
        np.save(rep_path, X)
    if dist_or_sim_or_feat in ("dist or sim"):
        try:
            D = np.load(sim_path)
        except:
            D = calculate_similarity_matrix(
                X, metric=similarity, dist_or_sim=dist_or_sim_or_feat
            )
            np.save(sim_path, D)
    else:
        D = None

    # cluster the time series
    if dist_or_sim_or_feat == "feat":
        # if the algorithm works only on features and not similarities
        D = X
    clusters, num_clusters = cluster_similarity_matrix(
        D, algorithm=algorithm, num_clusters=num_clusters
    )

    cluster_dist = {
        c: len(clusters[np.where(clusters == c)]) for c in range(num_clusters)
    }
    for k, v in cluster_dist.items():
        print(f"{k:2} : {v}")

    if clusters is None:
        raise RuntimeError("Clusters is None. It probably didn't converge!")

    # report clustering metrics
    # Not implemented, might skip and only evaluate qualitatively
    # clustering_performance = evaluate_clustering(clusters, D)

    # create prototypes
    prototypes, examples, sizes = create_prototypes(clusters, Y, num_examples=3)

    # plot prototypes and examples from clusters
    if plot:
        plot_clustering(
            prototypes,
            examples,
            sizes,
            algorithm,
            representation,
            similarity,
            length=168,
        )

    # write clustering and prototypes
    with open(
        f"clustering/clusters/{dataset}_train_{representation}_nc_{num_components}_{similarity}_{dist_or_sim_or_feat}_{algorithm}_nc_{num_clusters}_prototypes.pkl",
        "wb",
    ) as handle:
        pickle.dump(prototypes, handle)
    handle.close()

    cluster_dict = {i: clusters[i] for i in range(len(clusters))}

    with open(
        f"clustering/clusters/{dataset}_train_{representation}_nc_{num_components}_{similarity}_{dist_or_sim_or_feat}_{algorithm}_nc_{num_clusters}_clusters.pkl",
        "wb",
    ) as handle:
        pickle.dump(cluster_dict, handle)
    handle.close()


if __name__ == "__main__":
    np.random.seed(1729)
    # cluster_ts(
    #    "electricity", "pca", "euclidean", "KMeans", num_clusters=10, plot=True,
    # )

    inertias = []
    for i in range(3, 20):
        inertia = cluster_ts(
            "electricity", "pca", "euclidean", "KMeans", num_clusters=i
        )
        inertias.append(inertia)
    print(inertias)
    plt.plot(inertias)
    plt.show()
