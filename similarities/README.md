# Similarities

This folder contains different similarity measures used for clustering.

Pre computed matrices are in the similarity_matrices folder. The DTW is especially slow so it is better use the stored matrix if it exists.

For euclidean and correlation we use `sklearn.metrics.pairwise_distances`. DTW is calculated using `dtaidistance.dtw.distance_matrix_fast`. 

`sklearn.covariance`


## Available similarities
* DTW
* From `sklearn.metrics.pairwise_distances`
    - braycurtis
    - canberra
    - chebyshev
    - cityblock
    - correlation
    - cosine
    - dice
    - euclidean
    - hamming
    - jaccard
    - jensenshannon
    - kulsinski
    - mahalanobis
    - matching
    - minkowski
    - rogerstanimoto
    - russellrao
    - seuclidean
    - sokalmichener
    - sokalsneath
    - sqeuclidean
    - yule

## Pending
* Siamese