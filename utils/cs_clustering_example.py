print("Importing...")
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import ListedColormap

rc("text", usetex=True)

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

print("Imported")
print("Making dataset...")
n_samples = 1500
X, y = make_moons(n_samples=n_samples, noise=0.05)

X = StandardScaler().fit_transform(X)
print("Made dataset")
print("Clustering...")
spectral = SpectralClustering(
    n_clusters=2, eigen_solver="arpack", affinity="nearest_neighbors"
).fit(X)
kmeans = KMeans(n_clusters=2).fit(X)
aggl = AgglomerativeClustering(n_clusts=2).fit(X)


print("Done clustering")
print("Plotting...")
colors = ["#FFC000", "#00B0F0"]

size = 4.77 / 3
plt.figure(figsize=(size, size))
plt.scatter(
    X[:, 0],
    X[:, 1],
    s=10,
    c=spectral.labels_,
    cmap=ListedColormap(colors),
    edgecolors="black",
    linewidths=0.1,
)
# plt.show()
plt.axis("off")
plt.savefig("spectral_clustering.pdf", bbox_inches="tight")

plt.figure(figsize=(size, size))
plt.scatter(
    X[:, 0],
    X[:, 1],
    s=10,
    c=kmeans.labels_,
    cmap=ListedColormap(colors),
    edgecolors="black",
    linewidths=0.1,
)
# plt.show()
plt.axis("off")
plt.savefig("kmeans.pdf", bbox_inches="tight")

plt.figure(figsize=(size, size))
plt.scatter(
    X[:, 0],
    X[:, 1],
    s=10,
    c=aggl.labels_,
    cmap=ListedColormap(colors),
    edgecolors="black",
    linewidths=0.1,
)
# plt.show()
plt.axis("off")
plt.savefig("agglomerative.pdf", bbox_inches="tight")
