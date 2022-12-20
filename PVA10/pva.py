import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.title("Dendrogram nach hierarchischem Clustering -- Wine dataset")
    plt.xlabel("Number of points in node")
    plt.show()

iris = datasets.load_digits()
X = iris.data
# setting distance_threshold=0 ensures we compute the full tree.
iris_model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
iris_model.fit_predict(X)
plot_dendrogram(iris_model, truncate_mode="level", p=3)