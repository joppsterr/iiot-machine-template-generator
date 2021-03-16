from sklearn import cluster as cl
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from custom_functions.corr_df import corr_df
from custom_functions.result_plot import plot_result

def correlation_check(filename):
    dataset = pd.read_csv(filename)
    correlation_matrix = dataset.corr()
    return correlation_matrix

def main():

    # Read selected dataset
    dataset = pd.read_csv("datasets/gas_sensor_array.csv")

    # Correlation check
    #print(correlation_check("datasets/gas_sensor_array.csv"))

    # Select relevant columns of dataset with low correlation
    subset = corr_df(dataset, 0.5)

    # Reduce rows from 300 000 to specified n
    # TODO Create a correct sampling method
    small_subset = subset.sample(n=80000)

    # Transform data
    scaled_subset = StandardScaler().fit_transform(small_subset)

    # Reduce dimension with PCA
    pca = PCA(n_components=0.95)
    pca_subset = pca.fit_transform(scaled_subset)
    print(pca.components_)

    # Run DBSCAN
    db = cl.DBSCAN(eps=0.8, min_samples=20).fit(pca_subset)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Run Birch
    #db = cl.Birch(n_clusters=None).fit(pca_subset)

    # Start of results
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_subset, labels))

    plot_result(pca_subset,labels,core_samples_mask,n_clusters_)

if __name__ == '__main__':
    main()