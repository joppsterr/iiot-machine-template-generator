from sklearn import cluster as cl
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

def main():

    # Read selected dataset
    dataset = pd.read_csv("gas_sensor_array.csv")

    # Select relevant columns of dataset
    # subset = dataset
    subset = dataset[["CO","Humidity","Temperature","Flow rate","Heater voltage"]]
    # subset = dataset[["Humidity","Heater voltage"]]

    # Reduce rows from 300 000 to specified n
    small_subset = subset.sample(n=50000)

    # Transform data
    scaled_subset = StandardScaler().fit_transform(small_subset)

    # Reduce dimension with PCA
    pca = PCA(n_components=2)
    pca_subset = pca.fit_transform(scaled_subset)

    # Run DBSCAN
    db = cl.DBSCAN(min_samples=3).fit(pca_subset)

    # Start of results
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    #print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(pca_subset, labels))

if __name__ == '__main__':
    main()