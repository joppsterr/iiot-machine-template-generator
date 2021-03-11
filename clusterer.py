from sklearn import cluster as cl
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

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
    small_subset = subset.sample(n=10000)
    labels_true = subset.columns

    # Transform data
    X = StandardScaler().fit_transform(small_subset)

    # Run DBSCAN
    db = cl.DBSCAN(min_samples=3).fit(X)

    ################################################################
    # Start of results
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

if __name__ == '__main__':
    main()