from numba import jit

import matplotlib.pyplot as plt
import numpy as np

# @jit(target="cuda")
def plot_result_3d(X, labels, core_samples_mask, n_clusters_):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
        for each in np.linspace(0, 1, len(unique_labels))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xyz = X[class_member_mask & core_samples_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=tuple(col), edgecolors='none', marker='+')

        # xyz = X[class_member_mask & ~core_samples_mask]
        # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=tuple(col))

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

def plot_result_with_noise_2d(X, labels, core_samples_mask, n_clusters_):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
        for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=8)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'x', markerfacecolor=tuple(col), markeredgecolor='k', markersize=0)


    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()