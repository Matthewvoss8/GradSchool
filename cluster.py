import pandas as pd
from sklearn.cluster import KMeans
from main import LymeDisease
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import silhouette_samples
import numpy as np

l = LymeDisease()
data = l.data[l.features + [l.target]]
data = pd.get_dummies(data)
x = data.loc[:, data.columns != l.target].to_numpy()
y = data.loc[:, data.columns == l.target].to_numpy().reshape(-1)


def find_number_clusters(x, number):
    inertia = []
    for i in range(1, number):
        km = KMeans(
            n_clusters=i,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=767
        )
        km.fit(x)
        inertia.append(km.inertia_)
    plt.plot(range(1, number), inertia)
    plt.ylabel('km.inertia_')
    plt.xlabel('# of Clusters')
    plt.title('Inertia vs # Clusters')
    plt.show()


def plot_silhouette(x, k):
    km = KMeans(n_clusters=k,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=767)
    y_hat = km.fit_predict(x)
    c_labels = np.unique(y_hat)
    n_clusters = c_labels.shape[0]
    silhouette = silhouette_samples(x, y_hat)
    lower, upper = 0, 0
    yticks = []
    for i, c in enumerate(c_labels):
        c_sil = silhouette[y_hat == c]
        c_sil.sort()
        upper += len(c_sil)
        color = cm.jet(float(i), n_clusters)
        plt.barh(range(lower, upper),
                 c_sil,
                 height=1.0,
                 edgecolor='none',
                 color=color)
        yticks.append(np.mean([lower, upper]))
        lower += len(c_sil)
    sil_average = np.mean(c_sil)
    plt.axvline(sil_average,
                color='black')
    plt.yticks(yticks, np.unique(y_hat) + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette Cluster')
    plt.show()


if __name__ == '__main__':
    find_number_clusters(x, 5)
    plt.close()
    k = 2
    plot_silhouette(x, k)
