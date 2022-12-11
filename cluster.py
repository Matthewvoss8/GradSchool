import pandas as pd
from sklearn.cluster import KMeans
from main import LymeDisease
import matplotlib.pyplot as plt
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
    silhouette = silhouette_samples(x, y_hat)
    lower, upper = 0, 0
    xticks = []
    color = ['black', 'yellow']
    for i, c in enumerate(c_labels):
        c_sil = silhouette[y_hat == c]
        c_sil.sort()
        upper += len(c_sil)
        c = color[i]
        plt.bar(x=np.array(range(lower, upper)),
                height=c_sil,
                width=1.0,
                edgecolor='none',
                color=c)
        xticks.append(np.mean([lower, upper]))
        lower += len(c_sil)
    sil_average = np.mean(c_sil)
    plt.axhline(sil_average,
                color='red')
    plt.xticks(xticks, np.unique(y_hat) + 1)
    plt.ylabel('Silhouette Coefficient')
    plt.xlabel('Cluster')
    plt.title('Silhouette Coefficient by cluster')
    plt.show()


if __name__ == '__main__':
    find_number_clusters(x, 5)
    plt.close()
    k = 2
    plot_silhouette(x, k)
