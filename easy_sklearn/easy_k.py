# -*- coding: utf-8 -*-
from easy_base import EasySklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


class EasySklearnClustering(EasySklearn):
    def __init__(self):
        EasySklearn.__init__(self)
        self.n_clusters = 4

    @property
    def default_models_(self):
        return {
            'KM': {'clf': KMeans(n_clusters=self.n_clusters,  random_state=9),
                   'param': {}}
        }

    @property
    def default_models_name_(self):
        return [model for model in self.default_models_]

    # 画聚类图
    def plot_cluster(self, X, clf):
        clf.fit(X)
        n_class = clf.cluster_centers_.shape[0]
        new_df = pd.DataFrame(X)
        new_df['_label'] = clf.labels_
        pca = PCA(n_components=2)
        new_pca = pd.DataFrame(pca.fit_transform(new_df))
        colors = ['b.', 'go', 'r*', 'k.', 'c.', 'm*', 'y,', '#e24fff', '#524C90', '#845868']
        for i in range(0, n_class):
            d = new_pca[new_df['_label'] == i]
            plt.plot(d[0], d[1], colors[i])
        # plt.gcf().savefig('kmeans.png')
        plt.show()
