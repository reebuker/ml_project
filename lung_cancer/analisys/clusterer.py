import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE

class Clusterer:
    def __init__(self, n_clusters, dbscan_eps=0.5, dbscan_min_samples=5):
        """ Инициализация алгоритмов кластеризации и снижения размерности """
        self.pca = PCA(n_components=30, random_state=42)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        self.tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)

    def reduce_dims_pca(self, features):
        """ 1. Сжатие признаков через PCA до рекомендуемых 30 измерений """
        print(f"Сжатие признаков из формы {features.shape}...")
        features_30d = self.pca.fit_transform(features)
        return features_30d

    def run_clustering(self, features_30d):
        """ 2. Запуск кластеризации на сжатых данных """
        print("[Clustering] Запуск K-Means и DBSCAN")

        # Обучаем K-Means и получаем метки групп (0 или 1)
        kmeans_labels = self.kmeans.fit_predict(features_30d)
        # Обучаем DBSCAN. Метка -1 будет означать выбросы/шум
        dbscan_labels = self.dbscan.fit_predict(features_30d)

        return kmeans_labels, dbscan_labels

    def prepare_tsne_2d(self, features_30d):
        """ 3. Превращение 30д признаков в 2д координаты для графиков"""
        print(f"[t-SNE] Проекция признаков в 2д пространство...")
        features_2d = self.tsne.fit_transform(features_30d)
        return features_2d



        



        