import os
import numpy as np
import matplotlib.pyplot as plt
from clusterer import Clusterer
from sklearn.metrics import silhouette_score, adjusted_rand_score

features_dir = "data/features/"
plots_dir = "data/plots/"

def run_analysis():
    features = np.load(features_dir + "extracted_features.npy")
    true_labels = np.load(features_dir + "extracted_labels.npy")

    analyzer = Clusterer(n_clusters=4, dbscan_eps=0.5, dbscan_min_samples=5)
    
    # Сжимаем данные до 30 компонент и кластеризуем
    features_30d = analyzer.reduce_dims_pca(features)
    kmeans_labels, dbscan_labels = analyzer.run_clustering(features_30d)

    # Считаем формальные метрики качества разделения
    sil_score = silhouette_score(features_30d, kmeans_labels)
    ari_score = adjusted_rand_score(true_labels, kmeans_labels)

    print(f"\n--- Метрики кластеризации ---")
    print(f"Silhouette Score (Качество разделения векторов): {sil_score:.4f}")
    print(f"Adjusted Rand Score (Качество разделения векторов): {ari_score:.4f}")

    features_2d = analyzer.prepare_tsne_2d(features_30d)
    return features_2d, true_labels, kmeans_labels, dbscan_labels

def plot_clustering_result(features_2d, true_labels, kmeans_labels, dbscan_labels, class_names):
    """
    Отрисовка трех сравнительных графиков t-SNE: 
    Реальные диагнозы, результаты K-Means и результаты DBSCAN.
    """
    # Создаем три графика в один ряд
    fig, axes = plt.subplots(1, 3, figsize=(20,6))
    
    # 1. График реальных классов 
    scatter1 = axes[0].scatter(
        features_2d[:,0], features_2d[:,1],
        c=true_labels, cmap="coolwarm"
    )
    axes[0].set_title("1. Реальные диагнозы", fontsize=14)
    # Настраиваем легенду для реальных классов
    handles1, _ = scatter1.legend_elements()
    axes[0].legend(handles1, class_names, loc="upper right", title="Классы")

    # 2. График результатов K-MEANS
    scatter2 = axes[1].scatter(
        features_2d[:,0], features_2d[:,1],
        c=true_labels, cmap="plasma"
    )
    axes[1].set_title("2. Кластеризация K-Means", fontsize=14)
    # Настраиваем легенду для K-Means
    handles2, _ = scatter2.legend_elements()
    axes[1].legend(handles2, [f"Кластер {i}" for i in np.unique(kmeans_labels)], loc="upper right", title="Классы")

    # 3. График результатов DBSCAN (С выделением шума)
    scatter3 = axes[2].scatter(
        features_2d[:, 0], features_2d[:, 1], 
        c=dbscan_labels, cmap='Set1', alpha=0.6, edgecolors='none', s=30
    )
    axes[2].set_title("3. Кластеризация DBSCAN", fontsize=14, fontweight='bold')
    
    # Настраиваем легенду для DBSCAN, учитывая потенциальный шум (-1)
    unique_dbscan = np.unique(dbscan_labels)
    dbscan_names = [f"Кластер {i}" if i != -1 else "ШУМ / Аномалии" for i in unique_dbscan]
    handles3, _ = scatter3.legend_elements()
    axes[2].legend(handles3, dbscan_names, loc="upper right", title="Плотность")
    
    # Общая стилизация для всех трех графиков
    for ax in axes:
        ax.set_xlabel("t-SNE Component 1", fontsize=10)
        ax.set_ylabel("t-SNE Component 2", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.suptitle("Анализ скрытых признаков легких (ResNet18 + PCA + t-SNE)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "tsne_clustering.png")

    plt.savefig(save_path, dpi=300, bbox_inches = "tight")
    plt.close()
    print(f"График кластеризации сохранен в файл: {save_path}")


if (__name__ == "__main__"):
    features_2d, true_labels, kmeans_labels, dbscan_labels = run_analysis()

    plot_clustering_result(
        features_2d=features_2d,
        true_labels=true_labels,
        kmeans_labels=kmeans_labels,
        dbscan_labels=dbscan_labels,
        class_names=["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamos Cell Carcinoma"]
    )