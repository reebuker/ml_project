import os
import numpy as np
import matplotlib.pyplot as plt
from clusterer import Clusterer
from collections import Counter
from sklearn.metrics import silhouette_score, adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score

plots_dir = "data/plots/"
features_dir = "data/features/"
history_dir = "data/history"
class_names=["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamos Cell Carcinoma", "Noise"]

def run_analysis(class_names):
    features = np.load(features_dir + "extracted_features.npy")
    true_labels = np.load(features_dir + "extracted_labels.npy")

    analyzer = Clusterer(n_clusters=4, dbscan_eps=8, dbscan_min_samples=10)
    
    # Сжимаем данные до 30 компонент и кластеризуем
    features_30d = analyzer.reduce_dims_pca(features)
    kmeans_labels, dbscan_labels = analyzer.run_clustering(features_30d)

    # Считаем формальные метрики качества разделения
    sil_score = silhouette_score(features_30d, kmeans_labels)
    ari_score = adjusted_rand_score(true_labels, kmeans_labels)
    nmi_score = normalized_mutual_info_score(true_labels, kmeans_labels)
    fms_score = fowlkes_mallows_score(true_labels, kmeans_labels)

    print(f"\n--- Метрики кластеризации K-Means ---")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Adjusted Rand Score: {ari_score:.4f}")
    print(f"Normalized Mutual Info Score: {nmi_score:.4f}")
    print(f"Fowlkes Mallows Score: {fms_score:.4f}")

    cluster_purity = {}
    for cid in sorted(set(kmeans_labels)):
        if cid == -1: continue
        idxs = [i for i, lbl in enumerate(kmeans_labels) if lbl == cid]
        assigned = [true_labels[i] for i in idxs]
        common, count = Counter(assigned).most_common(1)[0]
        purity = count / len(idxs)
        cluster_purity[cid] = (common, purity)

    print("\nPurity по кластерам:")
    for cid, (cls, p) in cluster_purity.items():
        print(f" Кластер {cid}: Majority = {cls}, Purity = {p:.2%}")

    sil_score = silhouette_score(features_30d, dbscan_labels)
    ari_score = adjusted_rand_score(true_labels, dbscan_labels)
    nmi_score = normalized_mutual_info_score(true_labels, dbscan_labels)
    fms_score = fowlkes_mallows_score(true_labels, dbscan_labels)

    print(f"\n--- Метрики кластеризации DBSCAN ---")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Adjusted Rand Score: {ari_score:.4f}")
    print(f"Normalized Mutual Info Score: {nmi_score:.4f}")
    print(f"Fowlkes Mallows Score: {fms_score:.4f}")

    # Считаем чистоту кластеризации
    cluster_purity = {}
    for cid in sorted(set(dbscan_labels)):
        if cid == -1: continue
        idxs = [i for i, lbl in enumerate(dbscan_labels) if lbl == cid]
        assigned = [true_labels[i] for i in idxs]
        common, count = Counter(assigned).most_common(1)[0]
        purity = count / len(idxs)
        cluster_purity[cid] = (common, purity)

    print("\nPurity по кластерам:")
    for cid, (cls, p) in cluster_purity.items():
        print(f" Кластер {cid}: Majority = {cls}, Purity = {p:.2%}")

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

def learning_history():
    """
    Загружает историю из .npy матрицы и сохраняет графики Loss и Accuracy на диск.
    """
    save_path = os.path.join(history_dir, "train_history.npy")
    if not os.path.exists(save_path):
        print(f"[Ошибка] Файл истории не найден по пути: {save_path}")
        return

    # 1. Загружаем матрицу истории
    history_matrix = np.load(save_path)

    # 2. Распиливаем её обратно на вектор лосса и вектор точности
    train_loss = history_matrix[:, 0]
    train_acc = history_matrix[:, 1]
    
    # Создаем массив шагов (итераций) по количеству записанных батчей
    steps = range(1, len(train_loss) + 1)

    # 3. Настраиваем сетку графиков (1 строка, 2 колонки)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # --- График Левый: Лосс ---
    # Рисуем блеклую сырую линию батчей
    axes[0].plot(steps, train_loss, color='royalblue', alpha=0.3, label='Batch Loss')
    # Считаем и рисуем скользящее среднее (тренд), чтобы график не сильно «зубрил»
    if len(train_loss) > 10:
        smooth_loss = np.convolve(train_loss, np.ones(5)/5, mode='valid')
        axes[0].plot(range(5, len(smooth_loss) + 5), smooth_loss, color='darkblue', linewidth=2, label='Loss Trend')
    
    axes[0].set_title('Динамика ошибки (Train Loss Curve)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Итерации (Шаги оптимизатора)')
    axes[0].set_ylabel('Значение лосса')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    # --- График Правый: Точность ---
    # Рисуем блеклую линию сырой точности на батчах
    axes[1].plot(steps, train_acc, color='lightcoral', alpha=0.3, label='Batch Acc')
    # Добавляем тренд для точности
    if len(train_acc) > 10:
        smooth_acc = np.convolve(train_acc, np.ones(5)/5, mode='valid')
        axes[1].plot(range(5, len(smooth_acc) + 5), smooth_acc, color='crimson', linewidth=2, label='Acc Trend')
        
    axes[1].set_title('Динамика точности (Train Accuracy Curve)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Итерации (Шаги оптимизатора)')
    axes[1].set_ylabel('Точность (%)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    # 4. Сохраняем готовую картинку в общую с ПК папку
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[Успех] Графики обучения сгенерированы и сохранены в: {save_path}")


if (__name__ == "__main__"):
    features_2d, true_labels, kmeans_labels, dbscan_labels = run_analysis(class_names)

    plot_clustering_result(
        features_2d=features_2d,
        true_labels=true_labels,
        kmeans_labels=kmeans_labels,
        dbscan_labels=dbscan_labels,
        class_names=class_names
    )

    learning_history()

