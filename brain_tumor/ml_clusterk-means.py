import os
import torch
import numpy as np
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_absolute_error, r2_score
import math

# Устройство (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Классы в нужном порядке (glioma первой)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- Загрузка модели ---
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(in_features=512, out_features=4)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model

# --- Преобразования для изображений ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- Извлечение признаков из avgpool ResNet18 ---
def extract_features(model, image_dir):
    features = []
    paths = []
    for cls in class_names:
        folder = os.path.join(image_dir, cls)
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(folder, fname)
                img = Image.open(path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    x = model.conv1(tensor)
                    x = model.bn1(x)
                    x = model.relu(x)
                    x = model.maxpool(x)
                    x = model.layer1(x)
                    x = model.layer2(x)
                    x = model.layer3(x)
                    x = model.layer4(x)
                    x = model.avgpool(x)
                    x = torch.flatten(x, 1)
                features.append(x.cpu().numpy().squeeze())
                paths.append(path)
    return np.vstack(features), paths

# --- Визуализация кластеров PCA + KMeans ---
def visualize_clusters(features, labels):
    pca = PCA(n_components=2)
    pts = pca.fit_transform(features)

    cmap = plt.get_cmap('tab10', len(class_names))
    scatter = plt.scatter(pts[:,0], pts[:,1], c=labels, cmap=cmap, s=30)
    cbar = plt.colorbar(scatter, ticks=np.arange(len(class_names)))
    cbar.ax.set_yticklabels(class_names)

    plt.title("ResNet18 Features + KMeans Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.show()

# --- Анализ качества кластеризации ---
def analyze_clusters(features, labels, image_paths):
    # 1. Силуэтный коэффициент
    sil_score = silhouette_score(features, labels)
    print(f"Silhouette Score: {sil_score:.4f}")

    # 2. Чистота (purity) кластеров
    true_labels = [os.path.basename(os.path.dirname(p)) for p in image_paths]
    cluster_purity = {}
    for cid in sorted(set(labels)):
        idxs = [i for i, lbl in enumerate(labels) if lbl == cid]
        assigned = [true_labels[i] for i in idxs]
        common, count = Counter(assigned).most_common(1)[0]
        purity = count / len(idxs)
        cluster_purity[cid] = (common, purity)

    print("Cluster Purity:")
    for cid, (cls, p) in cluster_purity.items():
        print(f" Cluster {cid}: Majority = {cls}, Purity = {p:.2%}")

    # 3. Показ примеров из каждого кластера
    for cid in sorted(set(labels)):
        idxs = [i for i, lbl in enumerate(labels) if lbl == cid][:5]
        plt.figure(figsize=(12, 3))
        plt.suptitle(f'Cluster {cid} examples (Purity {cluster_purity[cid][1]:.2%})', fontsize=14)
        for i, idx in enumerate(idxs):
            img = Image.open(image_paths[idx]).convert('RGB')
            plt.subplot(1, 5, i+1)
            plt.imshow(img)
            plt.title(os.path.basename(os.path.dirname(image_paths[idx])))
            plt.axis('off')
        plt.show()

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

def compute_metrics(features, labels, image_paths):
    # Истинные метки
    true_labels = [class_names.index(os.path.basename(os.path.dirname(p))) for p in image_paths]

    # --- Silhouette Score ---
    sil_score = silhouette_score(features, labels)
    print(f"Silhouette Score: {sil_score:.4f}")

    # --- ARI: Adjusted Rand Index ---
    ari = adjusted_rand_score(true_labels, labels)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")

    # --- NMI: Normalized Mutual Information ---
    nmi = normalized_mutual_info_score(true_labels, labels)
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")

    # --- FMI: Fowlkes-Mallows Index ---
    fmi = fowlkes_mallows_score(true_labels, labels)
    print(f"Fowlkes-Mallows Index (FMI): {fmi:.4f}")

# --- Основная часть ---
if __name__ == "__main__":
    model_path = "resnet18_brain_tumor_model.pth"
    image_directory = "testing"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    if not os.path.isdir(image_directory):
        raise FileNotFoundError(f"Папка testing не найдена: {image_directory}")

    model = load_model(model_path)
    feats, paths = extract_features(model, image_directory)
    print(f"Извлечено признаков: {feats.shape}")

    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(feats)

    visualize_clusters(feats, labels)
    analyze_clusters(feats, labels, paths)
    compute_metrics(feats, labels, paths)