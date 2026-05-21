import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from torchvision import models, transforms
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def load_model(model_path):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(in_features=512, out_features=4)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

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

def analyze_dbscan(labels, features, paths):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"\nDBSCAN: кластеров = {n_clusters}, шумовых точек = {n_noise}")

    true = [class_names.index(os.path.basename(os.path.dirname(p))) for p in paths]

    if len(set(labels)) > 1:
        print(f"Silhouette Score: {silhouette_score(features, labels):.4f}")
        print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(true, labels):.4f}")
        print(f"Normalized Mutual Info (NMI): {normalized_mutual_info_score(true, labels):.4f}")
        print(f"Fowlkes-Mallows Index (FMI): {fowlkes_mallows_score(true, labels):.4f}")
    else:
        print("Все точки в одном кластере — метрики не рассчитываются.")

    # Purity
    true_labels = [os.path.basename(os.path.dirname(p)) for p in paths]
    cluster_purity = {}
    for cid in sorted(set(labels)):
        if cid == -1: continue
        idxs = [i for i, lbl in enumerate(labels) if lbl == cid]
        assigned = [true_labels[i] for i in idxs]
        common, count = Counter(assigned).most_common(1)[0]
        purity = count / len(idxs)
        cluster_purity[cid] = (common, purity)

    print("\nPurity по кластерам:")
    for cid, (cls, p) in cluster_purity.items():
        print(f" Кластер {cid}: Majority = {cls}, Purity = {p:.2%}")

    # Подробный состав кластеров
    print("\nПодробный состав кластеров:")
    for cid in sorted(set(labels)):
        if cid == -1: continue
        idxs = [i for i, lbl in enumerate(labels) if lbl == cid]
        assigned = [true_labels[i] for i in idxs]
        dist = Counter(assigned)
        print(f" Кластер {cid}: {dict(dist)}")

    # Примеры
    for cid in sorted(set(labels)):
        if cid == -1: continue
        majority_class = cluster_purity[cid][0]
        idxs = [i for i, lbl in enumerate(labels) if lbl == cid and true_labels[i] == majority_class][:8]
        plt.figure(figsize=(16, 3))
        plt.suptitle(f'Cluster {cid} (Purity: {cluster_purity[cid][1]:.2%})', fontsize=14)
        for i, idx in enumerate(idxs):
            img = Image.open(paths[idx]).convert('RGB')
            plt.subplot(1, 8, i+1)
            plt.imshow(img)
            plt.title(os.path.basename(os.path.dirname(paths[idx])))
            plt.axis('off')
        plt.tight_layout()
        plt.show()

def visualize_tsne(features, labels):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embedded = tsne.fit_transform(features)

    unique_labels = set(labels)
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))

    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        idxs = np.where(labels == label)
        color = 'k' if label == -1 else cmap(label)
        plt.scatter(embedded[idxs, 0], embedded[idxs, 1], label=f"Cluster {label}", s=20, c=[color])

    plt.title("t-SNE Visualization of DBSCAN Clusters")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.show()

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

    pca_high = PCA(n_components=30, random_state=42)
    features_30d = pca_high.fit_transform(feats)

    dbscan = DBSCAN(eps=5, min_samples=5)
    db_labels = dbscan.fit_predict(features_30d)

    analyze_dbscan(db_labels, features_30d, paths)
    visualize_tsne(features_30d, db_labels)
