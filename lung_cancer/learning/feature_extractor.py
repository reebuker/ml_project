import torch
import os
import numpy as np

class FeatureExtractor:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Выключаем полносвязный слой, данные проходят насквозь
        self.model.fc = torch.nn.Identity()
    
    def extract(self, dataloader):
        """ Прогон датасета через сверточную базу """
        features_list = []
        labels_list = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)

                outputs = self.model(images)

                features_list.append(outputs.cpu().numpy())
                labels_list.append(labels.numpy())

        all_features = np.vstack(features_list)
        all_labels = np.concatenate(labels_list)
    
        return all_features, all_labels

    def save_to_disk(self, features, labels, output_dir):
        """ Сохранение извлеченных матриц в бинарном формате """
        os.makedirs(output_dir, exist_ok=True)

        features_path = os.path.join(output_dir, "extracted_features.npy")
        labels_path = os.path.join(output_dir, "extracted_labels.npy")
        
        np.save(features_path, features);
        np.save(labels_path, labels)

        print(f"Признаки сохранены в {features_path} {features.shape}")
        print(f"Метки сохранены в {labels_path} {labels.shape}")
