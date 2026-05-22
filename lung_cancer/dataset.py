import kagglehub
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Средние значения и стандартные отклонения каналов RGB (посчитаны на обычных изображениях)
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]

# Преобразования для изображения (резайз, обрезка, нормализация)
def get_data_loaders(batch_size=32):
    print("Проверка наличия датасета в кеше Kaggle...")
    dataset_path = kagglehub.dataset_download("mohamedhanyyy/chest-ctscan-images")

    # Формируем абсолютные пути к папкам с данными
    train_dir = os.path.join(dataset_path, "Data/train")
    test_dir = os.path.join(dataset_path, "Data/test")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # Принудильно переводим ЧБ в RGB
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Загурзка данных из папок
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    # Создание загрузчиков данных
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
