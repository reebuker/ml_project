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
    valid_dir = os.path.join(dataset_path, "Data/valid")
    test_dir = os.path.join(dataset_path, "Data/test")

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),

        # Аугментация данных
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        
        # Форматируем для модели
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    normal_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # Форматируем для модели
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Загурзка данных из папок
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(root=valid_dir, transform=normal_transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=normal_transform)

    # Создание загрузчиков данных
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"Выборки успешно загружены из папок:\n"
          f"-> Train: {len(train_data)} картинок\n"
          f"-> Valid: {len(valid_data)} картинок\n"
          f"-> Test: {len(test_data)} картинок")

    return train_loader, valid_loader, test_loader
