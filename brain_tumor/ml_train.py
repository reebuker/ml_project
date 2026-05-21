import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
import os
from torch import nn

# Параметры
batch_size = 32
epochs = 10
learning_rate = 0.001

# Преобразования для изображения (резайз, обрезка, нормализация)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # для ResNet
])

# Загрузка данных из папок
train_data = datasets.ImageFolder(root='training', transform=transform)  # Данные для тренировки
test_data = datasets.ImageFolder(root='testing', transform=transform)    # Данные для тестирования

# Дataloader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Загрузка предобученной модели ResNet18
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=4)  # Изменяем последний слой для 4 классов

# Перенос модели на GPU, если он доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Критерий и оптимизатор
criterion = nn.CrossEntropyLoss()  # Кросс-энтропия для многоклассовой классификации
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Тренировка модели
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Обнуляем градиенты
        outputs = model(inputs)  # Прогоняем через модель
        loss = criterion(outputs, labels)  # Рассчитываем потерю
        
        loss.backward()  # Обратное распространение
        optimizer.step()  # Обновляем параметры модели
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Выводим статистику для каждого эпоxа
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

# Сохранение модели
torch.save(model.state_dict(), 'resnet18_brain_tumor_model.pth')

# Оценка на тестовых данных
model.eval()  # Переводим в режим оценки
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
