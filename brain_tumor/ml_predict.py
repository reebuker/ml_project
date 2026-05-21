import torch
from torchvision import models, transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt

# Классы строго в нужном порядке
class_names = [ 'glioma','meningioma', 'notumor', 'pituitary']

# Функция загрузки модели
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(in_features=512, out_features=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Предсказание изображения
def predict(image_path, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    model.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    
    predicted_idx = predicted.item()
    predicted_class = class_names[predicted_idx]
    return predicted_class, image

# Получение случайного изображения из папки
def get_random_image_from_directory(directory_path):
    random_class = random.choice(class_names)
    class_directory = os.path.join(directory_path, random_class)
    images = [f for f in os.listdir(class_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random_image_name = random.choice(images)
    image_path = os.path.join(class_directory, random_image_name)
    return image_path, random_class

# Основной запуск
if __name__ == "__main__":
    model_path = "resnet18_brain_tumor_model.pth"
    image_directory = "testing"

    model = load_model(model_path)
    image_path, true_class = get_random_image_from_directory(image_directory)
    predicted_class, image = predict(image_path, model)

    print(f"Image: {image_path}")
    print(f"True class: {true_class}")
    print(f"Predicted class: {predicted_class}")
    print(f"{'✅ Correct' if true_class == predicted_class else '❌ Incorrect'}")

    # Визуализация
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class} | True: {true_class}")
    plt.axis('off')
    plt.show()
