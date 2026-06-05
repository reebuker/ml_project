import torch
import os, sys
import torchvision
import cv2
import numpy as np
from PIL import Image
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class_names=["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamos Cell Carcinoma"]

# Средние значения и стандартные отклонения каналов RGB (посчитаны на обычных изображениях)
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]

def get_resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class ResnetClassifier:
    def __init__(self, wts_path, num_classes):
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Инициализация модели
        self.model = torchvision.models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, num_classes)

        # Загрузка весов
        wts_path = get_resource_path(wts_path)
        state_dict = torch.load(wts_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Инициализация grad-cam для подсветки областей
        target_layers = [self.model.layer4[-1]]
        # self.cam = GradCAM(model=self.model, target_layers=target_layers)
        self.cam = LayerCAM(model=self.model, target_layers=target_layers)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            # Форматируем для модели
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)
        ])

    def predict_and_visualize(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        rgb_img_np = np.array(img, dtype=np.float32) / 255.0
    
        img_t = self.transform(img)
        img_t = torch.unsqueeze(img_t, 0)

        # Инференс модели
        outputs = self.model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_dict = {self.class_names[i]: float(prob.item()) for i, prob in enumerate(probs)}

        # Находим индекс класса с наибольшей вероятностью
        highest_pred_idx = int(torch.argmax(probs).item())

        # 1. ТАРГЕТИНГ: Строго указываем класс для расчета градиентов
        targets = [ClassifierOutputTarget(highest_pred_idx)]

        # Генерируем маску важности для конкретного класса
        grayscale_cam = self.cam(input_tensor=img_t, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Разметка [0, 1]

        # 2. Мягкое медицинское подсвечивание (Вместо жесткого порога 0.8)
        cam_heatmap = np.uint8(255 * grayscale_cam)
        cam_heatmap = cv2.applyColorMap(cam_heatmap, cv2.COLORMAP_JET)
    
        # CV2 использует BGR, переводим в RGB для PIL/Tkinter
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        cam_heatmap = np.float32(cam_heatmap) / 255.0

        # Смешиваем оригинальный КТ-снимок и тепловую карту (0.6 исходник, 0.4 подсветка)
        cam_image = 0.6 * rgb_img_np + 0.4 * cam_heatmap
        # Защита от выхода за границы [0, 1] после сложения
        cam_image = np.clip(cam_image, 0, 1)

        # Конвертируем обратно в изображение для Tkinter
        result_pil_img = Image.fromarray((cam_image * 255).astype(np.uint8))

        return pred_dict, result_pil_img
