import torch
import torchvision
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class_names=["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamos Cell Carcinoma"]

# Средние значения и стандартные отклонения каналов RGB (посчитаны на обычных изображениях)
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]

class ResnetClassifier:
    def __init__(self, wts_path, num_classes):
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Инициализация модели
        self.model = torchvision.models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, num_classes)

        # Загрузка весов
        state_dict = torch.load(wts_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Инициализация grad-cam для подсветки областей
        target_layers = [self.model.layer4[-1]]
        self.cam = GradCAM(model=self.model, target_layers=target_layers)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            # Форматируем для модели
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)
        ])

    def predict_and_visualize(self, image_path: str):
        img = Image.open(image_path)
        resized_img = img.resize((224, 224))

        # Картинка в массив numpy [0,1] для grad-cam
        rgb_img_np = np.array(resized_img, dtype=np.float32) / 255.0
        
        img_t = self.transform(img)
        img_t = torch.unsqueeze(img_t, 0)

        with torch.no_grad():
            outputs = self.model(img_t)
            # dim=1, так как у нас есть размерность батча [1, 4]
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        pred_dict = {self.class_names[i]: float(prob.item()) for i, prob in enumerate(probs)}

        # Генерируем маску важности пикселей
        grayscale_cam = self.cam(input_tensor=img_t, targets=None)
        grayscale_cam = grayscale_cam[0, :]

        # Накладываем тепловую карту на исходное изображение
        # colormap=1 соответствует OpenCV COLORMAP_JET (синий -> зеленый -> красный)
        # Самые важные для модели зоны окрасятся в КРАСНЫЙ цвет
        cam_image = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True, colormap=1)
        result_pil_img = Image.fromarray((cam_image*255).astype(np.uint8))

        return pred_dict, result_pil_img

