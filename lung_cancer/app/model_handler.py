import torch
import torchvision
import PIL as pil

class_names=["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamos Cell Carcinoma"]
wts_path = "data/models/resnet18_weights.pth"

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

        state_dict = torch.load(wts_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            # Форматируем для модели
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)
        ])

    def predict(self, image_path: str) -> dict:
        img = pil.Image.open(image_path)
        
        img_t = self.transform(img)
        img_t = torch.unsqueeze(img_t, 0)

        with torch.no_grad():
            outputs = self.model(img_t)
            # dim=1, так как у нас есть размерность батча [1, 4]
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]

        return {
            self.classes[i]: float(prob.item()) for i, prob in enumerate(probs)
        }

