import torch
import torchvision
from torch import onnx

device = torch.device("cpu")

# Инициализация модели
model = torchvision.models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 4)

# Загрузка весов
state_dict = torch.load("resnet18_weights.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# 2. Создаем "фиктивный" входной тензор (размер батча=1, 3 канала, размер картинки 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Экспортируем в ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,  # Стандартная стабильная версия опсета
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)
print("Модель успешно конвертирована в model.onnx!")

