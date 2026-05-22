import torch
from torch import nn
from torchvision import models

def create_model(num_of_classes=4):
    model = models.resnet18(pretrained=True)

    # Меняем финальный слой, чтобы иметь на выходе 4 класса
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_of_classes)

    return model
