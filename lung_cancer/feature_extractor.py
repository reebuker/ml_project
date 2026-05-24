import torch

class FeatureExtractor:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Выключаем полносвязный слой
        self.model.fc = torch.nn.Identity()