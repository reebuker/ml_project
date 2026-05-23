import torch

class Evaluator:
    def __init__(self, model, device):
        self.model = model.to(device)