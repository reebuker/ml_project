import torch

class Trainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {"train_loss": [], "val_loss": []}

    # def train_epoch(self, dataloader):
    
