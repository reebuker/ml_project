import torch

class Trainer:
    def __init__(self, model, device, optimizer, criterion):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, dataloader):
        """Одна эпоха обучения."""
        self.model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
        return running_loss / len(dataloader.dataset)

    def val_epoch(self, dataloader):
        """Одна эпоха валидации."""
        self.model.eval()
        running_loss = 0.0

        # Отключаем вычисление градиентов
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
        return running_loss / len(dataloader.dataset)


    def fit(self, train_loader, val_loader, epochs):
        """ Основной цикл обучения """
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")
        return self.history
    
