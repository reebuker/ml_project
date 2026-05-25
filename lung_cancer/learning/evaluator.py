import torch
from sklearn.metrics import accuracy_score, classification_report

class Evaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def evaluate(self, test_loader):
        """ Запуск инференса на тестовых данных """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                # переносить labels на устройство необяз., тк функция потерь не считается
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        return all_labels, all_preds

    def print_metricks(self, y_true, y_pred):
        """ Вывод тестовых метрик """
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc * 100:.2f}%")
        print("\nClassification Report:\n",
              classification_report(y_true, y_pred))



