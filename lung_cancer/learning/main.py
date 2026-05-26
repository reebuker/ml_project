import torch
import os

from dataset import get_data_loaders
from model import create_model
from trainer import Trainer
from evaluator import Evaluator
from feature_extractor import FeatureExtractor

wts_dir = "data/models/"
features_dir = "data/features/"
history_dir = "data/history/"
learning_rate = 0.0005
epochs=10
batch_size=32

def get_choice(question: str, valid_choices: list[int]) -> int:
    while True:
        choice = int(input(question))
        if choice in valid_choices:
            return choice

def main():
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size)
    model = create_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Выбрано вычислительное устройство:", device)

    mode = get_choice(
        question="Обучить модель / использовать сущ. параметры? [1/2]: ",
        valid_choices=[1,2]
    ) 
    
    # --- Блок обучения ---
    if (mode == 1 or not os.path.exists(wts_dir)):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        trainer = Trainer(
            model=model, device=device, optimizer=optimizer, criterion=criterion
        )

        print("Старт обучения...")
        print(f"Кол-во эпох: {epochs} | Размер батча: {batch_size} | Скорость: {learning_rate}")
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs
        )
        print("Обучение завершено!")

    # --- Блок загрузки ---
    if (mode == 2):
        print("Загружаем веса модели...")
        model.load_state_dict(torch.load(wts_path))

    print("Тестируем модель...")
    evaluator = Evaluator(model=model, device=device)
    labels, preds = evaluator.evaluate(test_loader)
    evaluator.print_metricks(labels, preds)

    mode = get_choice(
        question="Сохранить результат обучения / Продолжить без сохранения? [1/2]: ",
        valid_choices=[1,2]
    )

    if (mode == 1):
        os.makedirs(wts_dir, exist_ok=True)
        wts_path = os.path.join(wts_dir, "resnet18_weights.pth")
        torch.save(model.state_dict(), wts_path)
        print(f"Веса успешно сохранены в {wts_path}!")

        os.makedirs(history_dir, exist_ok=True)
        trainer.save_history(history_dir)
    

    extractor = FeatureExtractor(
        model=model,
        device=device
    )

    print("Выделяем признаки из тренировочных данных...")
    features, labels = extractor.extract(dataloader=train_loader)

    mode = get_choice(
        question="Сохранить выделенные признаки / Продолжить без сохранения? [1/2]: ",
        valid_choices=[1,2]
    )

    if (mode == 1):
        extractor.save_to_disk(features, labels, features_dir)


if __name__ == "__main__":
    main()