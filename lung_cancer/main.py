import torch


from dataset import get_data_loaders
from model import create_model
from trainer import Trainer

def main():
    train_loader, val_loader, test_loader = get_data_loaders()
    model = create_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Выбрано вычислительное устройство:", device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion
    )

    print("Старт обучения...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10
    )
    print("Обучение завершено!")

if __name__ == "__main__":
    main()