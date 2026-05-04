from utils import split_dataset
from dataset import get_dataloaders
from model import get_model
from tqdm import tqdm

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def calculate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss


def save_learning_curve(train_losses, val_losses, save_path):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve - ResNet50 Scratch")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def train():

    # 1. 데이터 split (이미 했다면 False로)
    DO_SPLIT = False
    if DO_SPLIT:
        split_dataset(
            raw_dir="data/merged_300",
            output_dir="data/processed"
        )

    # 2. dataloader
    train_loader, val_loader, test_loader = get_dataloaders()

    num_classes = len(train_loader.dataset.classes)
    print("클래스 수:", num_classes)

    # 3. 모델
    model = get_model(
        name="resnet50",
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4. loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # loss 기록용 리스트
    train_losses = []
    val_losses = []

    # 5. 학습 loop
    EPOCHS = 5

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        total_val_loss = calculate_loss(model, val_loader, criterion, device)

        train_losses.append(total_train_loss)
        val_losses.append(total_val_loss)

        print(
            f"Epoch {epoch+1}, "
            f"Train Loss: {total_train_loss:.4f}, "
            f"Validation Loss: {total_val_loss:.4f}"
        )

    # 6. 모델 저장
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/resnet50_full.pth")
    print("모델 저장 완료!")

    # 7. loss 기록 저장
    os.makedirs("assets", exist_ok=True)

    with open("assets/loss_resnet50_full.json", "w") as f:
        json.dump(
            {
                "train_loss": train_losses,
                "val_loss": val_losses
            },
            f,
            indent=4
        )

    save_learning_curve(
        train_losses,
        val_losses,
        "assets/learning_curve_resnet50_full.png"
    )

    print("Loss 기록 및 learning curve 저장 완료!")

    # 8. 성능 평가
    val_acc = evaluate(model, val_loader, device)
    test_acc = evaluate(model, test_loader, device)

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    train()