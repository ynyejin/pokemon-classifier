from utils import split_dataset
from dataset import get_dataloaders
from model import get_model
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

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
        pretrained=False,
        freeze_backbone=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4. loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5. 학습 loop (일단 간단히)
    EPOCHS = 5

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # 6. 모델 저장
    torch.save(model.state_dict(), "models/resnet50_scratch.pth")
    print("모델 저장 완료!")

    # 7. 성능 평가
    val_acc = evaluate(model, val_loader, device)
    test_acc = evaluate(model, test_loader, device)

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    train()

