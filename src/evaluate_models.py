import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import get_dataloaders
from model import get_model

def evaluate_model(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return acc, precision, recall, f1


def main():
    _, _, test_loader = get_dataloaders()

    num_classes = len(test_loader.dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiments = [
        {
            "name": "resnet18_head",
            "model_name": "resnet18",
            "path": "models/resnet18_head.pth",
        },
        {
            "name": "resnet18_full",
            "model_name": "resnet18",
            "path": "models/resnet18_full.pth",
        },
        {
            "name": "resnet50_full",
            "model_name": "resnet50",
            "path": "models/resnet50_full.pth",
        },
        {
            "name": "resnet50_scratch",
            "model_name": "resnet50",
            "path": "models/resnet50_scratch.pth",
        },
    ]

    print("\nModel,Accuracy,Precision,Recall,F1")

    for exp in experiments:
        if not os.path.exists(exp["path"]):
            print(f"{exp['name']},FILE_NOT_FOUND,,,,")
            continue

        model = get_model(
            name=exp["model_name"],
            num_classes=num_classes,
            pretrained=False,
            freeze_backbone=False
        )

        model.load_state_dict(torch.load(exp["path"], map_location=device))
        model.to(device)

        acc, precision, recall, f1 = evaluate_model(model, test_loader, device)

        print(
            f"{exp['name']},"
            f"{acc:.4f},"
            f"{precision:.4f},"
            f"{recall:.4f},"
            f"{f1:.4f}"
        )


if __name__ == "__main__":
    main()