import timm
import torch.nn as nn

def get_model(name="resnet50", num_classes=150, pretrained=True, freeze_backbone=False):

    model = timm.create_model(name, pretrained=pretrained)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # classifier 교체
    if hasattr(model, "fc"):  # ResNet
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif hasattr(model, "classifier"):  # EfficientNet
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif hasattr(model, "head"):  # ViT
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    return model