import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from attribute_data import TYPE_ORDER, NUM_CLASSES_PER_TYPE


class AttributeResNet50(nn.Module):

    def __init__(self, num_classes_per_type: dict):
        super().__init__()

        # Load pretrained ResNet50
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Get number of input features for the classifier
        in_features = backbone.fc.in_features  # 2048

        # Replace the pretrained head with Identity — we attach our own heads
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # One classification head per attribute group (mirrors FastRCNNPredictor swap)
        self.heads = nn.ModuleDict({
            t: nn.Linear(in_features, num_classes_per_type[t])
            for t in TYPE_ORDER
        })

    def forward(self, x):
        features = self.backbone(x)                          # [B, 2048]
        return {t: self.heads[t](features) for t in TYPE_ORDER}


def create_attribute_model(device):

    model = AttributeResNet50(num_classes_per_type=NUM_CLASSES_PER_TYPE)

    model.to(device)
    return model
