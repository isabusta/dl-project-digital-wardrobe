import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from attribute_data import TYPE_ORDER, NUM_CLASSES_PER_TYPE


class AttributeEfficientNetB0(nn.Module):

    def __init__(self, num_classes_per_type: dict):
        super().__init__()

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        in_features = backbone.classifier[1].in_features  # 1280

        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.heads = nn.ModuleDict({
            t: nn.Linear(in_features, num_classes_per_type[t])
            for t in TYPE_ORDER
        })

    def forward(self, x):
        features = self.backbone(x)  # [B, 1280]
        return {t: self.heads[t](features) for t in TYPE_ORDER}


def create_attribute_efficientnet_model(device):
    model = AttributeEfficientNetB0(num_classes_per_type=NUM_CLASSES_PER_TYPE)
    model.to(device)
    return model
