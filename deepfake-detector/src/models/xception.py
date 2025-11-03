import torch
import torch.nn as nn
from torchvision import models

class Xception(nn.Module):
    """
    A lightweight Xception-like network using torchvision's pretrained model
    or a similar CNN backbone (for simplicity, we’ll use EfficientNet_B0).
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(Xception, self).__init__()

        # Use a modern backbone since Xception isn't in torchvision
        # You can replace with real Xception implementation if you prefer.
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # Replace final classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
