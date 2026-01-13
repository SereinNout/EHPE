import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = resnet50(weights=weights)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def build_backbone(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "resnet50":
        return ResNet50Backbone(pretrained=pretrained)
    raise NotImplementedError(f"Backbone {name} not implemented.")
