import torch
import torch.nn as nn

class HeatmapMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        return self.mse(pred, target)
