import torch
import torch.nn as nn

class MPJPELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        err = torch.norm(pred - gt, dim=-1)
        if self.reduction == "mean":
            return err.mean()
        elif self.reduction == "sum":
            return err.sum()
        return err

class Euclidean3DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpjpe = MPJPELoss("mean")

    def forward(self, pred, gt):
        return self.mpjpe(pred, gt)
