import torch
import torch.nn as nn

class L1Regularizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, params):
        reg = 0.0
        for p in params:
            reg = reg + p.abs().sum()
        return reg

class EdgeStabilityLoss(nn.Module):
    def __init__(self, mode="l2"):
        super().__init__()
        self.mode = mode

    def forward(self, alphas):
        loss = 0.0
        for a in alphas:
            if self.mode == "l1":
                loss = loss + (a - 1.0).abs().mean()
            else:
                loss = loss + ((a - 1.0) ** 2).mean()
        return loss / max(1, len(alphas))
