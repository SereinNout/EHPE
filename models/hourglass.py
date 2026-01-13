import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        s = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.bn3(self.conv3(x))
        return F.relu(x + s, inplace=True)

class Hourglass(nn.Module):
    def __init__(self, depth: int, ch: int):
        super().__init__()
        self.up1 = Residual(ch, ch)
        self.pool = nn.MaxPool2d(2)
        self.low1 = Residual(ch, ch)
        self.low2 = Hourglass(depth - 1, ch) if depth > 1 else Residual(ch, ch)
        self.low3 = Residual(ch, ch)

    def forward(self, x):
        up1 = self.up1(x)
        low = self.pool(x)
        low = self.low1(low)
        low = self.low2(low)
        low = self.low3(low)
        up2 = F.interpolate(low, scale_factor=2, mode="nearest")
        return up1 + up2

class StackedHourglass(nn.Module):
    def __init__(self, in_ch: int, hg_ch: int, depth: int, stacks: int, out_joints: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, hg_ch, 1, bias=False),
            nn.BatchNorm2d(hg_ch),
            nn.ReLU(inplace=True),
            Residual(hg_ch, hg_ch),
        )
        self.hgs = nn.ModuleList([Hourglass(depth, hg_ch) for _ in range(stacks)])
        self.features = nn.ModuleList([Residual(hg_ch, hg_ch) for _ in range(stacks)])
        self.outs = nn.ModuleList([nn.Conv2d(hg_ch, out_joints, 1) for _ in range(stacks)])
        self.merge_features = nn.ModuleList([nn.Conv2d(hg_ch, hg_ch, 1, bias=False) for _ in range(stacks - 1)])
        self.merge_preds = nn.ModuleList([nn.Conv2d(out_joints, hg_ch, 1, bias=False) for _ in range(stacks - 1)])

    def forward(self, x):
        x = self.pre(x)
        outputs = []
        for i, hg in enumerate(self.hgs):
            y = hg(x)
            y = self.features[i](y)
            pred = self.outs[i](y)
            outputs.append(pred)
            if i < len(self.hgs) - 1:
                x = x + self.merge_features[i](y) + self.merge_preds[i](pred)
        return outputs
