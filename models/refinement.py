import torch
import torch.nn as nn
import torch.nn.functional as F

class RefineResBlock(nn.Module):
    def __init__(self, ch: int, use_bn: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(ch) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(ch) if use_bn else nn.Identity()

    def forward(self, x):
        s = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + s, inplace=True)

class RefinementModule(nn.Module):
    def __init__(self, in_ch: int, ch: int, num_residual: int = 8, num_pools: int = 4, use_bn: bool = True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, ch, 1, bias=not use_bn),
            nn.BatchNorm2d(ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        self.num_residual = num_residual
        self.num_pools = num_pools
        self.blocks = nn.ModuleList([RefineResBlock(ch, use_bn=use_bn) for _ in range(num_residual)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.proj(x)
        pool_every = max(1, self.num_residual // self.num_pools)
        pools_done = 0
        for i, blk in enumerate(self.blocks, start=1):
            x = blk(x)
            if (i % pool_every == 0) and (pools_done < self.num_pools):
                x = self.pool(x)
                pools_done += 1
        return x
