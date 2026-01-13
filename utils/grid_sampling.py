import torch
import torch.nn.functional as F

def sample_feature_at_points(feat: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """
    feat: [B, C, H, W]
    xy:   [B, J, 2] in normalized grid coords [-1, 1]
    return: [B, J, C]
    """
    grid = xy.unsqueeze(2)  # [B, J, 1, 2]
    sampled = F.grid_sample(feat, grid, mode="bilinear", align_corners=True)  # [B, C, J, 1]
    sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # [B, J, C]
    return sampled

def xy_to_grid(xy_px: torch.Tensor, img_size: int) -> torch.Tensor:
    """
    xy_px: [B, J, 2] in pixel coords [0, img_size-1]
    return normalized [-1, 1] for grid_sample
    """
    x = xy_px[..., 0]
    y = xy_px[..., 1]
    gx = (x / (img_size - 1)) * 2 - 1
    gy = (y / (img_size - 1)) * 2 - 1
    return torch.stack([gx, gy], dim=-1)
