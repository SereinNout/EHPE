from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch

JOINT_NAMES_21 = [
    "wrist",
    "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

TW_IDXS = [0, 4, 8, 12, 16, 20]  # wrist + 5 tips

def interhand_skeleton_edges_21() -> List[Tuple[int, int]]:
    edges = []
    wrist = 0
    mcp = [1, 5, 9, 13, 17]
    for j in mcp:
        edges.append((wrist, j))
    chains = [
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ]
    for a, b, c, d in chains:
        edges += [(a, b), (b, c), (c, d)]
    return edges

def build_adjacency(num_joints: int = 21, edges: Optional[List[Tuple[int, int]]] = None,
                    self_loops: bool = True, undirected: bool = True) -> torch.Tensor:
    if edges is None:
        edges = interhand_skeleton_edges_21()
    adj = torch.zeros(num_joints, num_joints, dtype=torch.float32)
    for u, v in edges:
        adj[u, v] = 1.0
        if undirected:
            adj[v, u] = 1.0
    if self_loops:
        adj.fill_diagonal_(1.0)
    return adj

@dataclass
class ZBinConfig:
    depth_bins: int = 8
    rel_depth_range_mm: float = 200.0
    use_wrist_as_root: bool = True

def z_mm_to_zbin(z_mm: torch.Tensor, wrist_z_mm: torch.Tensor, cfg: ZBinConfig) -> torch.Tensor:
    D = cfg.depth_bins
    half = cfg.rel_depth_range_mm / 2.0
    if cfg.use_wrist_as_root:
        rel = z_mm - wrist_z_mm.unsqueeze(-1) if wrist_z_mm.ndim == 1 else (z_mm - wrist_z_mm)
    else:
        rel = z_mm
    rel = torch.clamp(rel, -half, half)
    z01 = (rel + half) / (2.0 * half)
    zbin = z01 * (D - 1)
    return zbin

def zbin_to_relz_mm(zbin: torch.Tensor, cfg: ZBinConfig) -> torch.Tensor:
    D = cfg.depth_bins
    half = cfg.rel_depth_range_mm / 2.0
    z01 = zbin / (D - 1)
    rel = z01 * (2.0 * half) - half
    return rel

@dataclass
class Init21Config:
    finger_ratios: Tuple[float, float, float] = (0.28, 0.62, 0.85)
    thumb_ratios: Tuple[float, float, float] = (0.35, 0.68, 0.88)
    lateral_offsets_px: Tuple[float, float, float, float, float] = (10.0, 6.0, 0.0, -6.0, -10.0)

def _interp_points(root: torch.Tensor, tip: torch.Tensor, ratios: Tuple[float, float, float]) -> torch.Tensor:
    v = tip - root
    pts = []
    for r in ratios:
        pts.append(root + r * v)
    return torch.stack(pts, dim=1)  # [B,3,3]

def init_21_from_tw(wrist_xyz: torch.Tensor, tips_xyz: torch.Tensor, cfg: Init21Config) -> torch.Tensor:
    B = wrist_xyz.shape[0]
    out = torch.zeros(B, 21, 3, device=wrist_xyz.device, dtype=wrist_xyz.dtype)
    out[:, 0] = wrist_xyz
    bases = [1, 5, 9, 13, 17]
    for f in range(5):
        tip = tips_xyz[:, f]
        base = bases[f]
        ratios = cfg.thumb_ratios if f == 0 else cfg.finger_ratios
        mids = _interp_points(wrist_xyz, tip, ratios)
        dx = cfg.lateral_offsets_px[f]
        mids[:, :, 0] = mids[:, :, 0] + dx
        tip2 = tip.clone()
        tip2[:, 0] = tip2[:, 0] + dx
        out[:, base + 0] = mids[:, 0]
        out[:, base + 1] = mids[:, 1]
        out[:, base + 2] = mids[:, 2]
        out[:, base + 3] = tip2
    return out

def project_points(K: torch.Tensor, xyz_cam: torch.Tensor) -> torch.Tensor:
    x = xyz_cam[..., 0]
    y = xyz_cam[..., 1]
    z = xyz_cam[..., 2].clamp(min=1e-6)
    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return torch.stack([u, v], dim=-1)
