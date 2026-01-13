
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from .hourglass import StackedHourglass
from .refinement import RefinementModule
from .graph_gat import SPI_GAT
from .fem_transformer import FEM

from ..utils.grid_sampling import sample_feature_at_points, xy_to_grid
from ..utils.geometry import build_adjacency, interhand_skeleton_edges_21, init_21_from_tw, Init21Config



def soft_argmax_2d(heat: torch.Tensor, beta: float = 1.0):
    """
    heat: [B, J, H, W]
    return xy: [B, J, 2] in heatmap coordinates (0..W-1, 0..H-1)
    """
    B, J, H, W = heat.shape
    heat = heat.view(B, J, -1)
    prob = F.softmax(heat * beta, dim=-1)

    xs = torch.linspace(0, W - 1, W, device=heat.device)
    ys = torch.linspace(0, H - 1, H, device=heat.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # [HW,2]

    xy = torch.einsum("bjn,nc->bjc", prob, grid)
    return xy


def soft_argmax_1d(logits: torch.Tensor, beta: float = 1.0):
    """
    logits: [B, J, D]
    return z: [B, J] in bin coordinate 0..D-1
    """
    prob = F.softmax(logits * beta, dim=-1)
    ds = torch.linspace(0, logits.shape[-1] - 1, logits.shape[-1], device=logits.device)
    z = torch.einsum("bjd,d->bj", prob, ds)
    return z


class TWStage(nn.Module):

    def __init__(
        self,
        backbone_out_ch: int,
        hg_ch: int,
        hg_depth: int,
        hg_stacks: int,
        tw_joints: int,
        heatmap_size: int,
        refine_ch: int,
        num_residual: int,
        num_pools: int,
        depth_bins: int,
        beta: float = 1.0,
        use_bn: bool = True,
    ):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.depth_bins = depth_bins
        self.tw_joints = tw_joints
        self.beta = beta

        self.upsample_to_hm = nn.Sequential(
            nn.Conv2d(backbone_out_ch, hg_ch, 1, bias=False),
            nn.BatchNorm2d(hg_ch),
            nn.ReLU(inplace=True),
        )

        self.hg = StackedHourglass(
            in_ch=hg_ch,
            hg_ch=hg_ch,
            depth=hg_depth,
            stacks=hg_stacks,
            out_joints=tw_joints,
        )

        self.refine = RefinementModule(
            in_ch=hg_ch + tw_joints,
            ch=refine_ch,
            num_residual=num_residual,
            num_pools=num_pools,
            use_bn=use_bn,
        )

        self.to_25d = nn.Conv2d(refine_ch, depth_bins * tw_joints, 1)

    def forward(self, feat8: torch.Tensor):
        # project backbone features to hg channel and resize to heatmap size
        x = self.upsample_to_hm(feat8)
        x = F.interpolate(x, size=(self.heatmap_size, self.heatmap_size), mode="bilinear", align_corners=False)

        heatmaps = self.hg(x)
        heat2d = heatmaps[-1]  # [B,6,Hh,Wh]

        r_in = torch.cat([x, heat2d], dim=1)  # [B, hg_ch+6, Hh, Wh]
        latent = self.refine(r_in)

        # bring latent back to heatmap resolution
        latent = F.interpolate(latent, size=(self.heatmap_size, self.heatmap_size), mode="bilinear", align_corners=False)

        logits_25d = self.to_25d(latent)  # [B, 6*D, Hh, Wh]
        B, _, Hh, Wh = logits_25d.shape
        logits_25d = logits_25d.view(B, self.tw_joints, self.depth_bins, Hh, Wh)  # [B,6,D,Hh,Wh]

        # 2D marginal heatmap
        heat2d_from_25d = logits_25d.logsumexp(dim=2)  # [B,6,Hh,Wh]

        # soft-argmax xy
        xy = soft_argmax_2d(heat2d_from_25d, beta=self.beta)  # [B,6,2] in hm coords

        # depth marginal: [B,6,D]
        z_logits = logits_25d.flatten(3).logsumexp(dim=-1)  # [B,6,D]
        z = soft_argmax_1d(z_logits, beta=self.beta).unsqueeze(-1)  # [B,6,1]

        coords_25d = torch.cat([xy, z], dim=-1)  # [B,6,3] (x_hm, y_hm, zbin)
        return heat2d, logits_25d, coords_25d



class PGStage(nn.Module):

    def __init__(
        self,
        num_joints: int = 21,
        node_dim: int = 512,
        gat_layers: int = 2,
        gat_heads: int = 8,
        gat_dropout: float = 0.1,
        fem_layers: int = 2,
        fem_heads: int = 8,
        fem_mlp_ratio: int = 4,
        fem_dropout: float = 0.1,
        learnable_fusion: bool = True,
    ):
        super().__init__()
        self.node_proj = nn.Linear(3 + node_dim, node_dim)

        self.spi = SPI_GAT(dim=node_dim, heads=gat_heads, layers=gat_layers, dropout=gat_dropout)
        self.fem = FEM(dim=node_dim, layers=fem_layers, heads=fem_heads, mlp_ratio=fem_mlp_ratio, dropout=fem_dropout)

        if learnable_fusion:
            self.omega_g = nn.Parameter(torch.eye(num_joints))
            self.omega_e = nn.Parameter(torch.ones(num_joints, 1))
        else:
            self.register_buffer("omega_g", torch.eye(num_joints))
            self.register_buffer("omega_e", torch.ones(num_joints, 1))

        self.out_head = nn.Linear(node_dim, 3)

    def forward(self, node_feat: torch.Tensor, img_tokens: torch.Tensor, adj: torch.Tensor):

        x = self.node_proj(node_feat)          # [B,J,512]
        spi_out, alphas = self.spi(x, adj)     # [B,J,512], attention list
        fem_out = self.fem(x, img_tokens)      # [B,J,512]

        fused_spi = torch.einsum("jk,bkd->bjd", self.omega_g, spi_out)
        fused_fem = fem_out * self.omega_e.unsqueeze(0)
        fused = fused_spi + fused_fem

        pred = self.out_head(fused)           # [B,J,3]
        return pred, alphas



class EHPE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.img_size = cfg["data"]["img_size"]
        self.heatmap_size = cfg["data"]["heatmap_size"]
        self.num_joints = cfg["data"]["num_joints"]
        self.tw_joints = cfg["data"]["tw_joints"]
        self.depth_bins = cfg["data"]["depth_bins"]

        self.backbone = build_backbone(cfg["model"]["backbone"]["name"], cfg["model"]["backbone"]["pretrained"])

        self.tw = TWStage(
            backbone_out_ch=cfg["model"]["backbone"]["out_channels"],
            hg_ch=cfg["model"]["hourglass"]["channels"],
            hg_depth=cfg["model"]["hourglass"]["depth"],
            hg_stacks=cfg["model"]["hourglass"]["stacks"],
            tw_joints=self.tw_joints,
            heatmap_size=self.heatmap_size,
            refine_ch=cfg["model"]["refinement"]["channels"],
            num_residual=cfg["model"]["refinement"]["num_residual"],
            num_pools=cfg["model"]["refinement"]["num_pools"],
            depth_bins=self.depth_bins,
            beta=cfg["model"]["tw_head"]["softargmax_beta"],
            use_bn=cfg["model"]["refinement"]["use_bn"],
        )

        self.pg = PGStage(
            num_joints=self.num_joints,
            node_dim=cfg["model"]["pg"]["node_dim"],
            gat_layers=cfg["model"]["pg"]["gat"]["layers"],
            gat_heads=cfg["model"]["pg"]["gat"]["heads"],
            gat_dropout=cfg["model"]["pg"]["gat"]["dropout"],
            fem_layers=cfg["model"]["pg"]["fem"]["layers"],
            fem_heads=cfg["model"]["pg"]["fem"]["heads"],
            fem_mlp_ratio=cfg["model"]["pg"]["fem"]["mlp_ratio"],
            fem_dropout=cfg["model"]["pg"]["fem"]["dropout"],
            learnable_fusion=cfg["model"]["pg"]["fusion"]["learnable"],
        )

        edges = interhand_skeleton_edges_21()
        adj = build_adjacency(num_joints=self.num_joints, edges=edges, self_loops=True, undirected=True)
        self.register_buffer("adj", adj)
        self.init21_cfg = Init21Config()

        self.img_token_proj = nn.Sequential(
            nn.Conv2d(cfg["model"]["backbone"]["out_channels"], cfg["model"]["pg"]["node_dim"], 1, bias=False),
            nn.BatchNorm2d(cfg["model"]["pg"]["node_dim"]),
            nn.ReLU(inplace=True),
        )

    def forward(self, img: torch.Tensor):
        feat8 = self.backbone(img)  # [B,2048,8,8]

        # TW-stage
        heat2d, heat25d, tw_coords = self.tw(feat8)  # [B,6,3] in heatmap coords (x,y,zbin)

        # image tokens for FEM
        img_feat = self.img_token_proj(feat8)                # [B,512,8,8]
        img_tokens = img_feat.flatten(2).transpose(1, 2)     # [B,64,512]

        # Convert TW coords to pixel coords (init21 expects x,y in pixels; z is zbin)
        xy_hm = tw_coords[..., :2]                            # [B,6,2]
        xy_px = xy_hm * (self.img_size - 1) / (self.heatmap_size - 1)
        zbin = tw_coords[..., 2:3]                            # [B,6,1]
        tw_pxz = torch.cat([xy_px, zbin], dim=-1)             # [B,6,3]

        # TW order: [wrist, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        wrist = tw_pxz[:, 0]                                  # [B,3]
        tips = tw_pxz[:, 1:6]                                 # [B,5,3]
        init21 = init_21_from_tw(wrist, tips, self.init21_cfg) # [B,21,3] (px,px,zbin)

        # Sample per-joint visual features at init positions
        xy_px_21 = init21[..., :2]                             # [B,21,2]
        grid = xy_to_grid(xy_px_21, self.img_size)             # [B,21,2] normalized [-1,1]

        img_feat_up = F.interpolate(img_feat, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        sampled = sample_feature_at_points(img_feat_up, grid)  # [B,21,512]

        node_feat = torch.cat([init21, sampled], dim=-1)        # [B,21,3+512]

        # PG-stage
        pg_pred, alphas = self.pg(node_feat, img_tokens, self.adj)

        return {
            "feat8": feat8,
            "tw_heat2d": heat2d,
            "tw_heat25d": heat25d,
            "tw_coords_25d": tw_coords,
            "init21": init21,
            "pg_pred": pg_pred,
            "alphas": alphas,
        }
