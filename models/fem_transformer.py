import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.lnq = nn.LayerNorm(dim)
        self.lnk = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        qn = self.lnq(q)
        kvn = self.lnk(kv)
        out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        q = q + out
        q = q + self.mlp(self.ln2(q))
        return q

class FEM(nn.Module):
    def __init__(self, dim: int = 512, layers: int = 2, heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.sa_blocks = nn.ModuleList([TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(layers)])
        self.ca_blocks = nn.ModuleList([CrossAttentionBlock(dim, heads, mlp_ratio, dropout) for _ in range(layers)])

    def forward(self, joint_tokens: torch.Tensor, img_tokens: torch.Tensor):
        x = joint_tokens
        for blk in self.sa_blocks:
            x = blk(x)
        for blk in self.ca_blocks:
            x = blk(x, img_tokens)
        return x
