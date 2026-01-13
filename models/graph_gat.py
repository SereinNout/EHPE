
import torch
import torch.nn as nn
import torch.nn.functional as F

def knn_graph(coords, k=8):
    """coords: (B,J,3) -> dynamic neighbor indices per node (B,J,k)."""
    with torch.no_grad():
        B,J,_ = coords.shape
        # pairwise distance
        d2 = torch.cdist(coords, coords, p=2)  # (B,J,J)
        knn_idx = d2.topk(k+1, largest=False).indices[...,1:]  # exclude self
        return knn_idx  # (B,J,k)

class MHGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=8, dropout=0.0):
        super().__init__()
        self.heads=heads; self.out_dim=out_dim
        self.lin_q = nn.Linear(in_dim, out_dim*heads, bias=False)
        self.lin_k = nn.Linear(in_dim, out_dim*heads, bias=False)
        self.lin_v = nn.Linear(in_dim, out_dim*heads, bias=False)
        self.proj = nn.Linear(out_dim*heads, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, nbr_idx):
        # x: (B,J,C); nbr_idx: (B,J,K)
        B,J,C = x.shape; K = nbr_idx.shape[-1]; H=self.heads; D=self.out_dim
        q = self.lin_q(x).view(B,J,H,D)                  # (B,J,H,D)
        k = self.lin_k(x).view(B,J,H,D)                  # (B,J,H,D)
        v = self.lin_v(x).view(B,J,H,D)                  # (B,J,H,D)
        # gather neighbors for k,v
        k_nbr = k.gather(1, nbr_idx.unsqueeze(2).unsqueeze(3).expand(B,J,H,D))  # (B,J,K,H,D) but shape dims mismatch, fix via indexing
        v_nbr = v.gather(1, nbr_idx.unsqueeze(2).unsqueeze(3).expand(B,J,H,D))
        # The above gather is tricky; do manual indexing
        k_nbr = torch.stack([k[b, nbr_idx[b]] for b in range(B)], dim=0)  # (B,J,K,H,D)
        v_nbr = torch.stack([v[b, nbr_idx[b]] for b in range(B)], dim=0)  # (B,J,K,H,D)

        # attention: q · k_nbr^T over K neighbors
        attn = (q.unsqueeze(2) * k_nbr).sum(dim=-1) / (D ** 0.5)  # (B,J,K,H)
        attn = self.leaky(attn)
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)
        out = (attn.unsqueeze(-1) * v_nbr).sum(dim=2)    # (B,J,H,D)
        out = out.view(B,J,H*D)
        return self.proj(out)                             # (B,J,D)

class SPI(nn.Module):
    def __init__(self, in_dim=256, hidden=256, heads=8, layers=2, num_joints=21, knn_k=8, fixed_adj=False):
        super().__init__()
        self.layers_num = layers
        self.knn_k = knn_k
        self.fixed_adj = fixed_adj
        self.mh = nn.ModuleList([MHGATLayer(in_dim if i==0 else hidden, hidden, heads=heads) for i in range(layers)])
        self.fc = nn.Linear(hidden, 3)
        # cross-layer skip
        self.skip = nn.ModuleList([nn.Linear(in_dim if i==0 else hidden, hidden, bias=False) for i in range(layers)])
        self.skip0 = nn.Linear(in_dim, hidden, bias=False)

        if fixed_adj:
            # star + finger chains
            A = torch.zeros(num_joints, num_joints)
            for f in range(5):
                for k in range(4):
                    i = 1 + f*4 + k
                    if i+1 < num_joints:
                        A[i, i+1] = 1; A[i+1, i] = 1
            A[:,0] = 1; A[0,:] = 1
            self.register_buffer("adj", A)

    def forward(self, joint_feats):
        # joint_feats: (B,J, in_dim). Its first 3 dims should be (x,y,z) so we can KNN on them.
        B,J,C = joint_feats.shape
        h0 = joint_feats
        h = h0
        alpha_stats = []
        for l in range(self.layers_num):
            if self.fixed_adj:
                # build neighbor idx from fixed adjacency -> take topK from fixed edges
                with torch.no_grad():
                    nbr_list = []
                    for b in range(B):
                        idxs = []
                        for i in range(J):
                            nbrs = torch.nonzero(self.adj[i]>0).flatten()
                            if nbrs.numel()==0:
                                nbrs = torch.tensor([i], device=h.device)
                            if nbrs.numel() > self.knn_k:
                                nbrs = nbrs[:self.knn_k]
                            elif nbrs.numel() < self.knn_k:
                                # pad with self
                                pad = nbrs.new_full((self.knn_k - nbrs.numel(),), i)
                                nbrs = torch.cat([nbrs, pad], dim=0)
                            idxs.append(nbrs)
                        idxs = torch.stack(idxs, dim=0)  # (J,K)
                        nbr_list.append(idxs)
                    nbr_idx = torch.stack(nbr_list, dim=0)  # (B,J,K)
            else:
                # dynamic KNN on coords (first 3 dims)
                coords = h[..., :3]
                nbr_idx = knn_graph(coords, k=self.knn_k)

            out = self.mh[l](h, nbr_idx)                                 # (B,J,hidden)
            h = torch.relu(out + self.skip[l](h) + self.skip0(h0))        # cross-layer fuse
            alpha_stats.append(h.abs().mean())

        coords = self.fc(h)
        alphas = torch.stack(alpha_stats).mean()
        return coords, alphas
