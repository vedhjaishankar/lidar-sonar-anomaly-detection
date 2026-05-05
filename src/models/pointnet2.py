"""
PointNet++ — Classification and Segmentation.

Implements the full PointNet++ architecture from Qi et al. (2017) including:
  - Set Abstraction (SA) modules with Farthest Point Sampling + ball query
  - Feature Propagation (FP) modules with k-NN interpolation + skip connections

Two models are provided:
  PointNet2     — scene-level binary classification (legacy, kept for reference)
  PointNet2Seg  — per-point binary segmentation (primary model for this project)

Radii are tuned for the seabed scan geometry:
  Point cloud footprint : ~22 m × 22 m  (90° FOV at ~11 m depth)
  Anomaly object size   : 1.5–2.5 m
  SA1 radius 0.5 m  → captures surface texture / local flatness
  SA2 radius 2.0 m  → captures object-scale geometry (cube, sphere, cylinder)
  SA3 global          → scene-level context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, global_max_pool, radius, knn_interpolate
from torch_geometric.nn.pool import fps


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def MLP(channels):
    """1-D MLP: Linear → ReLU → BatchNorm, stacked for each pair of channels."""
    return nn.Sequential(*[
        nn.Sequential(
            nn.Linear(channels[i - 1], channels[i]),
            nn.ReLU(),
            nn.BatchNorm1d(channels[i]),
        )
        for i in range(1, len(channels))
    ])


class SAModule(nn.Module):
    """
    Set Abstraction module.

    1. Farthest Point Sampling (FPS) to subsample `ratio` fraction of points.
    2. Ball query of radius `r` to find local neighbourhoods.
    3. PointNetConv to extract per-point features within each neighbourhood.
    """
    def __init__(self, ratio: float, r: float, nn: nn.Module):
        super().__init__()
        self.ratio = ratio
        self.r     = r
        self.conv  = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        # Farthest Point Sampling
        idx = fps(pos, batch, ratio=self.ratio)

        # Ball query: for each sampled point find neighbours within radius r
        row, col = radius(
            pos, pos[idx], self.r,
            batch, batch[idx],
            max_num_neighbors=64,
        )
        edge_index = torch.stack([col, row], dim=0)

        # Aggregate neighbourhood features
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)

        return x, pos[idx], batch[idx]


class GlobalSAModule(nn.Module):
    """
    Global Set Abstraction: pools all remaining points to a single global feature
    vector per point cloud (equivalent to PointNet's global max pool).
    """
    def __init__(self, nn: nn.Module):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x   = self.nn(torch.cat([x, pos], dim=1))
        x   = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))          # one "virtual" point at origin
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(nn.Module):
    """
    Feature Propagation module (decoder step).

    Upsamples coarse features to a finer resolution using inverse-distance
    weighted k-NN interpolation, then concatenates skip-connection features
    from the corresponding encoder SA layer and passes through a MLP.

    Args:
        k  : number of nearest neighbours for interpolation.
        nn : MLP applied after concatenation with skip features.
    """
    def __init__(self, k: int, nn: nn.Module):
        super().__init__()
        self.k  = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        # Interpolate coarse features (pos) → fine positions (pos_skip)
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)

        # Concatenate with encoder skip connection (if any)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)

        x = self.nn(x)
        return x, pos_skip, batch_skip


# ---------------------------------------------------------------------------
# Classification model (kept for backward compatibility / ablation)
# ---------------------------------------------------------------------------

class PointNet2(nn.Module):
    """PointNet++ scene-level binary classification (anomaly present / absent)."""

    def __init__(self):
        super().__init__()

        # Encoder — radii corrected for seabed scan scale
        self.sa1 = SAModule(0.50, 0.5, MLP([3 + 3,   64,  64, 128]))
        self.sa2 = SAModule(0.25, 2.0, MLP([128 + 3, 256, 256, 512]))
        self.sa3 = GlobalSAModule(MLP([512 + 3, 512, 1024]))

        # Classification head
        self.lin1    = nn.Linear(1024, 512)
        self.lin2    = nn.Linear(512,  256)
        self.lin3    = nn.Linear(256,    1)   # single logit for BCEWithLogitsLoss
        self.bn1     = nn.BatchNorm1d(512)
        self.bn2     = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1(*sa0_out)
        sa2_out = self.sa2(*sa1_out)
        sa3_out = self.sa3(*sa2_out)

        x, _, _ = sa3_out
        x = F.relu(self.bn1(self.lin1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.lin2(x)))
        x = self.dropout(x)
        return self.lin3(x)                   # (B, 1)


# ---------------------------------------------------------------------------
# Segmentation model
# ---------------------------------------------------------------------------

class PointNet2Seg(nn.Module):
    """
    PointNet++ per-point binary segmentation.

    Encoder:
        SA1 (r=0.5 m)  → local surface texture, flatness vs roughness
        SA2 (r=2.0 m)  → object-scale geometry
        SA3 (global)    → scene context

    Decoder:
        FP3: SA3 (1024) ──interpolate──► SA2 res, cat SA2 skip (512) → MLP → 512
        FP2: FP3 (512)  ──interpolate──► SA1 res, cat SA1 skip (128) → MLP → 256
        FP1: FP2 (256)  ──interpolate──► all N pts, no SA0 skip      → MLP → 128

    Head:
        Linear(128 → 128) → BN → ReLU → Dropout → Linear(128 → num_classes)
        Returns raw logits of shape (total_points_in_batch, num_classes).
        Use nn.CrossEntropyLoss with class weights outside this module.

    Args:
        num_classes : number of output classes (default 2: terrain, anomaly).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

        # ── Encoder ────────────────────────────────────────────────────────
        self.sa1 = SAModule(0.50, 0.5, MLP([3 + 3,      64,  64, 128]))
        self.sa2 = SAModule(0.25, 2.0, MLP([128 + 3,   256, 256, 512]))
        self.sa3 = GlobalSAModule(MLP([512 + 3,         512,      1024]))

        # ── Decoder (Feature Propagation) ──────────────────────────────────
        # FP3: global (1024) + SA2 skip (512) → 1536 in
        self.fp3 = FPModule(k=1, nn=MLP([1024 + 512,   512, 512]))
        # FP2: FP3 out (512) + SA1 skip (128) → 640 in
        self.fp2 = FPModule(k=3, nn=MLP([512 + 128,    256, 256]))
        # FP1: FP2 out (256) + SA0 skip (3 XYZ coords)→ 259 in
        self.fp1 = FPModule(k=3, nn=MLP([256 + 3,      128, 128]))

        # ── Per-point prediction head ───────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, data):
        """
        Args:
            data : PyG Data/Batch with fields:
                     pos   (total_N, 3)  — point coordinates
                     x     None          — no additional input features
                     batch (total_N,)    — batch index per point

        Returns:
            logits (total_N, num_classes) — raw class logits (no softmax).
        """
        # ── Encode ──────────────────────────────────────────────────────────
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1(*sa0_out)   # ~N/2  points, 128-dim
        sa2_out = self.sa2(*sa1_out)   # ~N/8  points, 512-dim
        sa3_out = self.sa3(*sa2_out)   #  1    point/cloud, 1024-dim

        # ── Decode ──────────────────────────────────────────────────────────
        fp3_out = self.fp3(*sa3_out, *sa2_out)  # → SA2 resolution, 512-dim
        fp2_out = self.fp2(*fp3_out, *sa1_out)  # → SA1 resolution, 256-dim
        fp1_out = self.fp1(*fp2_out, *sa0_out)  # → all N points,   128-dim

        # ── Per-point prediction ─────────────────────────────────────────────
        x = self.head(fp1_out[0])               # (total_N, num_classes)
        return x
