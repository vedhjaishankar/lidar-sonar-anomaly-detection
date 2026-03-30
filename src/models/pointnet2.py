import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, global_max_pool, radius
from torch_geometric.nn.pool import fps

class SAModule(torch.nn.Module):
    """Set Abstraction Module: Samples points and extracts local features."""
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        # Farthest point sampling
        idx = fps(pos, batch, ratio=self.ratio)
        
        # Radius search (find neighbors in radius r)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
                          
        # Construct graph
        edge_index = torch.stack([col, row], dim=0)
        
        # Apply PointNet Conv
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    """Global Set Abstraction Module: Pools features across the entire point cloud."""
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    """Helper for Multi-Layer Perceptrons."""
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])


class PointNet2(torch.nn.Module):
    """PointNet++ for binary anomaly classification."""
    def __init__(self):
        super(PointNet2, self).__init__()

        # Input features are just xyz coords initially, so x=None
        # Module 1: Sample 50% points, radius 0.2
        self.sa1_module = SAModule(0.5, 0.2, MLP([3 + 3, 64, 64, 128]))
        
        # Module 2: Sample 25% points, radius 0.4
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        
        # Global Module: pool everything
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # Classification Head
        self.lin1 = nn.Linear(1024, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 1) # Single output for binary classification

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        # We assume data is a PyG Data object or Batch with pos and batch indices.
        # x is initial features (could be surface normals), we just use pos initially
        sa0_out = (data.x, data.pos, data.batch)
        
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        
        x, pos, batch = sa3_out

        # Final MLP
        x = F.relu(self.bn1(self.lin1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.lin2(x)))
        x = self.dropout(x)
        x = self.lin3(x)
        
        return x
