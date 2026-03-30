import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    """Spatial Transformer Network (T-Net) for aligning point clouds."""
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Add identity matrix
        iden = torch.eye(self.k, requires_grad=True).view(1, self.k * self.k).repeat(batch_size, 1)
        if x.is_cuda:
            iden = iden.cuda()
            
        x = x + iden
        x = x.view(-1, self.k, self.k)
        
        return x


class PointNet(nn.Module):
    """Binary Anomaly Classification Model based on PointNet Architecture."""
    def __init__(self):
        super(PointNet, self).__init__()
        
        # Input transform (3x3)
        self.tnet = TNet(k=3)
        
        # First MLP
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Feature transform (64x64)
        self.fstn = TNet(k=64)
        
        # MLP before global max pooling
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        
        # Fully Connected Layers for Classification
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # Single output for binary classification
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x is (B, N, 3). Conv1D expects (B, 3, N)
        if x.size(-1) == 3:
            x = x.transpose(1, 2)
            
        batch_size = x.size(0)
        
        # Input transform
        trans = self.tnet(x)
        x = x.transpose(1, 2)
        x = torch.bmm(x, trans)
        x = x.transpose(1, 2)
        
        # First MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Feature transform
        trans_feat = self.fstn(x)
        x = x.transpose(1, 2)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(1, 2)
        
        # Second MLP
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Global Max Pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Classification Head
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # Linear logit output
        
        # Return logit for BCEWithLogitsLoss along with transformation matrices for optional regularization loss
        return x, trans, trans_feat

def pointnet_loss(logits, targets, trans_feat=None, alpha=0.001):
    """Calculates Binary Cross Entropy Loss and adds optional T-Net regularization."""
    bce = nn.BCEWithLogitsLoss()(logits, targets)
    if trans_feat is not None:
        batch_size, k, _ = trans_feat.size()
        identity = torch.eye(k, device=trans_feat.device).unsqueeze(0).repeat(batch_size, 1, 1)
        mat_diff = torch.bmm(trans_feat, trans_feat.transpose(1, 2)) - identity
        reg_loss = torch.mean(torch.norm(mat_diff, dim=(1, 2)))
        return bce + alpha * reg_loss
    return bce
