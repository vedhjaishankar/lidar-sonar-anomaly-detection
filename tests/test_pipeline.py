import os
import shutil
import numpy as np
import pandas as pd
import torch

from src.dataset import LidarAnomalyDataset
from src.models.pointnet import PointNet, pointnet_loss
from src.models.pointnet2 import PointNet2
from src.train import prepare_pyg_batch

def setup_mock_data(data_dir, num_samples=10, num_points=2048):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    
    os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val'), exist_ok=True)
    
    for split in ['train', 'val']:
        labels_data = []
        for i in range(num_samples):
            # Create random point cloud
            pts = np.random.rand(num_points, 3).astype(np.float32)
            filename = f"scene_{i:03d}.npy"
            np.save(os.path.join(data_dir, split, filename), pts)
            
            # Label
            label = np.random.randint(0, 2)
            labels_data.append({'filename': filename, 'label': label})
            
        df = pd.DataFrame(labels_data)
        df.to_csv(os.path.join(data_dir, split, 'labels.csv'), index=False)
        print(f"Created {num_samples} mock {split} samples in {data_dir}/{split}")

def test_dataset(data_dir):
    print("\n--- Testing LidarAnomalyDataset ---")
    dataset = LidarAnomalyDataset(data_dir, split="train", num_points=1024)
    print(f"Dataset length: {len(dataset)}")
    pts, lbl = dataset[0]
    print(f"Sample pts shape: {pts.shape}, label shape: {lbl.shape}")
    assert pts.shape == (1024, 3)
    assert lbl.shape == (1,)

def test_pointnet(device):
    print("\n--- Testing PointNet Forward Pass ---")
    model = PointNet().to(device)
    batch_pts = torch.rand(4, 1024, 3).to(device)
    labels = torch.randint(0, 2, (4, 1)).float().to(device)
    
    logits, trans, trans_feat = model(batch_pts)
    print(f"Logits shape: {logits.shape}")
    loss = pointnet_loss(logits, labels, trans_feat)
    print(f"Loss computed: {loss.item():.4f}")
    assert logits.shape == (4, 1)

def test_pointnet2(device):
    print("\n--- Testing PointNet++ Forward Pass ---")
    model = PointNet2().to(device)
    batch_pts = torch.rand(4, 1024, 3).to(device)
    labels = torch.randint(0, 2, (4, 1)).float().to(device)
    
    batch = prepare_pyg_batch(batch_pts, labels, device)
    logits = model(batch)
    print(f"Logits shape: {logits.shape}")
    loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
    print(f"Loss computed: {loss.item():.4f}")
    assert logits.shape == (4, 1)

if __name__ == '__main__':
    mock_dir = "mock_data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running tests on device: {device}")
    
    try:
        setup_mock_data(mock_dir)
        test_dataset(mock_dir)
        test_pointnet(device)
        test_pointnet2(device)
        print("\nAll Forward Passes Successful!")
    finally:
        if os.path.exists(mock_dir):
            shutil.rmtree(mock_dir)
