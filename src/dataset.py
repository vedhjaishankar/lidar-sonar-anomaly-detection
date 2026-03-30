import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class LidarAnomalyDataset(Dataset):
    """
    Dataset for binary anomaly classification of static lidar scenes.
    Assumes data is stored as .npy files, with an accompanying labels.csv.
    """
    def __init__(self, data_dir, split="train", num_points=2048, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.num_points = num_points
        self.transform = transform
        
        # Determine paths
        split_dir = os.path.join(data_dir, split)
        labels_path = os.path.join(split_dir, "labels.csv")
        
        # Load labels or mock them if labels.csv doesn't exist yet
        self.samples = []
        if os.path.exists(labels_path):
            df = pd.read_csv(labels_path)
            # Assuming labels.csv has columns: filename (e.g., 'scene_001.npy'), label (0 or 1)
            for _, row in df.iterrows():
                self.samples.append({
                    'path': os.path.join(split_dir, row['filename']),
                    'label': int(row['label'])
                })
        else:
            print(f"Warning: {labels_path} not found. Searching for .npy files and assigning dummy labels.")
            npy_files = sorted(glob.glob(os.path.join(split_dir, "*.npy")))
            for f in npy_files:
                self.samples.append({
                    'path': f,
                    'label': 0  # Dummy label
                })
                
    def __len__(self):
        return len(self.samples)

    def _sample_points(self, points):
        """Randomly sample exactly num_points from the point cloud."""
        n = points.shape[0]
        if n >= self.num_points:
            choice = np.random.choice(n, self.num_points, replace=False)
        else:
            # If fewer points, sample with replacement
            choice = np.random.choice(n, self.num_points, replace=True)
        return points[choice, :]
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        points = np.load(sample['path']).astype(np.float32) # Expected shape (N, 3)
        label = sample['label']
        
        # Standardize number of points
        points = self._sample_points(points)
        
        # Center the point cloud
        points = points - np.mean(points, axis=0)
        
        # Apply optional data augmentations (e.g. random rotation, jitter)
        if self.transform:
            points = self.transform(points)
            
        points_tensor = torch.from_numpy(points) # Shape (num_points, 3)
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        # For PointNet (standard PyTorch), returns (points, label)
        # For PointNet++ (PyTorch Geometric), it often expects a Data object.
        # We will return the raw tensors and let the training loop handle adaptation for PyG.
        return points_tensor, label_tensor

class PyGDataloaderWrapper:
    """Utility to convert dataset outputs to PyTorch Geometric Batch objects if needed."""
    @staticmethod
    def collate_fn(batch):
        # batch is a list of tuples (points_tensor, label_tensor)
        points_list = []
        labels_list = []
        for p, l in batch:
            points_list.append(p)
            labels_list.append(l)
            
        # points shape for PyG: (Total_Points, 3). Batch vector tracks which point belongs to which cloud.
        # Or standard batched tensor (Batch, num_points, 3)
        batched_points = torch.stack(points_list, dim=0) # (B, N, 3)
        batched_labels = torch.stack(labels_list, dim=0) # (B, 1)
        
        return batched_points, batched_labels
