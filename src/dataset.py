import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LidarAnomalyDataset(Dataset):
    """
    Dataset for point cloud anomaly detection.

    Supports two tasks:
      mode="classification" — returns (points, scene_label)
            points      : (num_points, 3) float32  XYZ
            scene_label : (1,) float32            0=normal, 1=anomaly

      mode="segmentation"  — returns (points, seg_labels)
            points      : (num_points, 3) float32  XYZ
            seg_labels  : (num_points,) int64      0=terrain, 1=anomaly

    Directory layout expected (produced by split_and_prepare.py):
        data_dir/
          train/
            labels.csv          ← columns: filename, label
            scene_0.npy         ← (N, 3) XYZ
            scene_0_seg.npy     ← (N,) per-point labels  [segmentation only]
            ...
          val/
            ...
          test/
            ...
    """

    def __init__(
        self,
        data_dir: str,
        split: str        = "train",
        num_points: int   = 2048,
        transform         = None,
        mode: str         = "segmentation",   # "classification" | "segmentation"
    ):
        super().__init__()
        self.data_dir   = data_dir
        self.split      = split
        self.num_points = num_points
        self.transform  = transform
        self.mode       = mode

        split_dir   = os.path.join(data_dir, split)
        labels_path = os.path.join(split_dir, "labels.csv")

        self.samples = []
        if os.path.exists(labels_path):
            df = pd.read_csv(labels_path)
            for _, row in df.iterrows():
                npy_path = os.path.join(split_dir, row["filename"])
                seg_path = npy_path.replace(".npy", "_seg.npy")

                if not os.path.exists(npy_path):
                    continue

                if mode == "segmentation" and not os.path.exists(seg_path):
                    raise FileNotFoundError(
                        f"Segmentation labels not found: {seg_path}\n"
                        "Run split_and_prepare.py to generate _seg.npy files."
                    )

                self.samples.append({
                    "path":     npy_path,
                    "seg_path": seg_path,
                    "label":    int(row["label"]),
                })
        else:
            # Fallback: scan for .npy files, no labels
            print(f"Warning: {labels_path} not found. Using dummy labels.")
            for f in sorted(glob.glob(os.path.join(split_dir, "*.npy"))):
                if "_seg" in f:
                    continue
                seg_path = f.replace(".npy", "_seg.npy")
                if mode == "segmentation" and not os.path.exists(seg_path):
                    continue
                self.samples.append({"path": f, "seg_path": seg_path, "label": 0})

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found in {split_dir}. "
                "Ensure split_and_prepare.py has been run successfully."
            )

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, n: int) -> np.ndarray:
        """Return `num_points` indices into a cloud of size n."""
        if n >= self.num_points:
            return np.random.choice(n, self.num_points, replace=False)
        return np.random.choice(n, self.num_points, replace=True)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # Load XYZ
        points = np.load(sample["path"]).astype(np.float32)    # (N, 3)
        choice = self._sample_indices(points.shape[0])
        points = points[choice]

        # Centre the cloud (removes absolute position, model sees relative geometry)
        points = points - np.mean(points, axis=0)

        # Spatial augmentations (XYZ only)
        if self.transform is not None:
            points = self.transform(points)

        pts_tensor = torch.from_numpy(points)   # (num_points, 3)

        # ── Classification ────────────────────────────────────────────────────
        if self.mode == "classification":
            label_tensor = torch.tensor([sample["label"]], dtype=torch.float32)
            return pts_tensor, label_tensor

        # ── Segmentation ──────────────────────────────────────────────────────
        seg = np.load(sample["seg_path"]).astype(np.int64)     # (N,)
        seg = seg[choice]                                        # same indices as XYZ
        seg_tensor = torch.from_numpy(seg)                       # (num_points,) int64
        return pts_tensor, seg_tensor


class PyGDataloaderWrapper:
    """Utility to collate standard (points, labels) batches."""

    @staticmethod
    def collate_fn(batch):
        points_list = []
        labels_list = []
        for p, l in batch:
            points_list.append(p)
            labels_list.append(l)
        batched_points = torch.stack(points_list, dim=0)
        batched_labels = torch.stack(labels_list, dim=0)
        return batched_points, batched_labels
