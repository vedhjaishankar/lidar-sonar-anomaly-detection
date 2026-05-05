"""
tests/test_pipeline.py
───────────────────────
Smoke tests for the full training pipeline.

Tests:
  - LidarAnomalyDataset  (classification mode)
  - LidarAnomalyDataset  (segmentation mode)
  - PointNet             forward pass + loss
  - PointNet2Seg         forward pass + segmentation loss
  - make_seg_batch       PyG batch construction
  - compute_iou          metric correctness

Run from repo root:
    python -m pytest tests/ -v
or:
    python tests/test_pipeline.py
"""

import os
import shutil
import numpy as np
import torch
import torch.nn as nn

# ── allow imports from src/ ───────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import LidarAnomalyDataset
from models.pointnet import PointNet, pointnet_loss
from models.pointnet2 import PointNet2Seg
from train import make_seg_batch, compute_iou


# ── Mock data helpers ─────────────────────────────────────────────────────────
MOCK_DIR   = os.path.join(os.path.dirname(__file__), "..", "mock_data")
N_SAMPLES  = 8
N_POINTS   = 1024
NUM_POINTS = 512    # sampled size used in dataset


def _setup_mock_data():
    """Create synthetic .npy and _seg.npy files with a minimal labels.csv."""
    if os.path.exists(MOCK_DIR):
        shutil.rmtree(MOCK_DIR)

    for split in ("train", "val"):
        split_dir = os.path.join(MOCK_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

        rows = []
        for i in range(N_SAMPLES):
            fname    = f"scene_{i:03d}.npy"
            seg_name = f"scene_{i:03d}_seg.npy"
            xyz      = np.random.rand(N_POINTS, 3).astype(np.float32)
            # Roughly 1% anomaly points — mirrors real data distribution
            seg      = np.zeros(N_POINTS, dtype=np.int32)
            n_anom   = max(1, N_POINTS // 100)
            seg[:n_anom] = 1
            np.random.shuffle(seg)

            np.save(os.path.join(split_dir, fname),    xyz)
            np.save(os.path.join(split_dir, seg_name), seg)
            rows.append(f"{fname},{i % 2}")   # alternating labels

        with open(os.path.join(split_dir, "labels.csv"), "w") as f:
            f.write("filename,label\n")
            f.write("\n".join(rows) + "\n")


def _teardown_mock_data():
    if os.path.exists(MOCK_DIR):
        shutil.rmtree(MOCK_DIR)


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_dataset_classification():
    print("\n─── LidarAnomalyDataset (classification) ───────────────")
    ds  = LidarAnomalyDataset(MOCK_DIR, split="train",
                               num_points=NUM_POINTS, mode="classification")
    assert len(ds) == N_SAMPLES, f"Expected {N_SAMPLES}, got {len(ds)}"
    pts, lbl = ds[0]
    assert pts.shape  == (NUM_POINTS, 3), f"Bad pts shape: {pts.shape}"
    assert lbl.shape  == (1,),            f"Bad label shape: {lbl.shape}"
    assert pts.dtype  == torch.float32
    assert lbl.dtype  == torch.float32
    print(f"  pts shape: {tuple(pts.shape)}  label: {lbl.item()}")
    print("  PASS")


def test_dataset_segmentation():
    print("\n─── LidarAnomalyDataset (segmentation) ─────────────────")
    ds  = LidarAnomalyDataset(MOCK_DIR, split="train",
                               num_points=NUM_POINTS, mode="segmentation")
    assert len(ds) == N_SAMPLES
    pts, seg = ds[0]
    assert pts.shape == (NUM_POINTS, 3),  f"Bad pts shape: {pts.shape}"
    assert seg.shape == (NUM_POINTS,),    f"Bad seg shape: {seg.shape}"
    assert seg.dtype == torch.int64,      f"Bad seg dtype: {seg.dtype}"
    unique = seg.unique().tolist()
    assert all(v in (0, 1) for v in unique), f"Unexpected label values: {unique}"
    print(f"  pts shape: {tuple(pts.shape)}  seg shape: {tuple(seg.shape)}")
    print(f"  Label values: {unique}")
    print("  PASS")


def test_pointnet_forward():
    print("\n─── PointNet forward pass ───────────────────────────────")
    device = "cpu"
    model  = PointNet().to(device)
    B, N   = 4, NUM_POINTS
    pts    = torch.rand(B, N, 3, device=device)
    labels = torch.randint(0, 2, (B, 1), dtype=torch.float32, device=device)

    logits, trans, trans_feat = model(pts)
    assert logits.shape == (B, 1), f"Bad logits shape: {logits.shape}"

    loss = pointnet_loss(logits, labels, trans_feat)
    assert loss.item() > 0
    print(f"  logits shape: {tuple(logits.shape)}   loss: {loss.item():.4f}")
    print("  PASS")


def test_pointnet2seg_forward():
    print("\n─── PointNet2Seg forward pass ───────────────────────────")
    device = "cpu"
    model  = PointNet2Seg(num_classes=2).to(device)
    B, N   = 2, NUM_POINTS    # small batch for speed on CPU
    pts    = torch.rand(B, N, 3, device=device)
    seg    = torch.randint(0, 2, (B, N), dtype=torch.int64, device=device)

    batch  = make_seg_batch(pts, seg, device)
    logits = model(batch)                           # (B*N, 2)

    assert logits.shape == (B * N, 2), f"Bad logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in logits"

    weight    = torch.tensor([1.0, 30.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    loss      = criterion(logits, batch.y)
    assert loss.item() > 0
    print(f"  logits shape: {tuple(logits.shape)}   loss: {loss.item():.4f}")
    print("  PASS")


def test_compute_iou():
    print("\n─── compute_iou metric ──────────────────────────────────")
    # Perfect predictions
    gt   = np.array([0, 0, 1, 1, 0, 1])
    pred = np.array([0, 0, 1, 1, 0, 1])
    ious, miou = compute_iou(gt, pred)
    assert abs(ious[0] - 1.0) < 1e-6, f"terrain IoU should be 1.0, got {ious[0]}"
    assert abs(ious[1] - 1.0) < 1e-6, f"anomaly IoU should be 1.0, got {ious[1]}"
    assert abs(miou    - 1.0) < 1e-6

    # All-terrain predictions (worst case for anomaly)
    pred_zeros = np.zeros_like(gt)
    ious2, miou2 = compute_iou(gt, pred_zeros)
    assert ious2[1] == 0.0, f"anomaly IoU should be 0.0, got {ious2[1]}"
    print(f"  Perfect: terrain={ious[0]:.3f} anomaly={ious[1]:.3f} mIoU={miou:.3f}")
    print(f"  Worst:   terrain={ious2[0]:.3f} anomaly={ious2[1]:.3f} mIoU={miou2:.3f}")
    print("  PASS")


# ── Runner ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Setting up mock data...")
    _setup_mock_data()

    try:
        test_dataset_classification()
        test_dataset_segmentation()
        test_compute_iou()
        test_pointnet_forward()
        # Note: PointNet2Seg requires torch_geometric — skip if not installed
        try:
            test_pointnet2seg_forward()
        except ImportError as e:
            print(f"\n  [SKIP] PointNet2Seg test — torch_geometric not available: {e}")

        print("\n" + "="*55)
        print("  All tests passed.")
        print("="*55)
    finally:
        _teardown_mock_data()
        print("Mock data cleaned up.")
