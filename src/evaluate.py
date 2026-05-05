"""
evaluate.py
───────────
Load a trained PointNet2Seg checkpoint and evaluate it on the test split.

Reports:
  - Per-class IoU  (terrain, anomaly)
  - mIoU
  - Confusion matrix (point-level)

Usage:
    python evaluate.py \\
        --checkpoint ../checkpoints/pointnet2seg_segmentation_best.pth \\
        --data_dir   ../data \\
        --split      test
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

from dataset import LidarAnomalyDataset
from models.pointnet2 import PointNet2Seg
from train import compute_iou, make_seg_batch


def parse_args():
    p = argparse.ArgumentParser("Evaluate PointNet2Seg on test split")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pth weights file.")
    p.add_argument("--data_dir",   default="../data",
                   help="Root dataset directory (contains train/ val/ test/).")
    p.add_argument("--split",      default="test",
                   choices=["train", "val", "test"],
                   help="Dataset split to evaluate on.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_points", type=int, default=2048)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--export_viz", type=int, default=0,
                   help="Number of scenes to export as .ply for visual comparison.")
    return p.parse_args()

import os
from pathlib import Path

def write_ply(path, xyz, rgb):
    with open(path, "wb") as f:
        f.write(b"ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(b"end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n".encode("ascii"))



def print_confusion_matrix(gt: np.ndarray, pred: np.ndarray, class_names: list[str]):
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    for g, p in zip(gt, pred):
        cm[g, p] += 1

    col_w = max(12, max(len(name) for name in class_names) + 2)
    header = " " * col_w + "".join(f"{'Pred: ' + name:>{col_w}}" for name in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = f"{'GT: ' + name:>{col_w}}" + "".join(f"{cm[i, j]:>{col_w},}" for j in range(n))
        print(row)


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Split      : {args.split}")
    print(f"  Device     : {args.device}")
    print(f"{'='*60}\n")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = LidarAnomalyDataset(
        args.data_dir,
        split=args.split,
        num_points=args.num_points,
        transform=None,
        mode="segmentation",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Loaded {len(dataset)} samples from '{args.split}' split.\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PointNet2Seg(num_classes=args.num_classes).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    import math
    
    # ── Inference ─────────────────────────────────────────────────────────────
    all_gt   = []
    all_pred = []
    viz_count = 0
    viz_dir = Path("evaluation_visuals")
    if args.export_viz > 0:
        viz_dir.mkdir(exist_ok=True)
        print(f"Exporting up to {args.export_viz} visuals to {viz_dir}/")

    with torch.no_grad():
        for i, (points, seg_labels) in enumerate(loader):
            points     = points.to(args.device)
            seg_labels = seg_labels.to(args.device)
            batch      = make_seg_batch(points, seg_labels, args.device)
            logits     = model(batch)                   # (B*N, num_classes)
            preds      = logits.argmax(dim=-1)
            
            # Export Visualizations
            if viz_count < args.export_viz:
                b_pts = points.cpu().numpy()
                b_gt  = seg_labels.cpu().numpy()
                b_prd = preds.view(points.size(0), -1).cpu().numpy()
                
                # Colors: Terrain=Tan, Anomaly=Red
                for b_idx in range(b_pts.shape[0]):
                    if viz_count >= args.export_viz:
                        break
                    
                    p = b_pts[b_idx]
                    g = b_gt[b_idx]
                    pr = b_prd[b_idx]
                    
                    rgb_gt = [(180, 150, 100) if cl == 0 else (255, 40, 40) for cl in g]
                    rgb_pr = [(180, 150, 100) if cl == 0 else (255, 40, 40) for cl in pr]
                    
                    write_ply(viz_dir / f"scene_{viz_count:02d}_gt.ply", p, rgb_gt)
                    write_ply(viz_dir / f"scene_{viz_count:02d}_pred.ply", p, rgb_pr)
                    viz_count += 1
            
            all_gt.extend(batch.y.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    gt   = np.array(all_gt,   dtype=np.int64)
    pred = np.array(all_pred, dtype=np.int64)

    # ── Metrics ───────────────────────────────────────────────────────────────
    class_names = ["terrain", "anomaly"]
    ious, miou  = compute_iou(gt, pred, n_classes=args.num_classes)

    print("─── Per-class IoU ─────────────────────────────────────")
    for name, iou in zip(class_names, ious):
        print(f"  {name:<12}: {iou:.4f}" if not np.isnan(iou) else f"  {name:<12}: N/A (no GT samples)")
    print(f"  {'mIoU':<12}: {miou:.4f}")

    print("\n─── Confusion Matrix (point-level) ────────────────────")
    print_confusion_matrix(gt, pred, class_names)

    # Anomaly-specific stats
    anom_gt   = (gt   == 1)
    anom_pred = (pred == 1)
    tp = int(np.sum(anom_gt & anom_pred))
    fp = int(np.sum(~anom_gt & anom_pred))
    fn = int(np.sum(anom_gt & ~anom_pred))
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n─── Anomaly class point-level stats ───────────────────")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}")


if __name__ == "__main__":
    main()
