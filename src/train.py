import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data, Batch

from dataset import LidarAnomalyDataset
from utils import get_train_transforms, get_val_transforms
from models.pointnet import PointNet, pointnet_loss
from models.pointnet2 import PointNet2, PointNet2Seg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Anomaly Detection — Point Cloud Training")
    p.add_argument("--task",    default="segmentation",
                   choices=["classification", "segmentation"],
                   help="'segmentation': per-point labeling with PointNet2Seg. "
                        "'classification': scene-level label (legacy).")
    p.add_argument("--model",   default="pointnet2seg",
                   choices=["pointnet", "pointnet2", "pointnet2seg"])
    p.add_argument("--data_dir",    default="../data")
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--num_points",  type=int,   default=2048)
    p.add_argument("--class_weight", type=float, default=30.0,
                   help="Weight for the anomaly class in loss function. "
                        "Counteracts class imbalance (~99%% terrain vs 1%% anomaly).")
    p.add_argument("--checkpoint",  default=None,
                   help="Path to weights file for transfer learning.")
    p.add_argument("--save_dir",    default="../checkpoints")
    p.add_argument("--resume",      action="store_true",
                   help="Resume training from the latest checkpoint.")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# IoU metric (for segmentation)
# ---------------------------------------------------------------------------
def compute_iou(gt: np.ndarray, pred: np.ndarray, n_classes: int = 2):
    """
    Compute per-class IoU and mIoU from flat label arrays.

    Returns:
        ious  : list of per-class IoU (NaN if class not present in gt)
        miou  : mean IoU over valid classes
    """
    ious = []
    for cls in range(n_classes):
        tp = int(np.sum((gt == cls) & (pred == cls)))
        fp = int(np.sum((gt != cls) & (pred == cls)))
        fn = int(np.sum((gt == cls) & (pred != cls)))
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float("nan"))
    valid = [v for v in ious if not np.isnan(v)]
    miou  = float(np.mean(valid)) if valid else float("nan")
    return ious, miou


# ---------------------------------------------------------------------------
# PyG batch helpers
# ---------------------------------------------------------------------------
def make_cls_batch(points, labels, device):
    """Classification: PyG batch for PointNet2 (scene-level label)."""
    data_list = [
        Data(pos=points[i], x=points[i], y=labels[i])
        for i in range(points.size(0))
    ]
    return Batch.from_data_list(data_list).to(device)


def make_seg_batch(points, seg_labels, device):
    """
    Segmentation: PyG batch for PointNet2Seg.
    seg_labels: (B, N) int64 per-point labels.
    After batching, batch.y is (B*N,) — the concatenated per-point labels.
    """
    data_list = [
        Data(pos=points[i], x=points[i], y=seg_labels[i].long())
        for i in range(points.size(0))
    ]
    return Batch.from_data_list(data_list).to(device)


# ---------------------------------------------------------------------------
# Segmentation training / validation
# ---------------------------------------------------------------------------
def seg_train_epoch(model, loader, optimizer, device, class_weight, scaler=None):
    model.train()
    total_loss = 0.0
    all_gt, all_pred = [], []

    # Upweight anomaly class to counter severe class imbalance
    weight    = torch.tensor([1.0, class_weight], device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for points, seg_labels in loader:
        points     = points.to(device)      # (B, N, 3)
        seg_labels = seg_labels.to(device)  # (B, N)

        optimizer.zero_grad()
        batch = make_seg_batch(points, seg_labels, device)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(batch)               # (B*N, 2)
                loss = criterion(logits, batch.y)
                
            scaler.scale(loss).backward()
            
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * points.size(0)
        preds = logits.argmax(dim=-1)
        all_gt.extend(batch.y.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())

    ious, miou = compute_iou(np.array(all_gt), np.array(all_pred))
    return total_loss / len(loader.dataset), ious, miou



def seg_validate(model, loader, device, class_weight):
    model.eval()
    total_loss = 0.0
    all_gt, all_pred = [], []

    weight    = torch.tensor([1.0, class_weight], device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)

    with torch.no_grad():
        for points, seg_labels in loader:
            points     = points.to(device)
            seg_labels = seg_labels.to(device)
            batch  = make_seg_batch(points, seg_labels, device)
            logits = model(batch)
            loss   = criterion(logits, batch.y)

            total_loss += loss.item() * points.size(0)
            preds = logits.argmax(dim=-1)
            all_gt.extend(batch.y.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    ious, miou = compute_iou(np.array(all_gt), np.array(all_pred))
    return total_loss / len(loader.dataset), ious, miou


# ---------------------------------------------------------------------------
# Classification training / validation (unchanged logic, kept for legacy)
# ---------------------------------------------------------------------------
def cls_train_epoch(model, loader, optimizer, device, model_type, class_weight):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pos_weight = torch.tensor([class_weight], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for points, labels in loader:
        points = points.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if model_type == "pointnet":
            logits, trans, trans_feat = model(points)
            loss = pointnet_loss(logits, labels, trans_feat)
        else:
            batch  = make_cls_batch(points, labels, device)
            logits = model(batch)
            loss   = criterion(logits, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * points.size(0)
        preds = torch.sigmoid(logits) > 0.5
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader.dataset), acc


def cls_validate(model, loader, device, model_type, class_weight):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pos_weight = torch.tensor([class_weight], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)
            if model_type == "pointnet":
                logits, _, _ = model(points)
                loss = criterion(logits, labels)
            else:
                batch  = make_cls_batch(points, labels, device)
                logits = model(batch)
                loss   = criterion(logits, batch.y)

            total_loss += loss.item() * points.size(0)
            preds = torch.sigmoid(logits) > 0.5
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader.dataset)
    metrics = {
        "acc":       accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
    }
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Task        : {args.task}")
    print(f"  Model       : {args.model}")
    print(f"  Device      : {args.device}")
    print(f"  class_weight: {args.class_weight}")
    print(f"{'='*60}\n")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    try:
        train_ds = LidarAnomalyDataset(
            args.data_dir, split="train",
            num_points=args.num_points,
            transform=get_train_transforms(),
            mode=args.task,
        )
        val_ds = LidarAnomalyDataset(
            args.data_dir, split="val",
            num_points=args.num_points,
            transform=get_val_transforms(),
            mode=args.task,
        )
        
        # SPEED OPTIMIZATION: Use workers and pin_memory for faster NVIDIA training
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=True, drop_last=True, 
            num_workers=2, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size,
            shuffle=False, 
            num_workers=2, pin_memory=True,
        )
        print(f"Train: {len(train_ds)} samples   Val: {len(val_ds)} samples\n")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model == "pointnet":
        model = PointNet().to(args.device)
    elif args.model == "pointnet2":
        model = PointNet2().to(args.device)
    elif args.model == "pointnet2seg":
        model = PointNet2Seg(num_classes=2).to(args.device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # ── Resume / Checkpoint Loading ──────────────────────────────────────────
    start_epoch = 1
    best_val_score = -1.0
    latest_ckpt_path = os.path.join(args.save_dir, f"{args.model}_{args.task}_latest.pth")
    history_csv_path = os.path.join(args.save_dir, f"{args.model}_{args.task}_history.csv")

    # AMP Scaler Initialization
    scaler = torch.cuda.amp.GradScaler() if args.device == "cuda" else None

    if args.resume and os.path.exists(latest_ckpt_path):
        print(f"Resuming from latest checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_score = checkpoint.get('best_val_score', -1.0)
        print(f"  => Starting from epoch {start_epoch}")
    elif args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading weights (transfer learning branch): {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    # Prep CSV Logger
    if not os.path.exists(history_csv_path) or start_epoch == 1:
        with open(history_csv_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_score,val_anomaly_iou\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    model_name_best = f"{args.model}_{args.task}_best.pth"

    for epoch in range(start_epoch, args.epochs + 1):

        if args.task == "segmentation":
            train_loss, train_ious, train_miou = seg_train_epoch(
                model, train_loader, optimizer, args.device, args.class_weight, scaler=scaler
            )
            val_loss, val_ious, val_miou = seg_validate(
                model, val_loader, args.device, args.class_weight
            )
            scheduler.step()

            terrain_iou = val_ious[0] if not np.isnan(val_ious[0]) else 0.0
            anomaly_iou = val_ious[1] if len(val_ious) > 1 and not np.isnan(val_ious[1]) else 0.0
            val_score   = val_miou

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {val_miou:.4f} | AnomalyIoU: {anomaly_iou:.4f}")

        else:  # classification
            train_loss, train_acc = cls_train_epoch(
                model, train_loader, optimizer,
                args.device, args.model, args.class_weight,
            )
            val_loss, val_metrics = cls_validate(
                model, val_loader, args.device, args.model, args.class_weight,
            )
            scheduler.step()
            val_score   = val_metrics["f1"]
            anomaly_iou = 0.0

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {val_score:.4f}")

        # ── Logging & Checkpointing ──────────────────────────────────────────
        
        # 1. Update CSV
        with open(history_csv_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_score:.6f},{anomaly_iou:.6f}\n")

        # 2. Save Latest (Crash Recovery)
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_score': best_val_score,
        }
        if scaler is not None:
            ckpt['scaler_state_dict'] = scaler.state_dict()
        torch.save(ckpt, latest_ckpt_path)

        # 3. Save Best (Production)
        if val_score > best_val_score:
            best_val_score = val_score
            save_path = os.path.join(args.save_dir, model_name_best)
            torch.save(model.state_dict(), save_path)
            print(f"  --> New Best! Saved weights: {save_path}")

    print(f"\nTraining complete. Best val score: {best_val_score:.4f}")


    print(f"\nTraining complete. Best val score: {best_val_score:.4f}")


if __name__ == "__main__":
    main()
