import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from dataset import LidarAnomalyDataset
from train import prepare_pyg_batch
from models.pointnet import PointNet
from models.pointnet2 import PointNet2

def parse_args():
    parser = argparse.ArgumentParser("Evaluate Anomaly Detection Point Cloud Model")
    parser.add_argument('--model', type=str, required=True, choices=['pointnet', 'pointnet2'])
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to weights file')
    parser.add_argument('--data_dir', type=str, default='../data', help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Evaluating {args.model} using checkpoint {args.checkpoint} on device {args.device}")

    try:
        dataset = LidarAnomalyDataset(args.data_dir, split=args.split, num_points=args.num_points, transform=None)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    except Exception as e:
        print(f"Could not load dataset from {args.data_dir}: {e}")
        return

    # Model Setup
    kwargs = {}
    if args.model == 'pointnet':
        model = PointNet(**kwargs).to(args.device)
    else:
        model = PointNet2(**kwargs).to(args.device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for points, labels in loader:
            points = points.to(args.device)
            labels = labels.to(args.device)

            if args.model == 'pointnet':
                logits, _, _ = model(points)
            else:
                batch = prepare_pyg_batch(points, labels, args.device)
                logits = model(batch)

            preds = torch.sigmoid(logits) > 0.5
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n--- Evaluation Results ---")
    print(f"Model: {args.model}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"[{cm[0][0]} {cm[0][1]}]\n[{cm[1][0]} {cm[1][1]}]")

if __name__ == '__main__':
    main()
