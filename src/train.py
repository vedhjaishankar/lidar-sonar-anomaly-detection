import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import Data, Batch

from dataset import LidarAnomalyDataset
from utils import get_train_transforms, get_val_transforms
from models.pointnet import PointNet, pointnet_loss
from models.pointnet2 import PointNet2

def parse_args():
    parser = argparse.ArgumentParser("Training Anomaly Detection Point Cloud Model")
    parser.add_argument('--model', type=str, default='pointnet', choices=['pointnet', 'pointnet2'])
    parser.add_argument('--data_dir', type=str, default='../data', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--checkpoint', type=str, default=None, help='Weights to load for transfer learning')
    parser.add_argument('--save_dir', type=str, default='../checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def prepare_pyg_batch(points, labels, device):
    """Converts a batch of points and labels into a PyTorch Geometric Batch."""
    data_list = []
    for i in range(points.size(0)):
        # points[i] is (N, 3), labels[i] is (1,)
        d = Data(pos=points[i], x=None, y=labels[i])
        data_list.append(d)
    batch = Batch.from_data_list(data_list)
    return batch.to(device)

def train_epoch(model, loader, optimizer, device, model_type):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    for points, labels in loader:
        points = points.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        if model_type == 'pointnet':
            # expects (B, N, 3)
            logits, trans, trans_feat = model(points)
            loss = pointnet_loss(logits, labels, trans_feat)
        else:
            # pointnet2 expects a PyG Data batch
            batch = prepare_pyg_batch(points, labels, device)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * points.size(0)
        
        preds = torch.sigmoid(logits) > 0.5
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader.dataset), acc

def validate(model, loader, device, model_type):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)
            
            if model_type == 'pointnet':
                logits, _, _ = model(points)
                loss = criterion(logits, labels)
            else:
                batch = prepare_pyg_batch(points, labels, device)
                logits = model(batch)
                loss = criterion(logits, batch.y)
                
            total_loss += loss.item() * points.size(0)
            
            preds = torch.sigmoid(logits) > 0.5
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
    avg_loss = total_loss / len(loader.dataset)
    metrics = {
        'acc': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }
    return avg_loss, metrics

def main():
    args = parse_args()
    print(f"Using device: {args.device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Intentionally ignoring missing data directories to allow for dry-run if desired,
    # but normally we want the dataloaders here. We will just wrap it in a try-except.
    try:
        train_dataset = LidarAnomalyDataset(args.data_dir, split="train", num_points=args.num_points, transform=get_train_transforms())
        val_dataset = LidarAnomalyDataset(args.data_dir, split="val", num_points=args.num_points, transform=get_val_transforms())
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"Loaded {len(train_dataset)} training max samples, {len(val_dataset)} validation samples.")
    except Exception as e:
        print(f"Could not load datasets from {args.data_dir}: {e}. Ensure the directory structure exists.")
        return

    # Model Setup
    if args.model == 'pointnet':
        model = PointNet().to(args.device)
    else:
        model = PointNet2().to(args.device)
        
    # Transfer Learning
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    best_val_f1 = -1.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, args.device, args.model)
        scheduler.step()
        
        val_loss, val_metrics = validate(model, val_loader, args.device, args.model)
        
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['acc']:.4f} | Val F1: {val_metrics['f1']:.4f}")
              
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path}")

if __name__ == '__main__':
    main()
