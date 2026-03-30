# Anomaly Detection with PointNet and PointNet++

This repository contains a PyTorch-based proof-of-concept pipeline for binary classification of anomalies in static lidar point cloud scenes. The pipeline supports both **PointNet** and **PointNet++** (via PyTorch Geometric), and is designed to train on synthetically-generated lidar data.

## Project Structure

```text
├── README.md
├── requirements.txt         # Project dependencies
├── tests/
│   └── test_pipeline.py     # Script to generate mock data and verify the models
└── src/
    ├── dataset.py           # PyTorch Dataset for loading .npy point clouds
    ├── train.py             # Training loop, validation, and checkpointing
    ├── evaluate.py          # Final evaluation on a hold-out test set
    ├── utils.py             # Point cloud augmentations (rotation, jitter)
    └── models/
        ├── pointnet.py      # Standard PointNet architecture (with T-Nets)
        └── pointnet2.py     # PointNet++ architecture (using Set Abstraction)
```

## Setup & Installation

It is recommended to use a virtual environment or Conda environment to manage dependencies, especially `torch_geometric`.

1. **Clone the repository**
2. **Install PyTorch** following the instructions for your system (CUDA recommended) from [pytorch.org](https://pytorch.org/).
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Format

The dataset loaders are configured to read directory splits (`train/`, `val/`, `test/`) containing `.npy` point cloud arrays and a `labels.csv` file. The coordinates should be standardized $(N, 3)$ float matrices.

**Expected File Layout:**
```text
data/
├── train/
│   ├── labels.csv
│   ├── scene_001.npy
│   └── ...
└── val/
    ├── labels.csv
    ├── scene_001.npy
    └── ...
```

The `labels.csv` is expected to have headers for the filename and its binary label:
```csv
filename,label
scene_001.npy,0
scene_002.npy,1
```

*(Note: `src/dataset.py` contains basic point sampling and centering, which normalizes point clouds to a uniform 2048 points per scene.)*

## Usage

### 1. Verify Your Environment
Before plugging in your synthetic data, verify that the environment and models work correctly on your hardware by running the mock data test script:
```bash
python tests/test_pipeline.py
```

### 2. Training the Model
You can start training by specifying the learning rate, batch size, and the model architecture (`pointnet` or `pointnet2`).

```bash
# Train PointNet
python src/train.py --data_dir data/ --model pointnet --epochs 50 --batch_size 16

# Train PointNet++ 
python src/train.py --data_dir data/ --model pointnet2 --epochs 50 --batch_size 16
```
To utilize transfer learning, supply the `--checkpoint` argument to load pre-trained weights.

### 3. Evaluation
Evaluate the optimal checkpoint on an unseen test split. The script will output raw metrics along with a confusion matrix:
```bash
python src/evaluate.py --checkpoint checkpoints/pointnet_best.pth --data_dir data/ --split test --model pointnet
```

## Features
- **PointNet Support**: Standard global max pooling architecture.
- **PointNet++ Support**: Leverages Farthest Point Sampling and Ball Queries for hierarchical feature learning.
- **Robustness**: Implementations for Random Z-Rotation and Point Jitter to counteract overfitting.
- **Ready for Transfer Learning**: Simple resuming from best `.pth` weights.
