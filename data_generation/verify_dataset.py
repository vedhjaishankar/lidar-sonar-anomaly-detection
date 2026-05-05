#!/usr/bin/env python3
"""
verify_and_convert.py
─────────────────────
Verifies a generated scene LAS file is correctly labeled, then converts
it to the .npy format that LidarAnomalyDataset expects, and does a full
dry-run through the dataset loader to confirm training readiness.

Usage (individual file):
    python verify_and_convert.py <scene_N.las> --label 1

Usage (whole session):
    python verify_and_convert.py --session runs/seabed_20260418_122608

The converted .npy files are written to:
    <session>/npy/scene_N.npy

A labels.csv is also written/updated in that folder.
"""

import sys
import argparse
import pathlib
import textwrap

import numpy as np
import laspy

# ── ANSI colours (graceful fallback on Windows without colour support)
try:
    import colorama; colorama.init()
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
except ImportError:
    GREEN = RED = YELLOW = RESET = BOLD = ""


def ok(msg):  print(f"  {GREEN}✓{RESET}  {msg}")
def err(msg): print(f"  {RED}✗{RESET}  {msg}")
def warn(msg):print(f"  {YELLOW}!{RESET}  {msg}")
def hdr(msg): print(f"\n{BOLD}{msg}{RESET}")


# ──────────────────────────────────────────────────────────────────────
# Step 1: Read & verify the LAS file
# ──────────────────────────────────────────────────────────────────────
def verify_las(las_path: pathlib.Path, expected_label: int | None = None):
    """
    Returns (xyz_array, label_ok, category_ids) or raises on critical error.
    xyz_array shape: (N, 3) float32.
    """
    hdr(f"1. Reading LAS: {las_path.name}")

    las = laspy.read(str(las_path))
    n   = len(las.x)
    print(f"     Points     : {n:,}")
    print(f"     LAS format : {las.point_format.id}")

    x = np.array(las.x, dtype=np.float32)
    y = np.array(las.y, dtype=np.float32)
    z = np.array(las.z, dtype=np.float32)
    xyz = np.stack([x, y, z], axis=1)   # (N, 3)

    # Coordinate sanity
    for axis, arr, name in [(0, x, "X"), (1, y, "Y"), (2, z, "Z")]:
        print(f"     {name} range    : [{arr.min():.3f}, {arr.max():.3f}]")
    ok("Coordinates loaded")

    # ── Label field
    hdr(f"2. Checking labels (point_source_id)")
    cats = np.array(las.point_source_id, dtype=np.int32)
    unique, counts = np.unique(cats, return_counts=True)
    for uid, cnt in zip(unique, counts):
        name = {0: "terrain/background", 1: "anomaly/cube"}.get(int(uid), "unknown")
        pct  = 100 * cnt / n
        print(f"     categoryID={uid}  ({name:<20})  {cnt:7,} pts  ({pct:.1f}%)")

    if len(unique) == 1:
        warn(f"Only one categoryID ({unique[0]}) — all points same class.")
        if unique[0] == 0 and expected_label == 1:
            err("Expected an anomaly scene but cube got no hits!")
            return xyz, False, cats
        else:
            warn("This is normal for background-only scenes (label=0).")
    else:
        ok(f"Multiple categories found — per-point labeling is working")

    # ── Scene-level label consistency
    hdr(f"3. Scene-level label check")
    has_anomaly_pts = np.any(cats >= 1)
    if expected_label is not None:
        if expected_label == 1 and not has_anomaly_pts:
            err("Scene is labeled ANOMALY but no anomaly points were scanned.")
            return xyz, False, cats
        elif expected_label == 0 and has_anomaly_pts:
            err("Scene is labeled NORMAL but anomaly points found!")
            return xyz, False, cats
        else:
            ok(f"Scene-level label ({expected_label}) is consistent with point-level labels")
    else:
        warn("No expected_label provided — skipping scene-level consistency check")

    return xyz, True, cats


# ──────────────────────────────────────────────────────────────────────
# Step 2: Convert to .npy
# ──────────────────────────────────────────────────────────────────────
def convert_to_npy(xyz: np.ndarray, cats: np.ndarray, out_path: pathlib.Path):
    """
    Saves two files:
      scene_N.npy      — (N, 3) float32 XYZ  → used for classification
      scene_N_seg.npy  — (N,)   int32 labels → used for segmentation
                           0 = terrain/background
                           1 = anomaly/cube
    """
    hdr("4. Converting to .npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # XYZ for classification / shared backbone
    np.save(str(out_path), xyz)
    size_kb = out_path.stat().st_size / 1024
    ok(f"Saved {out_path.name:<30}  shape={xyz.shape}   ({size_kb:.0f} KB)")

    # Per-point labels for segmentation
    seg_path = out_path.with_name(out_path.stem + "_seg.npy")
    seg_labels = cats.astype(np.int32)
    np.save(str(seg_path), seg_labels)
    size_kb2 = seg_path.stat().st_size / 1024
    ok(f"Saved {seg_path.name:<30}  shape={seg_labels.shape}   ({size_kb2:.0f} KB)")

    # Quick sanity: confirm anomaly points are present in seg labels
    n_anomaly = np.sum(seg_labels >= 1)
    n_terrain = np.sum(seg_labels == 0)
    print(f"     Seg label dist:  terrain={n_terrain:,}  anomaly={n_anomaly:,}")
    return seg_path


# ──────────────────────────────────────────────────────────────────────
# Step 3: Dry-run through LidarAnomalyDataset
# ──────────────────────────────────────────────────────────────────────
def dry_run_dataset(npy_path: pathlib.Path, label: int, num_points: int = 2048):
    hdr(f"5. Dataset dry-run  (num_points={num_points})")

    # ── Classification path (what LidarAnomalyDataset does)
    points = np.load(str(npy_path)).astype(np.float32)
    print(f"     XYZ shape    : {points.shape}")

    n = points.shape[0]
    choice = (np.random.choice(n, num_points, replace=False)
              if n >= num_points else
              np.random.choice(n, num_points, replace=True))
    if n < num_points:
        warn(f"Fewer points than num_points ({n} < {num_points}): sampling with replacement")

    sampled_xyz = points[choice] - np.mean(points[choice], axis=0)

    import torch
    pts_t   = torch.from_numpy(sampled_xyz)
    label_t = torch.tensor([label], dtype=torch.float32)
    assert pts_t.shape == (num_points, 3)
    assert not torch.isnan(pts_t).any()
    ok(f"Classification tensor : {tuple(pts_t.shape)}  label={label_t.item()}")

    # ── Segmentation path
    seg_path = npy_path.with_name(npy_path.stem + "_seg.npy")
    if seg_path.exists():
        seg = np.load(str(seg_path)).astype(np.int64)
        sampled_seg = seg[choice]                    # same indices as XYZ
        seg_t = torch.from_numpy(sampled_seg)        # (2048,) int64
        assert seg_t.shape == (num_points,)

        # Simulate IoU on ground-truth vs a dummy "all-terrain" prediction
        # (proves the metric pipeline works end-to-end)
        pred_dummy = torch.zeros_like(seg_t)         # model predicts all terrain
        iou_terrain, iou_anomaly, miou = compute_iou(seg_t.numpy(), pred_dummy.numpy(), n_classes=2)
        ok(f"Segmentation tensor  : {tuple(seg_t.shape)}  dtype={seg_t.dtype}")
        print(f"     Dummy IoU check  : terrain={iou_terrain:.3f}  anomaly={iou_anomaly:.3f}  mIoU={miou:.3f}")
        print(f"     (anomaly IoU=0 expected for all-terrain dummy prediction)")
    else:
        warn("_seg.npy not found — skipping segmentation dry-run")

    ok("All checks passed — ready for both classification AND segmentation training")
    return pts_t, label_t


# ──────────────────────────────────────────────────────────────────────
# IoU metric
# ──────────────────────────────────────────────────────────────────────
def compute_iou(gt: np.ndarray, pred: np.ndarray, n_classes: int = 2):
    """
    Compute per-class IoU and mIoU for a single point cloud.

    gt, pred : (N,) integer arrays with class indices.
    Returns  : (iou_per_class list, mIoU float)
    """
    ious = []
    for cls in range(n_classes):
        tp = np.sum((gt == cls) & (pred == cls))
        fp = np.sum((gt != cls) & (pred == cls))
        fn = np.sum((gt == cls) & (pred != cls))
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float("nan"))
    valid = [v for v in ious if not np.isnan(v)]
    miou  = float(np.mean(valid)) if valid else float("nan")
    return ious[0], ious[1] if len(ious) > 1 else float("nan"), miou


# ──────────────────────────────────────────────────────────────────────
# Session-level processing
# ──────────────────────────────────────────────────────────────────────
def process_session(session_dir: pathlib.Path):
    output_dir = session_dir / "output"
    npy_dir    = session_dir / "npy"

    import csv
    labels_csv = output_dir / "labels.csv"
    if not labels_csv.exists():
        err(f"labels.csv not found at {labels_csv}")
        sys.exit(1)

    rows = []
    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"\nFound {len(rows)} scene(s) in labels.csv")

    npy_csv_rows = []
    all_ok = True

    for row in rows:
        las_path = output_dir / row["filename"]
        label    = int(row["label"])
        print(f"\n{'='*60}")
        print(f"Scene: {las_path.name}  (label={label})")
        print('='*60)

        if not las_path.exists():
            err(f"LAS file missing: {las_path}")
            all_ok = False
            continue

        xyz, label_ok, cats = verify_las(las_path, expected_label=label)
        if not label_ok:
            all_ok = False

        npy_name = las_path.stem + ".npy"
        npy_path = npy_dir / npy_name
        convert_to_npy(xyz, cats, npy_path)
        dry_run_dataset(npy_path, label)

        npy_csv_rows.append({"filename": npy_name, "label": label})

    # Write labels.csv for the npy directory (what dataset.py reads)
    npy_dir.mkdir(parents=True, exist_ok=True)
    npy_labels = npy_dir / "labels.csv"
    with open(npy_labels, "w", newline="") as f:
        writer = __import__("csv").DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        writer.writerows(npy_csv_rows)

    hdr("Summary")
    if all_ok:
        ok(f"All {len(rows)} scene(s) verified and converted.")
        ok(f"NPY files + labels.csv written to: {npy_dir}")
        print(f"\n  To train, point dataset.py at:  {npy_dir}")
    else:
        err("Some scenes failed verification — check output above.")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Verify and convert BlAInder LAS → NPY for training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              # Verify + convert a whole session (reads labels.csv automatically):
              python verify_and_convert.py --session runs/seabed_20260418_122608

              # Verify a single file manually:
              python verify_and_convert.py scene_0.las --label 1
        """)
    )
    ap.add_argument("las",     nargs="?", help="Path to a single .las file")
    ap.add_argument("--label", type=int,  help="Expected scene label (0=normal, 1=anomaly)")
    ap.add_argument("--session", type=str, help="Path to a session directory")
    args = ap.parse_args()

    if args.session:
        process_session(pathlib.Path(args.session))
    elif args.las:
        las_path = pathlib.Path(args.las)
        xyz, label_ok, cats = verify_las(las_path, expected_label=args.label)
        npy_path = las_path.with_suffix(".npy")
        convert_to_npy(xyz, cats, npy_path)
        dry_run_dataset(npy_path, args.label if args.label is not None else 0)
        if not label_ok:
            sys.exit(1)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
