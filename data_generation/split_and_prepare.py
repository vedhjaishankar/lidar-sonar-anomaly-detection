#!/usr/bin/env python3
"""
split_and_prepare.py
────────────────────
Full post-generation pipeline: LAS → NPY conversion + train/val/test split.

Reads one or more generation session directories, converts all .las files to
.npy (XYZ) and _seg.npy (per-point labels), then splits into train / val / test
subdirectories that train.py can consume directly.

Usage
─────
  # Single session
  python split_and_prepare.py --session runs/seabed_20260418_124055 --out ../../data

  # All sessions under a root dir (for combining multiple generation batches)
  python split_and_prepare.py --sessions-root runs/ --out ../../data

  # Custom split ratios (default 70/15/15)
  python split_and_prepare.py --session runs/... --out ../../data --split 0.7 0.15 0.15

Output layout
─────────────
  <out>/
    train/
      labels.csv          ← filename, label
      scene_0.npy         ← (N, 3) float32 XYZ
      scene_0_seg.npy     ← (N,)   int32   per-point labels (0=terrain, 1=anomaly)
      ...
    val/
      ...
    test/
      ...
    prepare_info.json     ← summary (counts, split ratios, session sources)
"""

import argparse
import csv
import json
import pathlib
import random
import shutil
import sys

import laspy
import numpy as np


# ── label colour map for terminal output ─────────────────────────────────────
RESET = "\033[0m"
GREEN = "\033[92m"
RED   = "\033[91m"
BOLD  = "\033[1m"
YELLOW = "\033[93m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def err(msg):  print(f"  {RED}✗{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET}  {msg}")
def hdr(msg):  print(f"\n{BOLD}{msg}{RESET}")


# ── LAS reader ───────────────────────────────────────────────────────────────
def read_las(las_path: pathlib.Path):
    """
    Read a BlAInder-generated LAS file.

    Returns:
        xyz  : (N, 3) float32   — X, Y, Z coordinates
        cats : (N,)   int32     — per-point category ID (from point_source_id)
                                   0 = terrain / background
                                   1 = anomaly object
    """
    las  = laspy.read(str(las_path))
    x    = np.array(las.x, dtype=np.float32)
    y    = np.array(las.y, dtype=np.float32)
    z    = np.array(las.z, dtype=np.float32)
    xyz  = np.stack([x, y, z], axis=1)
    cats = np.array(las.point_source_id, dtype=np.int32)
    return xyz, cats


# ── Single-session processor ──────────────────────────────────────────────────
def collect_scenes(session_dir: pathlib.Path):
    """
    Collect all scenes from one session.

    Returns a list of dicts:
        { "las_path": Path, "label": int (scene-level), "scene_id": str }
    """
    output_dir  = session_dir / "output"
    labels_csv  = output_dir / "labels.csv"

    if not labels_csv.exists():
        warn(f"No labels.csv in {output_dir} — skipping session.")
        return []

    scenes = []
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            fname    = row["filename"]          # e.g. "scene_0.las"
            las_path = output_dir / fname
            if not las_path.exists():
                warn(f"  Missing LAS: {las_path}")
                continue
            stem = pathlib.Path(fname).stem     # "scene_0"
            scenes.append({
                "las_path": las_path,
                "label":    int(row["label"]),
                "scene_id": stem,
            })

    return scenes


# ── NPY converter ─────────────────────────────────────────────────────────────
def convert_scene(scene: dict, out_dir: pathlib.Path, idx: int):
    """
    Convert one scene's LAS to XYZ + seg NPY files in out_dir.
    Files are named by global index (idx) to avoid collisions across sessions.

    Returns dict { "filename": "scene_NNN.npy", "label": int } or None on error.
    """
    las_path = scene["las_path"]
    try:
        xyz, cats = read_las(las_path)
    except Exception as e:
        err(f"Failed to read {las_path.name}: {e}")
        return None

    n_anomaly = int(np.sum(cats >= 1))
    n_terrain = int(np.sum(cats == 0))

    # Warn if labeled anomaly but no anomaly points were scanned
    if scene["label"] == 1 and n_anomaly == 0:
        warn(f"  {las_path.name}: labeled ANOMALY but 0 anomaly points — including anyway.")

    base_name = f"scene_{idx:05d}"
    npy_path  = out_dir / f"{base_name}.npy"
    seg_path  = out_dir / f"{base_name}_seg.npy"

    np.save(str(npy_path), xyz)
    np.save(str(seg_path), cats.astype(np.int32))

    return {
        "filename":  f"{base_name}.npy",
        "label":     scene["label"],
        "n_pts":     len(xyz),
        "n_anomaly": n_anomaly,
        "n_terrain": n_terrain,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────
def prepare(sessions: list[pathlib.Path], out_root: pathlib.Path,
             split_ratios: tuple[float, float, float], seed: int = 42):

    hdr("1. Collecting scenes from sessions")
    all_scenes = []
    for session in sessions:
        scenes = collect_scenes(session)
        print(f"  {session.name:<40}  {len(scenes)} scenes")
        all_scenes.extend(scenes)

    if not all_scenes:
        err("No scenes found. Nothing to do.")
        sys.exit(1)

    ok(f"Total scenes collected: {len(all_scenes)}")

    # ── Shuffle deterministically ──────────────────────────────────────────
    hdr("2. Shuffling and splitting")
    rng = random.Random(seed)
    rng.shuffle(all_scenes)

    n       = len(all_scenes)
    n_train = int(n * split_ratios[0])
    n_val   = int(n * split_ratios[1])
    n_test  = n - n_train - n_val          # remainder to test

    splits = {
        "train": all_scenes[:n_train],
        "val":   all_scenes[n_train : n_train + n_val],
        "test":  all_scenes[n_train + n_val :],
    }
    for name, subset in splits.items():
        n_anom = sum(s["label"] for s in subset)
        print(f"  {name:<6}: {len(subset):4d} scenes  "
              f"({n_anom} anomalous / {len(subset)-n_anom} normal)")

    # ── Convert and write ──────────────────────────────────────────────────
    hdr("3. Converting LAS → NPY")
    out_root.mkdir(parents=True, exist_ok=True)

    stats = {}
    global_idx = 0   # unique index across all splits

    for split_name, subset in splits.items():
        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        fail_count = 0

        for scene in subset:
            result = convert_scene(scene, split_dir, global_idx)
            global_idx += 1
            if result is None:
                fail_count += 1
                continue
            rows.append({"filename": result["filename"], "label": result["label"]})

        # Write labels.csv for this split
        csv_path = split_dir / "labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label"])
            writer.writeheader()
            writer.writerows(rows)

        ok(f"{split_name:<6}: {len(rows)} scenes written to {split_dir}  "
           f"({'failed: '+str(fail_count) if fail_count else 'all OK'})")
        stats[split_name] = {"count": len(rows), "failed": fail_count}

    # ── Summary JSON ──────────────────────────────────────────────────────
    info = {
        "total_scenes":   n,
        "split_ratios":   list(split_ratios),
        "random_seed":    seed,
        "splits":         stats,
        "session_sources": [str(s) for s in sessions],
    }
    with open(out_root / "prepare_info.json", "w") as f:
        json.dump(info, f, indent=2)

    hdr("Done")
    ok(f"Ready dataset at: {out_root.resolve()}")
    print(f"\n  To train, run from src/:")
    print(f"    python train.py --task segmentation --model pointnet2seg "
          f"--data_dir {out_root.resolve()}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Convert BlAInder LAS output to train/val/test NPY dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--session",       type=str,
                     help="Path to a single generation session directory.")
    grp.add_argument("--sessions-root", type=str,
                     help="Root directory containing multiple session subdirs.")

    ap.add_argument("--out",    required=True, type=str,
                    help="Output directory for the prepared dataset.")
    ap.add_argument("--split",  nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="Fractional split ratios, must sum to 1. Default: 0.70 0.15 0.15")
    ap.add_argument("--seed",   type=int, default=42,
                    help="Random seed for reproducible shuffling.")
    args = ap.parse_args()

    # Validate split ratios
    split_ratios = tuple(args.split)
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        ap.error(f"--split values must sum to 1.0, got {sum(split_ratios):.4f}")

    # Collect session directories
    if args.session:
        sessions = [pathlib.Path(args.session)]
    else:
        root = pathlib.Path(args.sessions_root)
        sessions = sorted([d for d in root.iterdir() if d.is_dir()])
        if not sessions:
            ap.error(f"No subdirectories found in {root}")

    out_root = pathlib.Path(args.out)
    prepare(sessions, out_root, split_ratios, seed=args.seed)


if __name__ == "__main__":
    main()
