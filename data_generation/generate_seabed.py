#!/usr/bin/env python3
"""
Batch runner for underwater seabed anomaly-detection data generation.

Launches Blender in background once per scene.  Each run creates:
  - runs/<session>/generated/scene_<id>.blend
  - runs/<session>/output/scene_<id>_sonar.las
  - runs/<session>/output/scene_<id>_sonar.ply
  - runs/<session>/output/scene_<id>_meta.json
  - runs/<session>/labels.csv   (cumulative)

Usage:
  python generate_seabed.py --count 10 --blender "C:/path/to/blender.exe"
  python generate_seabed.py --count 1  --blender "C:/path/to/blender.exe" --anomaly-probability 1.0
"""

import subprocess
import os
import sys
import argparse
import json
import csv
import random
from datetime import datetime
from pathlib import Path

DIR     = Path(__file__).parent.resolve()
SCRIPT  = DIR / "seabed_scene.py"


# ---------------------------------------------------------------------------
# Blender launcher
# ---------------------------------------------------------------------------
def run_blender_scene(scene_id, session_id, anomaly_label, blender_cmd, blueprint=None):
    env = os.environ.copy()
    env["SCENE_ID"]           = str(scene_id)
    env["SAR_SESSION_ID"]     = session_id
    env["SCENE_GEN_DIR"]      = str(DIR)
    env["ANOMALY_LABEL"]      = str(int(anomaly_label))

    cmd = [blender_cmd, "--background"]
    if blueprint and Path(blueprint).exists():
        cmd = [blender_cmd, str(blueprint), "--background"]
    cmd += ["--python", str(SCRIPT)]

    try:
        result = subprocess.run(cmd, cwd=str(DIR), env=env, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        print(f"Blender not found at: {blender_cmd}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate seabed sonar training scenes.")
    ap.add_argument("--count",               "-n", type=int,   default=5,
                    help="Number of scenes to generate (default: 5)")
    ap.add_argument("--blender",             "-b", type=str,   default="blender",
                    help="Path to blender executable")
    ap.add_argument("--anomaly-probability",       type=float, default=0.5,
                    help="Fraction of scenes that contain the anomaly cube [0–1]")
    ap.add_argument("--exact-balance",             action="store_true",
                    help="Guarantee exact split instead of random per scene")
    ap.add_argument("--session",                   type=str,   default=None,
                    help="Reuse an existing session ID (resumes numbering)")
    args = ap.parse_args()

    if not SCRIPT.exists():
        print(f"ERROR: scene script not found: {SCRIPT}")
        sys.exit(1)

    session_id  = args.session or datetime.now().strftime("seabed_%Y%m%d_%H%M%S")
    session_dir = DIR / "runs" / session_id
    output_dir  = session_dir / "output"
    gen_dir     = session_dir / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Build label sequence
    prob = max(0.0, min(1.0, args.anomaly_probability))
    if args.exact_balance:
        n_anom  = int(round(args.count * prob))
        labels  = [1] * n_anom + [0] * (args.count - n_anom)
        random.Random(42).shuffle(labels)
    else:
        rng    = random.Random(42)
        labels = [1 if rng.random() < prob else 0 for _ in range(args.count)]

    # Session manifest
    manifest = {
        "session_id":          session_id,
        "created_at":          datetime.now().isoformat(),
        "count":               args.count,
        "anomaly_probability": prob,
        "exact_balance":       args.exact_balance,
        "planned_anomaly":     int(sum(labels)),
        "planned_normal":      int(len(labels) - sum(labels)),
        "blender":             args.blender,
    }
    # Write session info (manifest)
    with open(session_dir / "session_info.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerating {args.count} seabed scene(s)")
    print(f"  Session  : {session_id}")
    print(f"  Output   : {output_dir}")
    print(f"  Blender  : {args.blender}")
    print(f"  Class mix: {sum(labels)} anomalous / {len(labels)-sum(labels)} normal\n")

    labels_path = output_dir / "labels.csv"
    
    # Write header if it's a new file
    if not labels_path.exists():
        with open(labels_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=["filename", "label"]).writeheader()

    ok_count = 0
    for i, anomaly_label in enumerate(labels):
        canonical_las = output_dir / f"scene_{i}.las"
        
        # Check if scene already exists (for resuming)
        if canonical_las.exists():
            print(f"--- Scene {i+1}/{args.count}  (ALREADY EXISTS, skipping) ---")
            ok_count += 1
            continue

        print(f"--- Scene {i+1}/{args.count}  (label={anomaly_label}) ---")
        success = run_blender_scene(
            scene_id      = i,
            session_id    = session_id,
            anomaly_label = anomaly_label,
            blender_cmd   = args.blender,
        )
        
        if success:
            if canonical_las.exists():
                # Write to CSV immediately so we don't lose it if we crash
                with open(labels_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=["filename", "label"]).writerow({
                        "filename": canonical_las.name, 
                        "label": anomaly_label
                    })
                ok_count += 1
                print(f"  => OK (saved to labels.csv)")
            else:
                print(f"  => FAILED (Blender exited OK but no LAS found)")
        else:
            print(f"  => FAILED (Blender error)")

    print(f"\nDone: {ok_count}/{args.count} scenes present in {output_dir}")
    print(f"labels.csv: {labels_path}")

    sys.exit(0 if ok_count == args.count else 1)


if __name__ == "__main__":
    main()
