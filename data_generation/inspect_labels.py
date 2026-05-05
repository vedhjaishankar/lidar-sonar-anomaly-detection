#!/usr/bin/env python3
"""
inspect_labels.py  —  Read a BlAInder LAS file and verify category labeling.

Prints a per-category point count breakdown, then writes two LAS files:
  *_labeled.las   — colour-coded by categoryID (terrain=tan, anomaly=RED)
  *_anomaly.las   — only the anomaly points (for easy visual check)

Usage:
    python inspect_labels.py <scan.las>
    python inspect_labels.py runs/seabed_20260418_121207/output/scene_0_sonar_frames_1_to_1.las
"""

import sys
import pathlib
import struct
import laspy
import numpy as np

# ── colour map  (categoryID → RGB)
# BlAInder assigns IDs in order of first appearance during the scan.
# We colour-code the two known classes and leave unknowns grey.
COLOURS = {
    # category name → RGB
    "terrain":  (180, 150, 100),   # sandy brown
    "anomaly":  (255,  40,  40),   # bright red
    "water":    ( 60, 160, 230),   # blue
    "debris":   ( 80, 200, 100),   # green
    "rock":     (130, 120, 110),   # grey-brown
}
FALLBACK_COLOURS = [
    (200, 200, 200),
    (255, 200,  50),
    ( 50, 200, 255),
    (200,  50, 255),
]


def write_las(path: pathlib.Path, base_las, indices, rgb=None):
    header = base_las.header
    # Promote point format to support RGB if it doesn't already
    if rgb is not None and header.point_format.id not in (2, 3, 5):
        new_header = laspy.LasHeader(point_format=3, version=header.version)
        new_header.scales = header.scales
        new_header.offsets = header.offsets
        out_las = laspy.LasData(new_header)
        out_las.x = base_las.x[indices]
        out_las.y = base_las.y[indices]
        out_las.z = base_las.z[indices]
    else:
        out_las = laspy.LasData(header)
        out_las.points = base_las.points[indices]

    if rgb is not None:
        r = np.array([col[0] for col in rgb], dtype=np.uint16) * 256
        g = np.array([col[1] for col in rgb], dtype=np.uint16) * 256
        b = np.array([col[2] for col in rgb], dtype=np.uint16) * 256
        out_las.red = r
        out_las.green = g
        out_las.blue = b

    out_las.write(str(path))
    print(f"  Written: {path}  ({len(out_las.points):,} pts)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    las_path = pathlib.Path(sys.argv[1])
    if not las_path.exists():
        print(f"File not found: {las_path}")
        sys.exit(1)

    print(f"\nReading: {las_path}")
    las = laspy.read(str(las_path))

    # ── Print all available fields
    print("\n── Available fields ──────────────────────────────────")
    for dim in las.point_format.dimensions:
        arr = las[dim.name]
        print(f"  {dim.name:<30}  dtype={arr.dtype}  "
              f"min={arr.min():.4g}  max={arr.max():.4g}")

    # ── XYZ
    x = np.array(las.x, dtype=np.float32)
    y = np.array(las.y, dtype=np.float32)
    z = np.array(las.z, dtype=np.float32)
    xyz = list(zip(x, y, z))
    print(f"\n── Total points: {len(xyz):,}")

    # ── Category field (BlAInder uses 'classification' or extra dims)
    # Check for BlAInder's custom category field (stored in 'classification'
    # or as an extra byte field named 'category_id' / 'scalar_category_id')
    # BlAInder writes categoryID into point_source_id (see export_las.py line 22).
    # The _parts.las uses part IDs in that same field.
    # We use point_source_id as the primary label source.
    cat_field = None
    for candidate in ["point_source_id", "scalar_categoryID", "categoryID",
                       "category_id", "scalar_category_id", "classification"]:
        try:
            arr = las[candidate]
            if arr.max() > 0:          # only accept if there's actual variation
                cat_field = candidate
                break
            elif candidate == "point_source_id":
                # Even if max==0 use it — it may just mean all points are cat 0
                cat_field = candidate
                break
        except Exception:
            pass

    if cat_field is None:
        print("\n⚠  No category field found in this LAS file.")
        print("   Falling back: colouring by material RGB embedded in LAS.")

        # Use the RGB colour that BlAInder bakes from the object material.
        # The anomaly cube has a distinct metallic colour vs sandy terrain.
        r16 = np.array(las.red,   dtype=np.float32)
        g16 = np.array(las.green, dtype=np.float32)
        b16 = np.array(las.blue,  dtype=np.float32)
        mx  = 65535.0
        rgb = [(int(r/mx*255), int(g/mx*255), int(b/mx*255))
               for r, g, b in zip(r16, g16, b16)]
        out = las_path.with_name(las_path.stem + "_labeled.las")
        write_las(out, las, np.arange(len(las)), rgb)
        print("  (LAS coloured by material RGB — anomaly cube should appear "
              "lighter/metallic grey vs warm sandy terrain)")
        return

    # ── Category breakdown
    cats = np.array(las[cat_field], dtype=np.int32)
    unique_ids, counts = np.unique(cats, return_counts=True)

    # BlAInder assigns IDs in order of first appearance during scene scan.
    # In our seabed scenes the order is typically:
    #   0 = terrain (Seabed + Rocks, first objects scanned)
    #   1 = anomaly (AnomalyCube, added next)
    # But the mapping can shift.  Print the raw breakdown so you can verify.
    print(f"\n── Category breakdown (field: '{cat_field}') ─────────")
    print("   (BlAInder assigns IDs in order of first scan hit per object)")
    for uid, cnt in zip(unique_ids, counts):
        pct = 100 * cnt / len(cats)
        # Infer name from seabed scene category order
        guessed = {0: "terrain/seabed", 1: "anomaly/cube"}.get(int(uid), "unknown")
        print(f"  categoryID={uid:3d}  ({guessed:<15})  →  {cnt:8,} pts  ({pct:.1f}%)")

    # ID=1 is the anomaly in our scene (second unique category hit)
    anomaly_ids   = unique_ids[unique_ids >= 1]
    anomaly_count = np.sum(cats >= 1)
    terrain_count = np.sum(cats == 0)

    print(f"\n── Anomaly check ─────────────────────────────────────")
    if len(unique_ids) > 1:
        print(f"  ✅  Multiple categories found — labeling is working!")
        if anomaly_count > 0:
            print(f"  ✅  {anomaly_count:,} pts with categoryID≥1 (likely anomaly cube)")
    else:
        print(f"  ⚠   Only one categoryID ({unique_ids[0]}) — all points same class.")
        print("      The cube may be outside the scanner's FOV, or category IDs")
        print("      are not being set correctly on the mesh objects.")
    print(f"  Background (cat=0): {terrain_count:,} pts")

    # ── Build colour-coded PLY
    id_to_colour = {}
    fallback_idx = 0
    for uid in unique_ids:
        # Try to match by position in COLOURS dict given insertion order
        names = list(COLOURS.keys())
        if uid < len(names):
            id_to_colour[uid] = COLOURS[names[uid]]
        else:
            id_to_colour[uid] = FALLBACK_COLOURS[fallback_idx % len(FALLBACK_COLOURS)]
            fallback_idx += 1

    rgb_all     = [id_to_colour[c] for c in cats]
    
    # Get indices instead of XYZ tuples for LAS generation
    all_indices = np.arange(len(las))
    anomaly_indices = np.where(cats >= 1)[0]
    rgb_anomaly = [id_to_colour[c] for c in cats[anomaly_indices]]

    print("\n── Writing LAS files ─────────────────────────────────")
    out_labeled  = las_path.with_name(las_path.stem + "_labeled.las")
    out_anomonly = las_path.with_name(las_path.stem + "_anomaly_only.las")

    write_las(out_labeled, las, all_indices, rgb_all)

    if len(anomaly_indices) > 0:
        write_las(out_anomonly, las, anomaly_indices, rgb_anomaly)
        print(f"\n  Open '{out_anomonly.name}' to see ONLY the anomaly points (red).")
        print(f"  Open '{out_labeled.name}' to see all points colour-coded by class.")
    else:
        print("  (anomaly-only LAS skipped — 0 anomaly points)")

    print("\nDone.")


if __name__ == "__main__":
    main()
