#!/usr/bin/env python3
"""
Underwater seabed scene generator for blainder-range-scanner.

Simulates a downward-looking sonar mounted just below the waterline,
scanning a natural bumpy seabed with scattered rocks.

Normal scenes  : bumpy seabed + rocks only.
Anomaly scenes : same seabed + a foreign cube sitting on the seafloor.

Environment variables (set by the runner):
  SCENE_ID, SAR_SESSION_ID, ANOMALY_LABEL, ANOMALY_PROBABILITY,
  SCENE_GEN_DIR, RANDOM_SEED
"""

import bpy
import os
import sys
import math
import random
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration (overridable via env vars)
# ---------------------------------------------------------------------------
SCENE_ID            = os.environ.get("SCENE_ID", "0")
RANDOM_SEED         = os.environ.get("RANDOM_SEED", None)
SESSION_ID          = os.environ.get("SAR_SESSION_ID", "manual")
ANOMALY_LABEL       = os.environ.get("ANOMALY_LABEL", None)   # "0" or "1"
ANOMALY_PROBABILITY = float(os.environ.get("ANOMALY_PROBABILITY", "0.5"))

# Scene geometry
WATER_SURFACE_Z     = 0.0
SEABED_Z_BASE       = -12.0    # centre depth of seabed
SEABED_Z_VARIATION  = 2.0      # ± random shift per scene
SEABED_SIZE         = 40.0     # width / height of the seabed plane
SEABED_SUBDIVISIONS = 30       # cuts per side (higher = smoother bumps)

# Sonar sensor
SONAR_Z             = -0.3     # just below waterline (add-on needs to be under water)
SONAR_XY_JITTER     = 2.0      # random XY position offset for the sensor

# Anomaly objects
CUBE_SIZE_MIN       = 1.5    # was 0.4 — too small to hit reliably at 10m depth
CUBE_SIZE_MAX       = 2.5    # was 1.2
CUBE_XY_RANGE       = 3.0   # was 8.0 — kept near centre of scan footprint
ANOMALY_TYPES       = ["cube", "sphere", "cylinder"]  # randomised per scene

# Rocks / clutter
ROCKS_MIN           = 6
ROCKS_MAX           = 20

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_seed():
    if RANDOM_SEED:
        try: return int(RANDOM_SEED)
        except ValueError: pass
    try: return int(SCENE_ID)
    except ValueError: return hash(SCENE_ID) % (2 ** 32)


def choose_anomaly_label():
    if ANOMALY_LABEL is not None:
        return 1 if str(ANOMALY_LABEL).strip() == "1" else 0
    return 1 if random.random() < ANOMALY_PROBABILITY else 0


def _script_dir():
    base = os.environ.get("SCENE_GEN_DIR")
    if base:
        return base
    try:
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    except Exception:
        return bpy.path.abspath("//")


def output_dir():
    return Path(_script_dir()) / "runs" / SESSION_ID / "output"


def generated_dir():
    return Path(_script_dir()) / "runs" / SESSION_ID / "generated"


def _make_material(name, r, g, b, metallic=0.0, roughness=1.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value    = (r, g, b, 1.0)
        bsdf.inputs["Metallic"].default_value      = metallic
        bsdf.inputs["Roughness"].default_value     = roughness
    return mat


def _assign_mat(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def _set_category(obj, cat, part=None):
    try:
        obj["categoryID"] = cat
        if part is not None:
            obj["partID"] = part
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------
def clear_scene():
    """Remove every object, mesh, and material from the startup scene."""
    for obj in list(bpy.data.objects):
        if obj.type in ("MESH", "CURVE", "CAMERA", "LIGHT", "EMPTY"):
            bpy.data.objects.remove(obj, do_unlink=True)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)


# ---------------------------------------------------------------------------
# Seabed
# ---------------------------------------------------------------------------
def _terrain_height(x, y, phases, amplitudes):
    """
    Sum of sine/cosine waves with pre-seeded phases — gives smooth bumpy terrain.
    phases: list of 8 floats in [0, 2π]
    amplitudes: list of (amp, freq_x, freq_y) tuples
    """
    h = 0.0
    for i, (amp, fx, fy) in enumerate(amplitudes):
        h += amp * math.sin(x * fx + phases[2 * i]) * math.cos(y * fy + phases[2 * i + 1])
    return h


def add_seabed():
    """
    Build a subdivided plane displaced by layered sine waves.
    Returns (seabed_obj, base_z, phases, amplitudes, tilt_x, tilt_y)
    so callers can evaluate terrain height at any (x, y) for object placement.
    """
    base_z = SEABED_Z_BASE + random.uniform(-SEABED_Z_VARIATION, SEABED_Z_VARIATION)

    bpy.ops.mesh.primitive_plane_add(size=SEABED_SIZE, location=(0.0, 0.0, base_z))
    seabed = bpy.context.object
    seabed.name = "Seabed"

    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.subdivide(number_cuts=SEABED_SUBDIVISIONS)
    bpy.ops.object.editmode_toggle()

    phases = [random.uniform(0, math.pi * 2) for _ in range(12)]  # 6 pairs
    amplitudes = [         # (height_amplitude, freq_x, freq_y)
        (0.50, 0.25, 0.20),   # large rolling hills
        (0.30, 0.60, 0.55),   # medium undulations
        (0.15, 1.20, 1.30),   # smaller ripples
        (0.07, 2.80, 2.60),   # fine surface texture
        (0.04, 6.00, 5.50),   # high-freq ripple marks (NEW)
        (0.02, 12.0, 11.0),   # very fine grain texture (NEW)
    ]

    # Random global slope — simulates non-level seabed or angled vessel pass
    tilt_x = random.uniform(-0.12, 0.12)   # up to ~7° slope in X
    tilt_y = random.uniform(-0.12, 0.12)   # up to ~7° slope in Y

    mesh = seabed.data
    for v in mesh.vertices:
        v.co.z += (_terrain_height(v.co.x, v.co.y, phases, amplitudes)
                   + tilt_x * v.co.x + tilt_y * v.co.y)
    mesh.update()

    r = 0.30 + random.uniform(0, 0.08)
    g = 0.24 + random.uniform(0, 0.06)
    b = 0.16 + random.uniform(0, 0.05)
    mat = _make_material("SeabedMat", r, g, b, metallic=0.0, roughness=0.95)
    _assign_mat(seabed, mat)
    _set_category(seabed, "terrain", "seabed")

    return seabed, base_z, phases, amplitudes, tilt_x, tilt_y


# ---------------------------------------------------------------------------
# Rocks / natural clutter
# ---------------------------------------------------------------------------
def add_rocks(base_z, phases, amplitudes, tilt_x, tilt_y, n_rocks=None):
    """Scatter icosphere rocks across the seabed for natural visual clutter."""
    if n_rocks is None:
        n_rocks = random.randint(ROCKS_MIN, ROCKS_MAX)

    half = SEABED_SIZE * 0.40

    for i in range(n_rocks):
        subdivs   = random.randint(1, 3)
        rx        = random.uniform(-half, half)
        ry        = random.uniform(-half, half)
        # Place rock at actual terrain surface at (rx, ry)
        terrain_z = (base_z
                     + _terrain_height(rx, ry, phases, amplitudes)
                     + tilt_x * rx + tilt_y * ry)
        rz = terrain_z + random.uniform(0.0, 0.2)

        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=subdivs, radius=1.0, location=(rx, ry, rz)
        )
        rock = bpy.context.object
        rock.name = f"Rock_{i:03d}"

        sx = random.uniform(0.15, 0.70)
        sy = sx * random.uniform(0.7, 1.4)
        sz = sx * random.uniform(0.35, 0.80)
        rock.scale = (sx, sy, sz)
        rock.rotation_euler = (
            random.uniform(0.0, 0.5),
            random.uniform(0.0, 0.5),
            random.uniform(0.0, math.pi * 2),
        )
        bpy.ops.object.transform_apply(scale=True, rotation=True)

        r = 0.22 + random.uniform(0, 0.18)
        g = 0.19 + random.uniform(0, 0.12)
        b = 0.15 + random.uniform(0, 0.10)
        mat = _make_material(f"RockMat_{i:03d}", r, g, b, roughness=0.90)
        _assign_mat(rock, mat)
        _set_category(rock, "terrain", "rock")

    return n_rocks


# ---------------------------------------------------------------------------
# Anomaly cube
# ---------------------------------------------------------------------------
def add_anomaly_object(base_z, phases, amplitudes, tilt_x, tilt_y):
    """
    Place a foreign object on the seabed at its actual terrain surface height.
    Randomly selects from cube / sphere / cylinder to avoid training a
    shape-specific detector rather than an anomaly detector.
    """
    cx   = random.uniform(-CUBE_XY_RANGE, CUBE_XY_RANGE)
    cy   = random.uniform(-CUBE_XY_RANGE, CUBE_XY_RANGE)
    size = random.uniform(CUBE_SIZE_MIN, CUBE_SIZE_MAX)

    # Evaluate actual terrain height at the anomaly XY position
    terrain_z = (base_z
                 + _terrain_height(cx, cy, phases, amplitudes)
                 + tilt_x * cx + tilt_y * cy)
    cz = terrain_z + size / 2.0   # object rests on the terrain surface

    anomaly_type = random.choice(ANOMALY_TYPES)

    if anomaly_type == "cube":
        bpy.ops.mesh.primitive_cube_add(size=size, location=(cx, cy, cz))
        obj = bpy.context.object
        obj.name = "AnomalyCube"
        # Random rotation around Z only — cubes can be any orientation
        obj.rotation_euler[2] = random.uniform(0, math.pi * 2)

    elif anomaly_type == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=size / 2.0, location=(cx, cy, cz)
        )
        obj = bpy.context.object
        obj.name = "AnomalySphere"

    elif anomaly_type == "cylinder":
        # Cylinder (pipe / barrel shape)
        height = size * random.uniform(0.8, 2.0)   # can be tall or squat
        bpy.ops.mesh.primitive_cylinder_add(
            radius=size / 2.0,
            depth=height,
            location=(cx, cy, terrain_z + height / 2.0)
        )
        obj = bpy.context.object
        obj.name = "AnomalyCylinder"
        # Cylinders can be upright or lying on their side
        if random.random() < 0.4:
            obj.rotation_euler[0] = math.pi / 2.0
            obj.rotation_euler[2] = random.uniform(0, math.pi * 2)
        else:
            obj.rotation_euler[2] = random.uniform(0, math.pi * 2)

    # Slightly metallic — acoustically distinct from sand/rock
    mat = _make_material("AnomalyMat", 0.55, 0.58, 0.62, metallic=0.6, roughness=0.25)
    _assign_mat(obj, mat)
    _set_category(obj, "anomaly", anomaly_type)

    bpy.ops.object.transform_apply(rotation=True)

    print(f"  Anomaly type: {anomaly_type}  size={size:.2f}m  xy=({cx:.2f},{cy:.2f})  terrain_z={terrain_z:.2f}")
    return obj, (round(cx, 2), round(cy, 2)), anomaly_type


# ---------------------------------------------------------------------------
# Sonar sensor
# ---------------------------------------------------------------------------
def add_sonar_camera():
    """
    Add a camera just below the waterline, pointing straight down.

    Strategy: place the camera, then place an empty directly below it,
    and use a Track To constraint so the camera's -Z axis (look direction)
    points toward the empty (i.e. straight down).  Baking the constraint
    ensures matrix_world is correct before the scanner reads it.
    """
    jx = random.uniform(-SONAR_XY_JITTER, SONAR_XY_JITTER)
    jy = random.uniform(-SONAR_XY_JITTER, SONAR_XY_JITTER)

    bpy.ops.object.camera_add(location=(jx, jy, SONAR_Z))
    cam = bpy.context.object
    cam.name = "CameraSonar"

    # Manually point the camera downward:
    # In Blender, a camera with rotation_euler = (pi, 0, 0) has its -Z flipped
    # to point in +Z (upward).  To point downward (-Z world): identity (0,0,0).
    # However some Blender ops reset this.  We use mathutils to be explicit.
    import math
    # Camera local -Z  → world -Z (down)  requires no rotation at all.
    cam.rotation_euler = (0.0, 0.0, 0.0)

    return cam



# ---------------------------------------------------------------------------
# Lighting (for visual inspection of the .blend)
# ---------------------------------------------------------------------------
def add_light():
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 3))
    sun = bpy.context.object
    sun.name = "Sun"
    sun.data.energy = 3.0
    return sun


# ---------------------------------------------------------------------------
# Water profile
# ---------------------------------------------------------------------------
def setup_water_profile():
    scene = bpy.context.scene
    if hasattr(scene, "scannerProperties"):
        try:
            scene.scannerProperties.surfaceHeight     = WATER_SURFACE_Z
            scene.scannerProperties.simulateWaterProfile = True
        except Exception:
            pass
    if hasattr(scene, "custom"):
        try:
            scene.custom.clear()
            item          = scene.custom.add()
            item.depth    = 0.0
            item.speed    = 1500.0   # speed of sound in water (m/s)
            item.density  = 1.025
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Sonar scan
# ---------------------------------------------------------------------------
def run_sonar_scan(camera):
    try:
        import range_scanner
    except ImportError:
        print("ERROR: range_scanner add-on not available.")
        return False

    if camera is None or camera.name not in bpy.data.objects:
        print("ERROR: sonar camera object is missing.")
        return False

    output_dir().mkdir(parents=True, exist_ok=True)

    try:
        range_scanner.ui.user_interface.scan_static(
            bpy.context,
            scannerObject          = camera,
            resolutionX            = 256,
            fovX                   = 90.0,
            resolutionY            = 256,
            fovY                   = 90.0,
            resolutionPercentage   = 100,
            reflectivityLower      = 0.0,
            distanceLower          = 0.0,
            reflectivityUpper      = 0.0,
            distanceUpper          = 99999.9,
            maxReflectionDepth     = 1,
            enableAnimation        = False,
            frameStart=1, frameEnd=1, frameStep=1, frameRate=1,
            addNoise               = True,
            noiseType              = "gaussian",
            mu                     = 0.0,
            sigma                  = 0.08,  # ~8cm — realistic for sonar at 10m depth
            noiseAbsoluteOffset    = 0.0,
            noiseRelativeOffset    = 0.0,
            simulateRain           = False,
            rainfallRate           = 0.0,
            addMesh                = False,  # no extra mesh in the scene
            exportLAS              = True,
            exportHDF              = False,
            exportCSV              = False,
            exportPLY              = False,  # .las only
            exportSingleFrames     = False,
            exportRenderedImage    = False,
            exportSegmentedImage   = False,
            exportPascalVoc        = False,
            exportDepthmap         = False,
            depthMinDistance       = 0.0,
            depthMaxDistance       = 30.0,
            dataFilePath           = "//../output",
            dataFileName           = "scene_%s_sonar" % SCENE_ID,
            debugLines             = False,
            debugOutput            = False,
            outputProgress         = True,
            measureTime            = False,
            singleRay              = False,
            destinationObject      = None,
            targetObject           = None,
        )
        return True
    except Exception as e:
        print("Static scan raised an exception: %s" % e)
        import traceback; traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Output cleanup — keep only .blend + one labeled .las
# ---------------------------------------------------------------------------
def cleanup_outputs():
    """
    BlAInder generates up to 4 LAS variants per scan:
      scene_N_sonar_frames_1_to_1.las         (clean XYZ, labels in pt_src_id)
      scene_N_sonar_frames_1_to_1_noise.las   (noisy XYZ, same labels)  ← KEEP
      scene_N_sonar_frames_1_to_1_parts.las          (part IDs, clean)
      scene_N_sonar_frames_1_to_1_noise_parts.las    (part IDs, noisy)

    We keep only the noise LAS (labels + noise, best for training) and
    rename it to:  scene_<id>.las
    Everything else in the output dir except the meta JSON is deleted.
    """
    out = output_dir()
    stem = "scene_%s_sonar_frames_1_to_1" % SCENE_ID

    noise_src  = out / ("%s_noise.las" % stem)
    final_name = out / ("scene_%s.las" % SCENE_ID)

    # Rename the noise LAS to the canonical name
    if noise_src.exists():
        noise_src.rename(final_name)
        print("  Output LAS: %s" % final_name)
    else:
        print("  WARNING: noise LAS not found — keeping whatever is in output/")

    # Delete all other scan artefacts
    patterns_to_delete = [
        "%s.las"             % stem,   # clean (non-noise) LAS
        "%s_parts.las"       % stem,   # part-ID LAS
        "%s_noise_parts.las" % stem,   # part-ID noise LAS
    ]
    for name in patterns_to_delete:
        p = out / name
        if p.exists():
            p.unlink()
            print("  Removed:    %s" % p.name)



# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_blend():
    path = generated_dir() / ("scene_%s.blend" % SCENE_ID)
    generated_dir().mkdir(parents=True, exist_ok=True)
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        print("Saved scene: %s" % path)
    except Exception as e:
        print("Could not save .blend: %s" % e)


def write_metadata(params: dict):
    output_dir().mkdir(parents=True, exist_ok=True)
    path = output_dir() / ("scene_%s_meta.json" % SCENE_ID)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print("Metadata: %s" % path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    random.seed(get_seed())
    print("\n=== Seabed scene generator | SCENE_ID=%s | seed=%s ===" % (SCENE_ID, get_seed()))

    clear_scene()

    label = choose_anomaly_label()
    print("  anomaly_label = %d (%s)" % (label, "ANOMALY" if label else "normal"))

    seabed, base_z, phases, amplitudes, tilt_x, tilt_y = add_seabed()
    n_rocks = add_rocks(base_z, phases, amplitudes, tilt_x, tilt_y)

    anomaly_type = None
    cube_pos     = None
    if label == 1:
        _, cube_pos, anomaly_type = add_anomaly_object(
            base_z, phases, amplitudes, tilt_x, tilt_y
        )

    cam = add_sonar_camera()
    add_light()
    setup_water_profile()

    # Must save the .blend before scanning so the add-on can resolve
    # "//../output" relative to the saved file location.
    save_blend()

    ok = run_sonar_scan(cam)
    cleanup_outputs()

    meta = {
        "scene_id":        SCENE_ID,
        "label_anomaly":   label,
        "seabed_base_z":   round(base_z, 3),
        "num_rocks":       n_rocks,
        "anomaly_present": bool(label),
        "anomaly_type":    anomaly_type,
        "anomaly_xy":      cube_pos,
        "sonar_z":         SONAR_Z,
        "note":            "scan_static used (uniform depth grid, not true sonar physics)",
    }
    write_metadata(meta)

    if ok:
        print("Done — scene_%s.las written." % SCENE_ID)
    else:
        print("Sonar scan failed — .blend is saved for manual inspection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
