#!/usr/bin/env python3
"""Stage 2: Dataset conversion, camera validation, training launch, evaluation"""
import sys, os, json, csv, shutil, subprocess, time
import numpy as np
from PIL import Image

BASE = "/data/wyh/DeformTransGS"
STAGE1_GT = f"{BASE}/experiments/stage1_minimal_gt"
DATASET_DIR = f"{BASE}/data/stage2_canonical_tsgs"
OUTPUT_DIR = f"{BASE}/experiments/stage2_canonical_asg_gate"
BASELINE_DIR = f"{BASE}/baselines"
TSGS_DIR = "/data/wyh/repos/TSGS"
os.makedirs(f"{OUTPUT_DIR}/renders", exist_ok=True)

log_lines = []
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)

# ═══════════════════════════════════════════════════════════
# 1. Dataset Conversion
# ═══════════════════════════════════════════════════════════
hdr("1. Dataset conversion")

cams = json.load(open(f"{STAGE1_GT}/cameras.json"))
W, H = 512, 512
fov_deg = 45
fov_rad = np.deg2rad(fov_deg)

# Compute camera-to-world matrices
def look_at_c2w(origin, target, up):
    forward = np.array(origin) - np.array(target)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(np.array(up), forward)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = cam_up
    c2w[:3, 2] = -forward  # OpenGL: camera looks along -Z
    c2w[:3, 3] = origin
    return c2w

frames_full = []
frames_diag_train = []
frames_diag_test = []
diag_test_ids = {3, 7, 11}

for ci, cam in enumerate(cams):
    c2w = look_at_c2w(cam["origin"], cam["target"], cam["up"])
    frame = {
        "file_path": f"./images/cam_{ci:03d}",
        "transform_matrix": c2w.tolist()
    }
    frames_full.append(frame)
    if ci in diag_test_ids:
        frames_diag_test.append(frame)
    else:
        frames_diag_train.append(frame)

# Copy images and create transforms.json
for dataset_name, frames, cam_ids in [
    ("dataset_full12", frames_full, list(range(12))),
    ("dataset_diag9_3", frames_diag_train + frames_diag_test,
     [i for i in range(12) if i not in diag_test_ids] + list(diag_test_ids))
]:
    ds_dir = f"{DATASET_DIR}/{dataset_name}"
    img_dir = f"{ds_dir}/images"
    os.makedirs(img_dir, exist_ok=True)

    for ci in cam_ids:
        src = f"{STAGE1_GT}/render_gt/canonical/cam_{ci:03d}.png"
        dst = f"{img_dir}/cam_{ci:03d}.png"
        shutil.copy2(src, dst)

    transforms = {"camera_angle_x": fov_rad, "frames": frames}
    json.dump(transforms, open(f"{ds_dir}/transforms_train.json", "w"), indent=2)
    json.dump(transforms, open(f"{ds_dir}/transforms_test.json", "w"), indent=2)
    log(f"  {dataset_name}: {len(frames)} cameras, images copied")

# Also copy background_only
bg_dir = f"{DATASET_DIR}/background_only"
os.makedirs(bg_dir, exist_ok=True)
for ci in range(12):
    shutil.copy2(f"{STAGE1_GT}/render_gt/background_only/cam_{ci:03d}.png", f"{bg_dir}/cam_{ci:03d}.png")

# ═══════════════════════════════════════════════════════════
# 2. Camera Round-Trip Validation
# ═══════════════════════════════════════════════════════════
hdr("2. Camera round-trip validation")
sys.path.insert(0, TSGS_DIR)
sys.path.insert(0, f"{TSGS_DIR}/pytorch3d_stub")
import torch
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, fov2focal

max_center_err = 0.0
max_angle_err = 0.0
max_fov_err = 0.0

for ci, cam in enumerate(cams):
    c2w = look_at_c2w(cam["origin"], cam["target"], cam["up"])
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T  # Transposed for CUDA convention
    T = w2c[:3, 3]

    focal_x = fov2focal(fov_rad, W)
    FovX = focal2fov(focal_x, W)
    FovY = focal2fov(focal_x, H)

    tsgs_cam = Camera(
        colmap_id=ci, R=R, T=T, FoVx=FovX, FoVy=FovY,
        image=torch.zeros(3, H, W), gt_alpha_mask=None,
        image_name=f"cam_{ci:03d}", uid=ci,
        data_device="cpu"
    )

    # Recover camera center
    recovered_center = tsgs_cam.camera_center.numpy()
    center_err = np.linalg.norm(recovered_center - np.array(cam["origin"]))
    max_center_err = max(max_center_err, center_err)

    # Forward direction
    recovered_forward = -tsgs_cam.world_view_transform[:3, 2].numpy()
    recovered_forward = recovered_forward / np.linalg.norm(recovered_forward)
    orig_forward = np.array(cam["target"]) - np.array(cam["origin"])
    orig_forward = orig_forward / np.linalg.norm(orig_forward)
    angle_err = np.rad2deg(np.arccos(np.clip(np.dot(recovered_forward, orig_forward), -1, 1)))
    max_angle_err = max(max_angle_err, angle_err)

    # FOV
    fov_err = abs(np.rad2deg(FovX) - fov_deg)
    max_fov_err = max(max_fov_err, fov_err)

log(f"  Camera center max error: {max_center_err:.6e}")
log(f"  Forward direction max angular error: {max_angle_err:.6e} deg")
log(f"  FOV max error: {max_fov_err:.6e} deg")

assert max_center_err < 1e-5, f"Camera center error too large: {max_center_err}"
assert max_angle_err < 1e-3, f"Camera angular error too large: {max_angle_err}"
assert max_fov_err < 1e-5, f"FOV error too large: {max_fov_err}"
log("  Camera validation PASSED")

# ═══════════════════════════════════════════════════════════
# 3. Launch Training
# ═══════════════════════════════════════════════════════════
hdr("3. Launching training")

import GPUtil

def find_free_gpu():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        if gpu.memoryUtil < 0.3:
            return gpu.id
    return None

gpu_id = find_free_gpu()
if gpu_id is None:
    log("  WARNING: No free GPU found, using GPU 2")
    gpu_id = 2

log(f"  Using GPU {gpu_id}")

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
env["LD_LIBRARY_PATH"] = f"/home/wyh/.local/lib/python3.10/site-packages/torch/lib:{env.get('LD_LIBRARY_PATH', '')}"
env["PYTHONPATH"] = f"{TSGS_DIR}:{TSGS_DIR}/pytorch3d_stub:{env.get('PYTHONPATH', '')}"

dataset_path = f"{DATASET_DIR}/dataset_full12"

def train_model(model_name, extra_args):
    out_dir = f"{BASELINE_DIR}/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    log(f"  Training {model_name}...")

    cmd = [
        "python3", f"{TSGS_DIR}/train.py",
        "-s", dataset_path,
        "-m", out_dir,
        "--sh_degree", "3",
        "--asg_degree", "24",
        "--resolution", "2",
        "--iterations", "30000",
        "--save_iterations", "15000", "30000",
        "--test_iterations", "15000", "30000",
        "--checkpoint_iterations", "15000", "30000",
        "--data_device", "cuda",
    ] + extra_args

    log(f"  Command: {' '.join(cmd)}")
    with open(f"{out_dir}/train.log", "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        for line in iter(proc.stdout.readline, b''):
            decoded = line.decode().rstrip()
            log(f"    {decoded}")
            log_f.write(decoded + "\n")
        proc.wait()
    log(f"  {model_name} training exit code: {proc.returncode}")
    return proc.returncode

# Model A: ASG enabled
train_model("tsgs_synth_canonical_asg", ["--use_asg"])

# Model B: ASG disabled (SH only) 
train_model("tsgs_synth_canonical_sh", [])

log("\n=== Both models trained ===")
