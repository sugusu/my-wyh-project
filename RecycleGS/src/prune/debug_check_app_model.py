#!/usr/bin/env python3
"""Debug: check if AppModel weights are actually loaded and applied."""
import os, sys, torch, numpy as np

sys.path.insert(0, '/data/wyh/RecycleGS/src')
sys.path.insert(0, '/data/wyh/repos/TSGS')
sys.path.insert(0, '/data/wyh/repos/TSGS/pytorch3d_stub')
os.environ['LD_LIBRARY_PATH'] = '/home/wyh/.local/lib/python3.10/site-packages/torch/lib'

from scene.app_model import AppModel

baseline_dir = "/data/wyh/RecycleGS/baselines/tsgs_scene01_full"

# Check raw weights
weights_path = os.path.join(baseline_dir, "app_model", "iteration_15000", "app.pth")
state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
print(f"AppModel state_dict keys: {list(state_dict.keys())}")
print(f"appear_ab shape: {state_dict['appear_ab'].shape}")
print(f"appear_ab stats: mean={state_dict['appear_ab'].mean():.6f}, std={state_dict['appear_ab'].std():.6f}")
print(f"appear_ab min={state_dict['appear_ab'].min():.6f}, max={state_dict['appear_ab'].max():.6f}")
print(f"First 5 values:\n{state_dict['appear_ab'][:5]}")
print(f"Non-zero count: {(state_dict['appear_ab'] != 0).sum()} / {state_dict['appear_ab'].numel()}")

device = 'cuda:0'
app_model = AppModel()
app_model.load_state_dict(state_dict)
app_model.cuda()
app_model.eval()

print(f"\nappear_ab after load: mean={app_model.appear_ab.mean():.6f}, std={app_model.appear_ab.std():.6f}")

# Check what app_image would look like for camera 0
alpha, beta = app_model.appear_ab[0]
print(f"\nCamera 0: alpha={alpha.item():.6f}, beta={beta.item():.6f}")
print(f"  exp(alpha) = {torch.exp(alpha).item():.6f}")
print(f"  This means: app_image = {torch.exp(alpha).item():.4f} * render + {beta.item():.4f}")

# Check a range of cameras
for i in range(0, 50, 10):
    a, b = app_model.appear_ab[i]
    print(f"  Camera {i}: alpha={a.item():.6f}, beta={b.item():.6f}, exp(alpha)={torch.exp(a).item():.6f}")
