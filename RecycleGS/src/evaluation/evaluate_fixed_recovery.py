#!/usr/bin/env python3
"""Evaluate fixed recovery artifacts for Stage 2B-AC."""
import json, os, sys, numpy as np
sys.path.insert(0, '/data/wyh/RecycleGS/src')

BASE = '/data/wyh/RecycleGS'
SCENE_CONFIG = f'{BASE}/configs/stage1/reliability_scene01.yaml'
METRICS_DIR = f'{BASE}/outputs/debug/stage2bac'

methods = ['schedule_control', 'random', 'mask_risk']

with open(f'{METRICS_DIR}/unified_baseline_scene01.json') as f:
    baseline = json.load(f)

print("=== Fixed Recovery Evaluation (Stage 2B-AC) ===\n")

for method in methods:
    path = f'{METRICS_DIR}/scene01_{method}500_fixed.json'
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        continue
    with open(path) as f:
        m = json.load(f)
    delta = baseline['psnr_mean'] - m['psnr_mean']
    print(f"  Method: {method}")
    print(f"    requested_ply: {m['requested_ply']}")
    print(f"    actual_loaded_count: {m['actual_loaded_count']}")
    print(f"    PSNR: {m['psnr_mean']:.4f} (baseline: {baseline['psnr_mean']:.4f}, delta: {delta:.4f})")
    print(f"    SSIM: {m['ssim_mean']:.4f}")
    print(f"    LPIPS: {m['lpips_mean']:.4f}")
    print()
