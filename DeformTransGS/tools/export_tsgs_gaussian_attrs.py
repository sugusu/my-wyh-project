#!/usr/bin/env python3
"""
export_tsgs_gaussian_attrs.py
Stage 0: 导出 TSGS Gaussian 所有属性到 NPZ，并生成属性审计报告。
"""

import sys, os, time, json
import numpy as np
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────
TSGS_REPO = "/data/wyh/repos/TSGS"
CHECKPOINT_PLY = "/data/wyh/RecycleGS/baselines/tsgs_scene01_full/point_cloud/iteration_15000/point_cloud.ply"
OUTPUT_DIR = "/data/wyh/DeformTransGS/experiments/stage0_attribute_audit"

# 将 TSGS 源码加入 sys.path
for p in [TSGS_REPO, os.path.join(TSGS_REPO, "pytorch3d_stub")]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 日志辅助 ──────────────────────────────────────────────
log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

def log_separator(title=""):
    line = "=" * 72
    log(line)
    if title:
        log(f"  {title}")
        log(line)

# ── 步骤 1: 分析 PLY 属性列表 ────────────────────────────
log_separator("步骤 1: 检查 PLY 文件属性")
from plyfile import PlyData

plydata = PlyData.read(CHECKPOINT_PLY)
ply_props = [p.name for p in plydata.elements[0].properties]
log(f"PLY 文件: {CHECKPOINT_PLY}")
log(f"Gaussian 数量: {plydata.elements[0].count}")
log(f"PLY 属性列表 ({len(ply_props)}):")
for i, name in enumerate(ply_props):
    log(f"  [{i:3d}] {name}")

# 检测 ASG 属性数量
asg_names = sorted([n for n in ply_props if n.startswith("f_asg_")], key=lambda x: int(x.split("_")[-1]))
detected_asg_degree = len(asg_names)
log(f"\n检测到 ASG 属性数量: {detected_asg_degree}")
max_sh_degree = 3  # TSGS 默认

# ── 步骤 2: 加载 GaussianModel ───────────────────────────
log_separator("步骤 2: 加载 GaussianModel")
import torch

from scene.gaussian_model import GaussianModel

gaussians = GaussianModel(sh_degree=max_sh_degree, asg_degree=detected_asg_degree if detected_asg_degree > 0 else None)
gaussians.load_ply(CHECKPOINT_PLY)

N = gaussians.get_xyz.shape[0]
log(f"GaussianModel 加载成功: {N} 个 Gaussian")
log(f"active_sh_degree = {gaussians.active_sh_degree}")
log(f"max_sh_degree = {gaussians.max_sh_degree}")
log(f"max_asg_degree = {gaussians.max_asg_degree}")

# ── 步骤 3: 导出激活后属性 ───────────────────────────────
log_separator("步骤 3: 导出激活后属性")

def to_np(t):
    return t.detach().cpu().numpy()

attrs = {}

# 3a. xyz
xyz = to_np(gaussians.get_xyz)
attrs["xyz"] = xyz
log(f"xyz:          shape={xyz.shape}  min={xyz.min():.4f}  max={xyz.max():.4f}  mean={xyz.mean():.4f}")

# 3b. scale (activated)
scale = to_np(gaussians.get_scaling)
attrs["scale"] = scale
log(f"scale:        shape={scale.shape}  min={scale.min():.6f}  max={scale.max():.6f}  mean={scale.mean():.6f}")

# 3c. rotation (normalized)
rot = to_np(gaussians.get_rotation)
attrs["rotation"] = rot
rot_norm = np.linalg.norm(rot, axis=1)
log(f"rotation:     shape={rot.shape}  quat_norm: min={rot_norm.min():.6f}  max={rot_norm.max():.6f}  mean={rot_norm.mean():.6f}")

# 3d. opacity (sigmoid activated)
opacity = to_np(gaussians.get_opacity)
attrs["opacity"] = opacity
log(f"opacity:      shape={opacity.shape}  min={opacity.min():.6f}  max={opacity.max():.6f}  mean={opacity.mean():.6f}")

# 3e. transparency (sigmoid activated)
transparency = to_np(gaussians.get_transparency)
attrs["transparency"] = transparency
log(f"transparency: shape={transparency.shape}  min={transparency.min():.6f}  max={transparency.max():.6f}  mean={transparency.mean():.6f}")

# 3f. features_dc
feat_dc = to_np(gaussians._features_dc)
attrs["features_dc"] = feat_dc
log(f"features_dc:  shape={feat_dc.shape}  min={feat_dc.min():.6f}  max={feat_dc.max():.6f}  mean={feat_dc.mean():.6f}")

# 3g. features_rest
feat_rest = to_np(gaussians._features_rest)
attrs["features_rest"] = feat_rest
log(f"features_rest: shape={feat_rest.shape}  min={feat_rest.min():.6f}  max={feat_rest.max():.6f}  mean={feat_rest.mean():.6f}")

# 3h. ASG features
if gaussians.max_asg_degree is not None and gaussians._features_asg is not None:
    feat_asg = to_np(gaussians._features_asg)
    attrs["features_asg"] = feat_asg
    log(f"features_asg: shape={feat_asg.shape}  min={feat_asg.min():.6f}  max={feat_asg.max():.6f}  mean={feat_asg.mean():.6f}")
else:
    feat_asg = None
    log("features_asg: NOT AVAILABLE (no ASG in this checkpoint)")

# ── 步骤 4: 计算 Canonical Normal ────────────────────────
log_separator("步骤 4: 计算 Canonical Normal")

rotation_matrices = gaussians.get_rotation_matrix()  # (N, 3, 3), quaternion_to_matrix 使用 wxyz
scaling_activated = gaussians.get_scaling
smallest_axis_idx = scaling_activated.min(dim=-1)[1]  # (N,)
idx_expanded = smallest_axis_idx[..., None, None].expand(-1, 3, -1)
canonical_normal = rotation_matrices.gather(2, idx_expanded).squeeze(dim=2)  # (N, 3)
canonical_normal_np = to_np(canonical_normal)

normal_norm = np.linalg.norm(canonical_normal_np, axis=1)
log(f"normal (raw): shape={canonical_normal_np.shape}")
log(f"  norm: min={normal_norm.min():.6f}  max={normal_norm.max():.6f}  mean={normal_norm.mean():.6f}")
log(f"  NaN count: {np.isnan(canonical_normal_np).sum()}")
log(f"  Inf count: {np.isinf(canonical_normal_np).sum()}")

# 单位化
normal_unit = canonical_normal_np / (normal_norm[:, np.newaxis] + 1e-8)
attrs["normal"] = normal_unit

norm_after = np.linalg.norm(normal_unit, axis=1)
log(f"normal (unit): min_norm={norm_after.min():.6f}  max_norm={norm_after.max():.6f}  mean_norm={norm_after.mean():.6f}")

# ── 步骤 5: 保存 NPZ ─────────────────────────────────────
log_separator("步骤 5: 保存 NPZ")
npz_path = os.path.join(OUTPUT_DIR, "canonical_attrs.npz")
np.savez_compressed(npz_path, **attrs)
log(f"NPZ 保存至: {npz_path}")
log(f"NPZ 中包含键: {list(attrs.keys())}")

# 验证重新加载
check = np.load(npz_path)
log(f"NPZ 验证: 成功加载，键列表 = {list(check.keys())}")
for k, v in check.items():
    log(f"  {k}: shape={v.shape}, dtype={v.dtype}")

# ── 步骤 6: 属性统计 ─────────────────────────────────────
log_separator("步骤 6: 属性统计")
stat_fields = ["xyz", "scale", "rotation", "opacity", "transparency", "normal"]
for field in stat_fields:
    data = attrs[field]
    log(f"\n{field}:")
    log(f"  shape: {data.shape}")
    log(f"  dtype: {data.dtype}")
    log(f"  min:   {data.min():.8f}")
    log(f"  max:   {data.max():.8f}")
    log(f"  mean:  {data.mean():.8f}")
    log(f"  std:   {data.std():.8f}")

if feat_asg is not None:
    log(f"\nfeatures_asg:")
    log(f"  shape: {feat_asg.shape}")
    log(f"  dtype: {feat_asg.dtype}")
    log(f"  min:   {feat_asg.min():.8f}")
    log(f"  max:   {feat_asg.max():.8f}")
    log(f"  mean:  {feat_asg.mean():.8f}")
    log(f"  std:   {feat_asg.std():.8f}")

# ── 步骤 7: 保存日志 ─────────────────────────────────────
log_path = os.path.join(OUTPUT_DIR, "export_log.txt")
with open(log_path, "w") as f:
    f.write("\n".join(log_lines))
log(f"\n日志已保存: {log_path}")

# ── 步骤 8: 生成 JSON 摘要 ──────────────────────────────
summary = {
    "gaussian_count": int(N),
    "ply_path": CHECKPOINT_PLY,
    "npz_path": npz_path,
    "ply_properties": ply_props,
    "has_asg": gaussians.max_asg_degree is not None,
    "detected_asg_degree": detected_asg_degree,
    "max_sh_degree": max_sh_degree,
    "exported_attrs": list(attrs.keys()),
}
summary_path = os.path.join(OUTPUT_DIR, "export_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
log(f"摘要已保存: {summary_path}")

log_separator("导出完成")
