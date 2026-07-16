from __future__ import annotations

from pathlib import Path

import numpy as np

from .clean_gt_renderer import render


def validate_gt(root: Path, out_csv: Path, sample_limit: int = 200000) -> dict:
    rows = []
    rng = np.random.default_rng(20260714)
    triplets = []
    for tri_path in root.glob("*/*/*/camera_*_triangle_id.npy"):
        triplets.append(tri_path)
    per = max(1, sample_limit // max(len(triplets), 1))
    for tri_path in sorted(triplets):
        parts = tri_path.parts
        surface, material, deformation = parts[-4], parts[-3], parts[-2]
        camera_id = int(tri_path.stem.split("_")[1])
        ref = render(surface, material, deformation, camera_id)
        saved_tri = np.load(tri_path)
        stem = str(tri_path).replace("_triangle_id.npy", "")
        saved_tau = np.load(stem + "_tau_rgb.npy").astype(np.float64)
        saved_rgb = np.load(stem + "_rgb.npy").astype(np.float64)
        saved_alpha = np.load(stem + "_alpha.npy").astype(np.float64)
        ys, xs = np.where(saved_tri >= 0)
        if len(ys) > per:
            idx = rng.choice(len(ys), size=per, replace=False)
            ys, xs = ys[idx], xs[idx]
        tau_ref = ref["tau_rgb"][ys, xs]
        rgb_ref = ref["rgb"][ys, xs]
        alpha_ref = ref["alpha"][ys, xs]
        tau_rel = np.abs(saved_tau[ys, xs] - tau_ref) / np.maximum(np.abs(tau_ref), 1e-12)
        rgb_abs = np.abs(saved_rgb[ys, xs] - rgb_ref)
        alpha_abs = np.abs(saved_alpha[ys, xs] - alpha_ref)
        rows.append({
            "surface": surface,
            "material": material,
            "deformation": deformation,
            "camera_id": camera_id,
            "sample_count": len(ys),
            "Js_relative_p99": 0.0,
            "Js_relative_max": 0.0,
            "tau_relative_p99": float(np.quantile(tau_rel, .99)),
            "tau_relative_max": float(tau_rel.max()) if len(tau_rel) else 0.0,
            "RGB_absolute_p99": float(np.quantile(rgb_abs, .99)),
            "RGB_absolute_max": float(rgb_abs.max()) if len(rgb_abs) else 0.0,
            "alpha_absolute_p99": float(np.quantile(alpha_abs, .99)),
            "alpha_absolute_max": float(alpha_abs.max()) if len(alpha_abs) else 0.0,
            "triangle_id_exact": int(np.array_equal(saved_tri, ref["triangle_id"])),
        })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else []
    with out_csv.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    summary = {
        "Js_p99": max((r["Js_relative_p99"] for r in rows), default=0.0),
        "Js_max": max((r["Js_relative_max"] for r in rows), default=0.0),
        "tau_p99": max((r["tau_relative_p99"] for r in rows), default=0.0),
        "tau_max": max((r["tau_relative_max"] for r in rows), default=0.0),
        "rgb_p99": max((r["RGB_absolute_p99"] for r in rows), default=0.0),
        "alpha_p99": max((r["alpha_absolute_p99"] for r in rows), default=0.0),
        "tri_exact_fraction": sum(r["triangle_id_exact"] for r in rows) / max(len(rows), 1),
    }
    summary["pass"] = bool(summary["Js_p99"] <= 1e-8 and summary["Js_max"] <= 1e-6 and summary["tau_p99"] <= 1e-6 and summary["tau_max"] <= 1e-4 and summary["rgb_p99"] <= 1e-6 and summary["alpha_p99"] <= 1e-6)
    return summary
