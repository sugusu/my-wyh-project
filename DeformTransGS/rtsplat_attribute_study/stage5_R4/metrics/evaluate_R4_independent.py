from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


BASE = Path("/data/wyh/DeformTransGS")
OUT = BASE / "experiments/stage5_0_R4_rtsplat_v2_canonical_capacity"
V2 = BASE / "experiments/stage5_0_R3_C2_perspective_v2_validity/perspective_clean_gt_v2"
CASES = {
    "K0": ("S0_PLANAR_SHEET", "MAT0_NEUTRAL_FIXED_THICKNESS", "D0_IDENTITY"),
    "K1": ("S0_PLANAR_SHEET", "MAT1_NEUTRAL_MASS_CONSERVING", "D0_IDENTITY"),
    "K2": ("S1_WAVY_MEMBRANE", "MAT2_TINTED_MASS_CONSERVING", "D0_IDENTITY"),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def ssim_rgb(pred: np.ndarray, gt: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0))
    except Exception:
        vals = []
        for ch in range(3):
            x = pred[:, :, ch].astype(np.float64)
            y = gt[:, :, ch].astype(np.float64)
            mux, muy = x.mean(), y.mean()
            vx, vy = x.var(), y.var()
            cov = ((x - mux) * (y - muy)).mean()
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            vals.append(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux ** 2 + muy ** 2 + c1) * (vx + vy + c2)))
        return float(np.mean(vals))


def metric(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    mse = float(np.mean((pred - gt) ** 2))
    psnr = 99.0 if mse <= 1e-20 else -10.0 * math.log10(mse)
    pred_tau = -np.log(np.clip(pred, 1e-6, 1.0))
    gt_tau = -np.log(np.clip(gt, 1e-6, 1.0))
    mask = np.repeat(valid[:, :, None], 3, axis=2)
    elog = np.abs(np.log((pred_tau + 1e-6) / (gt_tau + 1e-6)))
    ratio = np.maximum((pred_tau + 1e-6) / (gt_tau + 1e-6), (gt_tau + 1e-6) / (pred_tau + 1e-6))
    samples = elog[mask]
    return (
        psnr,
        ssim_rgb(pred, gt),
        float(np.median(samples)),
        float(np.percentile(samples, 90)),
        float(np.percentile(samples, 95)),
        float(np.percentile(samples, 99)),
        float(np.mean(ratio[mask] <= 2.0)),
    )


def main() -> None:
    rows = []
    for item in csv.DictReader((OUT / "R4_render_manifest.csv").open(encoding="utf-8")):
        case = item["case"]
        cid = int(item["camera_id"])
        surface, material, deformation = CASES[case]
        gt_root = V2 / surface / material / deformation
        pred = np.load(item["path"]).transpose(1, 2, 0)
        gt = np.load(gt_root / f"camera_{cid:02d}_rgb.npy")
        valid = np.load(gt_root / f"camera_{cid:02d}_triangle_id.npy") >= 0
        psnr, ssim, elog, p90, p95, p99, factor2 = metric(pred, gt, valid)
        rows.append({
            "case": case,
            "split": item["split"],
            "camera_id": cid,
            "PSNR": psnr,
            "SSIM": ssim,
            "tau_eq_Elog_median": elog,
            "tau_eq_Elog_p90": p90,
            "tau_eq_Elog_p95": p95,
            "tau_eq_Elog_p99": p99,
            "factor2_fraction": factor2,
        })
    summary = []
    for case in CASES:
        for split in ["TRAIN", "TEST"]:
            part = [row for row in rows if row["case"] == case and row["split"] == split]
            summary.append({
                "case": case,
                "split": split,
                "PSNR": float(np.mean([row["PSNR"] for row in part])),
                "tau_eq_Elog_median": float(np.median([row["tau_eq_Elog_median"] for row in part])),
            })
    write_csv(OUT / "R4_metrics_per_camera.csv", rows)
    write_csv(OUT / "R4_metrics_summary.csv", summary)
    print(f"independent metric rows: {len(rows)}")


if __name__ == "__main__":
    main()
