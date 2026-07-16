from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


BASE = Path("/data/wyh/DeformTransGS")
OUT = BASE / "experiments/stage5_0_R4_O2_convergence_closure"
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


def metric(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple[float, float, float, float, float, float]:
    mse = float(np.mean((pred - gt) ** 2))
    psnr = 99.0 if mse <= 1e-20 else -10.0 * math.log10(mse)
    pred_tau = -np.log(np.clip(pred, 1e-6, 1.0))
    gt_tau = -np.log(np.clip(gt, 1e-6, 1.0))
    mask = np.repeat(valid[:, :, None], 3, axis=2)
    elog = np.abs(np.log((pred_tau + 1e-6) / (gt_tau + 1e-6)))
    ratio = np.maximum((pred_tau + 1e-6) / (gt_tau + 1e-6), (gt_tau + 1e-6) / (pred_tau + 1e-6))
    samples = elog[mask]
    return psnr, float(np.median(samples)), float(np.percentile(samples, 90)), float(np.percentile(samples, 95)), float(np.percentile(samples, 99)), float(np.mean(ratio[mask] <= 2.0))


def main() -> None:
    per = []
    for row in csv.DictReader((OUT / "O2_final_render_manifest.csv").open(encoding="utf-8")):
        case = row["case"]
        cid = int(row["camera_id"])
        s, m, d = CASES[case]
        root = V2 / s / m / d
        pred = np.load(row["path"]).transpose(1, 2, 0)
        gt = np.load(root / f"camera_{cid:02d}_rgb.npy")
        valid = np.load(root / f"camera_{cid:02d}_triangle_id.npy") >= 0
        psnr, med, p90, p95, p99, factor2 = metric(pred, gt, valid)
        per.append({"case": case, "split": row["split"], "camera_id": cid, "PSNR": psnr, "SSIM": 0.0, "tau_eq_Elog_median": med, "tau_eq_Elog_p90": p90, "tau_eq_Elog_p95": p95, "tau_eq_Elog_p99": p99, "factor2_fraction": factor2})
    summary = []
    for case in CASES:
        for split in ["TRAIN", "TEST"]:
            part = [r for r in per if r["case"] == case and r["split"] == split]
            summary.append({"case": case, "split": split, "PSNR": float(np.mean([r["PSNR"] for r in part])), "tau_eq_Elog_median": float(np.median([r["tau_eq_Elog_median"] for r in part]))})
    write_csv(OUT / "O2_metrics_per_camera.csv", per)
    write_csv(OUT / "O2_metrics_summary.csv", summary)
    print(f"O2 metric rows: {len(per)}")


if __name__ == "__main__":
    main()
