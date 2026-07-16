from __future__ import annotations

import csv
import math
import random
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


def metric(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((pred - gt) ** 2))
    psnr = 99.0 if mse <= 1e-20 else -10.0 * math.log10(mse)
    pred_tau = -np.log(np.clip(pred, 1e-6, 1.0))
    gt_tau = -np.log(np.clip(gt, 1e-6, 1.0))
    mask = np.repeat(valid[:, :, None], 3, axis=2)
    elog = np.abs(np.log((pred_tau + 1e-6) / (gt_tau + 1e-6)))
    return psnr, float(np.median(elog[mask]))


def main() -> None:
    manifest = {(r["case"], r["split"], int(r["camera_id"])): r for r in csv.DictReader((OUT / "O2_final_render_manifest.csv").open(encoding="utf-8"))}
    rows = list(csv.DictReader((OUT / "O2_metrics_per_camera.csv").open(encoding="utf-8")))
    tests = [r for r in rows if r["split"] == "TEST"]
    chosen = [next(r for r in tests if r["case"] == case) for case in CASES]
    chosen.extend(random.Random(20260714).sample([r for r in tests if r not in chosen], 3))
    repro = []
    for rec in chosen:
        case = rec["case"]
        cid = int(rec["camera_id"])
        s, m, d = CASES[case]
        item = manifest[(case, "TEST", cid)]
        root = V2 / s / m / d
        pred = np.load(item["path"]).transpose(1, 2, 0)
        gt = np.load(root / f"camera_{cid:02d}_rgb.npy")
        valid = np.load(root / f"camera_{cid:02d}_triangle_id.npy") >= 0
        psnr, med = metric(pred, gt, valid)
        repro.append({"case": case, "split": "TEST", "camera_id": cid, "PSNR_diff": abs(psnr - float(rec["PSNR"])), "tau_eq_diff": abs(med - float(rec["tau_eq_Elog_median"]))})
    with (OUT / "O2_metric_reproduction.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, ["case", "split", "camera_id", "PSNR_diff", "tau_eq_diff"])
        writer.writeheader()
        writer.writerows(repro)
    print(f"O2 reproduction rows: {len(repro)}")


if __name__ == "__main__":
    main()
