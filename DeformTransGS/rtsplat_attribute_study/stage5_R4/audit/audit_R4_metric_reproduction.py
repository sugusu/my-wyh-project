from __future__ import annotations

import csv
import math
import random
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


def metric(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((pred - gt) ** 2))
    psnr = 99.0 if mse <= 1e-20 else -10.0 * math.log10(mse)
    pred_tau = -np.log(np.clip(pred, 1e-6, 1.0))
    gt_tau = -np.log(np.clip(gt, 1e-6, 1.0))
    mask = np.repeat(valid[:, :, None], 3, axis=2)
    elog = np.abs(np.log((pred_tau + 1e-6) / (gt_tau + 1e-6)))
    return psnr, float(np.median(elog[mask]))


def select_rows(rows: list[dict]) -> list[dict]:
    tests = [row for row in rows if row["split"] == "TEST"]
    selected = []
    for case in CASES:
        selected.append(next(row for row in tests if row["case"] == case))
    remaining = [row for row in tests if row not in selected]
    rng = random.Random(20260714)
    selected.extend(rng.sample(remaining, 3))
    return sorted(selected, key=lambda row: (row["case"], int(row["camera_id"])))


def main() -> None:
    manifest = {(r["case"], r["split"], int(r["camera_id"])): r for r in csv.DictReader((OUT / "R4_render_manifest.csv").open(encoding="utf-8"))}
    rows = []
    recorded_rows = list(csv.DictReader((OUT / "R4_metrics_per_camera.csv").open(encoding="utf-8")))
    for recorded in select_rows(recorded_rows):
        case = recorded["case"]
        split = recorded["split"]
        cid = int(recorded["camera_id"])
        item = manifest[(case, split, cid)]
        surface, material, deformation = CASES[case]
        gt_root = V2 / surface / material / deformation
        pred = np.load(item["path"]).transpose(1, 2, 0)
        gt = np.load(gt_root / f"camera_{cid:02d}_rgb.npy")
        valid = np.load(gt_root / f"camera_{cid:02d}_triangle_id.npy") >= 0
        psnr, elog = metric(pred, gt, valid)
        rows.append({
            "case": case,
            "split": split,
            "camera_id": cid,
            "PSNR_diff": abs(psnr - float(recorded["PSNR"])),
            "tau_eq_diff": abs(elog - float(recorded["tau_eq_Elog_median"])),
        })
    with (OUT / "R4_metric_reproduction.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, ["case", "split", "camera_id", "PSNR_diff", "tau_eq_diff"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"metric reproduction rows: {len(rows)}")


if __name__ == "__main__":
    main()
