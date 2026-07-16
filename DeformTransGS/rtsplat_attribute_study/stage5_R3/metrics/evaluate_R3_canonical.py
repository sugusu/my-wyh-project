from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    out = Path("/data/wyh/DeformTransGS/experiments/stage5_0_R3_native_state_canonical_gate")
    for name in ["R3_canonical_metrics_per_camera.csv", "R3_canonical_metrics_summary.csv", "R3_metric_reproduction.csv"]:
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=["status", "reason"]).writeheader()


if __name__ == "__main__":
    main()
