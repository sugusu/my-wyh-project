import argparse
import json
import sys
from pathlib import Path

from stage0r_utils import OUT_BASELINE, ensure_dirs, write_json, write_md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--iterations", nargs="+", type=int, required=True)
    args = parser.parse_args()
    ensure_dirs()
    pending = []
    for it in args.iterations:
        out = {
            "iteration": it,
            "status": "FORMAL_GEOMETRY_PENDING",
            "reason": "Formal mesh extraction/evaluation is not available for this incomplete official 30k trajectory.",
            "required_metrics": ["Chamfer Distance", "accuracy", "completeness", "F-score @ 0.5% object diameter", "F-score @ 1.0% object diameter"],
        }
        write_json(OUT_BASELINE / f"official_scene01_{it//1000}k_geometry.json", out)
        pending.append(out)
    write_md(OUT_BASELINE / "official_scene01_geometry_report.md", [
        "# Official Scene 01 Geometry Evaluation",
        "",
        "Conclusion: `FORMAL_GEOMETRY_PENDING`",
        "",
        "The official 30k model root is incomplete/failed, so Stage 0R cannot claim formal geometry readiness.",
    ])
    sys.exit(2)


if __name__ == "__main__":
    main()
