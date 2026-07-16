import os, ast

OUT_DIR = "/data/wyh/RecycleGS/outputs/debug/stage0r"
OLD_CFG_PATH = "/data/wyh/RecycleGS/baselines/tsgs_scene01_full/cfg_args"

# Official recipe flags from run_translab.sh (training portion only)
OFFICIAL_FLAGS = {
    "delight": True,
    "normal": True,
    "eval": True,
    "use_asg": True,
    "resolution": 2,
    "iterations": 30000,
    "sd_normal_until_iter": 30000,
    "delight_iterations": 15000,
    "normal_cos_threshold_iter": 3000,
    "ncc_loss_from_iter": 7000,
    "nofix_position": True,
    "nofix_scaling": True,
    "nofix_rotation": True,
    "seed": 42,
}

FIELDS = [
    "iterations", "delight", "normal", "eval", "use_asg",
    "resolution", "seed", "nofix_position", "nofix_scaling",
    "nofix_rotation", "sd_normal_until_iter", "delight_iterations",
    "normal_cos_threshold_iter", "ncc_loss_from_iter",
]

def parse_old_cfg(path):
    with open(path) as f:
        raw = f.read().strip()
    # cfg_args stores Namespace(...) as a string; safe to eval
    ns = eval(raw, {"Namespace": lambda **kw: kw})
    return ns

def main():
    old = parse_old_cfg(OLD_CFG_PATH)
    rows = []
    for f in FIELDS:
        old_val = old.get(f, "<missing>")
        off_val = OFFICIAL_FLAGS.get(f, "<not specified>")
        match = "MATCH" if str(old_val) == str(off_val) else "DIFF"
        rows.append((f, str(old_val), str(off_val), match))

    # CSV
    csv_lines = ["field,old_value,official_value,status"]
    for r in rows:
        csv_lines.append(f"{r[0]},{r[1]},{r[2]},{r[3]}")
    csv_path = os.path.join(OUT_DIR, "old_vs_official_config.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"Wrote {csv_path}")

    # MD
    md_lines = [
        "# Old vs Official TSGS Config Audit",
        "",
        "| Field | Old Value | Official Value | Status |",
        "|-------|-----------|----------------|--------|",
    ]
    for r in rows:
        md_lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    md_path = os.path.join(OUT_DIR, "old_vs_official_config.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Wrote {md_path}")

    print("\nSummary of differences:")
    for r in rows:
        if r[3] == "DIFF":
            print(f"  {r[0]}: old={r[1]}  official={r[2]}")

if __name__ == "__main__":
    main()
