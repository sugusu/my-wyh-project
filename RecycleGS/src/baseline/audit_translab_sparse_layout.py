import argparse
import shutil
from pathlib import Path

from stage0r_utils import OUT_STAGE0RC, ensure_dirs, sha256_file, write_json, write_md


NAMES = ["cameras.bin", "images.bin", "points3D.bin"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True)
    args = parser.parse_args()
    ensure_dirs()
    scene = Path(args.scene_dir)
    rows = []
    needs_copy = False
    for name in NAMES:
        top = scene / "sparse" / name
        nested = scene / "sparse" / "0" / name
        top_sha = sha256_file(top)
        nested_sha = sha256_file(nested)
        ok = top.exists() and nested.exists() and top_sha == nested_sha
        needs_copy = needs_copy or not ok
        rows.append({"file": name, "top_exists": top.exists(), "nested_exists": nested.exists(), "top_sha256": top_sha, "nested_sha256": nested_sha, "match": ok})
    copied = False
    if needs_copy:
        for item in (scene / "sparse" / "0").iterdir():
            dst = scene / "sparse" / item.name
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
        copied = True
        rows = []
        for name in NAMES:
            top = scene / "sparse" / name
            nested = scene / "sparse" / "0" / name
            rows.append({"file": name, "top_exists": top.exists(), "nested_exists": nested.exists(), "top_sha256": sha256_file(top), "nested_sha256": sha256_file(nested), "match": sha256_file(top) == sha256_file(nested)})
    ok_all = all(r["match"] for r in rows)
    write_json(OUT_STAGE0RC / "sparse_layout_audit.json", {"scene_dir": str(scene), "copied_sparse0_to_sparse": copied, "ok": ok_all, "rows": rows})
    lines = ["# TransLab Sparse Layout Audit", "", f"copied_sparse0_to_sparse: `{copied}`", f"SPARSE_LAYOUT_OK: `{ok_all}`", "", "| File | Top exists | Nested exists | Top SHA256 | Nested SHA256 | Match |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['file']}` | `{r['top_exists']}` | `{r['nested_exists']}` | `{r['top_sha256']}` | `{r['nested_sha256']}` | `{r['match']}` |")
    write_md(OUT_STAGE0RC / "sparse_layout_audit_report.md", lines)


if __name__ == "__main__":
    main()
