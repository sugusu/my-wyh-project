#!/usr/bin/env python3
"""Data integrity check for scene_03."""
import json, os, hashlib, sys
from pathlib import Path
from PIL import Image

SCENE_DIR = Path("/data/wyh/RecycleGS/data/translab_full/scene_03")
OUT_DIR = Path("/data/wyh/RecycleGS/outputs/debug")

def check_images(d, expected_exts=(".png", ".jpg", ".jpeg")):
    if not d.exists():
        return {"status": "MISSING", "count": 0, "errors": [f"Directory not found: {d}"]}
    files = sorted(d.iterdir())
    results = []
    errors = []
    md5s = set()
    symlinks = 0
    for f in files:
        if f.is_symlink():
            symlinks += 1
            errors.append(f"Symlink: {f.name}")
        try:
            img = Image.open(f)
            img.verify()
            with open(f, "rb") as fh:
                md5 = hashlib.md5(fh.read()).hexdigest()
            if md5 in md5s:
                errors.append(f"Duplicate MD5: {f.name}")
            md5s.add(md5)
            results.append({"file": f.name, "md5": md5, "size": f.stat().st_size})
        except Exception as e:
            errors.append(f"Corrupt: {f.name} - {e}")
    return {
        "status": "OK" if not errors else "ERRORS",
        "count": len(files),
        "valid": len(results),
        "symlinks": symlinks,
        "unique_md5": len(md5s),
        "errors": errors[:20],
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {
        "scene": "scene_03",
        "data_dir": str(SCENE_DIR),
        "checks": {}
    }

    for name, subdir in [
        ("images", "images"),
        ("delights", "delights"),
        ("masks", "masks"),
        ("normals", "normals"),
        ("transparent_masks", "transparent_masks"),
        ("transnormals", "transnormals"),
    ]:
        print(f"Checking {name}...")
        report["checks"][name] = check_images(SCENE_DIR / subdir)

    # Check meshes
    mesh_dir = SCENE_DIR / "meshes"
    if mesh_dir.exists():
        meshes = list(mesh_dir.iterdir())
        report["checks"]["meshes"] = {
            "status": "OK",
            "count": len(meshes),
            "files": [m.name for m in meshes],
        }
    else:
        report["checks"]["meshes"] = {"status": "MISSING", "count": 0}

    # Check sparse
    sparse_dir = SCENE_DIR / "sparse"
    if sparse_dir.exists():
        sparse_files = sorted(sparse_dir.iterdir())
        required = ["cameras.bin", "images.bin", "points3D.bin"]
        present = all((sparse_dir / f).exists() for f in required)
        report["checks"]["sparse"] = {
            "status": "OK" if present else "MISSING_FILES",
            "count": len(sparse_files),
            "has_colmap_bins": present,
            "files": [f.name for f in sparse_files],
        }
    else:
        report["checks"]["sparse"] = {"status": "MISSING", "count": 0}

    # Save JSON
    json_path = OUT_DIR / "scene03_data_check.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Generate MD
    md_lines = [
        f"# Scene 03 Data Integrity Check",
        f"",
        f"| Directory | Status | Count | Valid | Symlinks | Unique MD5 | Errors |",
        f"|-----------|--------|-------|-------|----------|------------|--------|",
    ]
    for name in ["images", "delights", "masks", "normals", "transparent_masks", "transnormals"]:
        c = report["checks"].get(name, {})
        n_err = len(c.get("errors", []))
        md_lines.append(
            f"| {name} | {c.get('status', 'N/A')} | {c.get('count', 0)} | "
            f"{c.get('valid', 0)} | {c.get('symlinks', 0)} | "
            f"{c.get('unique_md5', 0)} | {n_err} |"
        )

    c = report["checks"].get("meshes", {})
    md_lines.append(f"| meshes | {c.get('status', 'N/A')} | {c.get('count', 0)} | - | - | - | - |")
    c = report["checks"].get("sparse", {})
    md_lines.append(f"| sparse | {c.get('status', 'N/A')} | {c.get('count', 0)} | - | - | - | - |")
    md_lines.append("")
    md_lines.append("## Summary")
    all_ok = all(
        report["checks"].get(n, {}).get("status") == "OK"
        for n in ["images", "delights", "masks", "normals", "transparent_masks", "meshes", "sparse"]
    )
    if all_ok:
        md_lines.append("- All directories passed integrity check.")
    else:
        md_lines.append("- Some directories have issues. See details above.")

    md_path = OUT_DIR / "scene03_data_check.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved MD: {md_path}")
    print(f"\nAll OK: {all_ok}")

if __name__ == "__main__":
    main()
