#!/usr/bin/env python3
"""Download scene_03 from hf-mirror.com using requests + multithreading."""
import os, sys, json, time, requests, concurrent.futures
from pathlib import Path

BASE_URL = "https://hf-mirror.com"
DATASET = "Longxiang-ai/TransLab"
SCENE = "scene_03"
OUT_DIR = Path("/data/wyh/RecycleGS/data/translab_full") / SCENE
NUM_THREADS = 8

def list_all_files():
    """Recursively list all files for scene_03 via HF API with pagination."""
    api_url = f"{BASE_URL}/api/datasets/{DATASET}"
    print(f"Fetching dataset info from {api_url}...")
    r = requests.get(api_url, timeout=60)
    r.raise_for_status()
    data = r.json()
    siblings = data.get("siblings", [])
    scene_files = [s["rfilename"] for s in siblings if s["rfilename"].startswith(f"{SCENE}/")]
    return scene_files

def download_file(rfilename):
    """Download a single file."""
    url = f"{BASE_URL}/datasets/{DATASET}/resolve/main/{rfilename}"
    rel_path = Path(rfilename).relative_to(SCENE)
    out_path = OUT_DIR / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return f"SKIP {rel_path}"
    try:
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        return f"OK   {rel_path} ({len(r.content)} bytes)"
    except Exception as e:
        return f"FAIL {rel_path}: {e}"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    print("\n=== Listing scene_03 files ===")
    files = list_all_files()
    print(f"Found {len(files)} files")

    print(f"\n=== Downloading with {NUM_THREADS} threads ===")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
        fut_to_file = {pool.submit(download_file, f): f for f in files}
        for i, fut in enumerate(concurrent.futures.as_completed(fut_to_file)):
            res = fut.result()
            results.append(res)
            if i % 20 == 0:
                print(f"  [{i+1}/{len(files)}] {res}")

    ok = sum(1 for r in results if r.startswith("OK"))
    skip = sum(1 for r in results if r.startswith("SKIP"))
    fail = sum(1 for r in results if r.startswith("FAIL"))
    print(f"\n=== Summary ===")
    print(f"  OK: {ok}, SKIP: {skip}, FAIL: {fail}")
    if fail > 0:
        for r in results:
            if r.startswith("FAIL"):
                print(f"  {r}")

    print("\n=== Verifying ===")
    verify()

def verify():
    check_dir(OUT_DIR / "images", "images", 400)
    check_dir(OUT_DIR / "delights", "delights", 400)
    check_dir(OUT_DIR / "masks", "masks", None)
    check_dir(OUT_DIR / "normals", "normals", 400)
    check_dir(OUT_DIR / "transparent_masks", "transparent_masks", 400)
    meshes = list((OUT_DIR / "meshes").glob("*")) if (OUT_DIR / "meshes").exists() else []
    print(f"  meshes: {len(meshes)} files - {[m.name for m in meshes]}")
    sparse_dir = OUT_DIR / "sparse"
    if sparse_dir.exists():
        sparse_files = [f.name for f in sparse_dir.iterdir() if f.is_file()]
        print(f"  sparse: {sparse_files}")

def check_dir(d, name, expected):
    if d.exists():
        files = list(d.iterdir())
        n = len(files)
        suffix = f" (expected {expected})" if expected else ""
        status = "OK" if not expected or n == expected else "MISMATCH"
        print(f"  {name}: {n} files{suffix} [{status}]")
    else:
        print(f"  {name}: DIRECTORY NOT FOUND")

if __name__ == "__main__":
    main()
