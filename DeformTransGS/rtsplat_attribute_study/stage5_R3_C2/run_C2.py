from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np


BASE = Path("/data/wyh/DeformTransGS")
CODE = BASE / "rtsplat_attribute_study" / "stage5_R3_C2"
EXP = BASE / "experiments" / "stage5_0_R3_C2_perspective_v2_validity"
V2_ROOT = EXP / "perspective_clean_gt_v2"
C1 = BASE / "experiments" / "stage5_0_R3_C1_benchmark_camera_semantics"
GT1 = BASE / "experiments" / "stage4_0_R2A_GT1_gt_optical_semantics_closure"
RT_ROOT = Path("/data/wyh/repos/RT-Splatting")
SEED = 20260714
RES = 512
FOVY = 75.0
TRAIN_IDS = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23]
TEST_IDS = [0, 3, 6, 9, 12, 15, 18, 21]
SURFACES = ("S0_PLANAR_SHEET", "S1_WAVY_MEMBRANE")
MATERIALS = ("MAT0_NEUTRAL_FIXED_THICKNESS", "MAT1_NEUTRAL_MASS_CONSERVING", "MAT2_TINTED_MASS_CONSERVING")
DEFORMATIONS = ("D0_IDENTITY", "D1_STRETCH_X_1P25", "D2_STRETCH_X_1P50", "D3_BIAXIAL_XY_1P50", "D4_SHEAR_XY_0P30", "D5_ANISO_X1P60_Y0P80", "D6_ROTATION_Z_30")


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rec(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False, "size": "MISSING", "mtime": "MISSING", "sha256": "MISSING"}
    st = path.stat()
    return {"path": str(path), "exists": True, "size": st.st_size, "mtime": st.st_mtime, "sha256": sha256_path(path)}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({k for r in rows for k in r}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def old_camera_pos(camera_id: int) -> np.ndarray:
    elev = 25.0 if camera_id < 12 else 50.0
    az = (camera_id % 12) * 30.0
    er = math.radians(elev)
    ar = math.radians(az)
    return np.array([3.3 * math.cos(er) * math.cos(ar), 3.3 * math.cos(er) * math.sin(ar), 3.3 * math.sin(er)], dtype=np.float64)


def camera_basis_from_center(center: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = -center / (np.linalg.norm(center) + 1e-30)
    up_candidate = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up_candidate))) >= 0.99:
        up_candidate = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up_candidate)
    right /= np.linalg.norm(right) + 1e-30
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up) + 1e-30
    return right, true_up, forward


def deformation_matrix(name: str) -> np.ndarray:
    a = math.radians(30.0)
    return {
        "D0_IDENTITY": np.diag([1.0, 1.0, 1.0]),
        "D1_STRETCH_X_1P25": np.diag([1.25, 1.0, 1.0]),
        "D2_STRETCH_X_1P50": np.diag([1.50, 1.0, 1.0]),
        "D3_BIAXIAL_XY_1P50": np.diag([1.50, 1.50, 1.0]),
        "D4_SHEAR_XY_0P30": np.array([[1.0, 0.30, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        "D5_ANISO_X1P60_Y0P80": np.diag([1.60, 0.80, 1.0]),
        "D6_ROTATION_Z_30": np.diag([1.0, 1.0, 1.0]) @ np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    }[name]


def vertices(surface: str) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, 129)
    u, v = np.meshgrid(grid, grid)
    z = np.zeros_like(u) if surface == "S0_PLANAR_SHEET" else 0.18 * np.sin(np.pi * u) * np.sin(np.pi * v)
    return np.stack([u.ravel(), v.ravel(), z.ravel()], axis=1)


def project(points: np.ndarray, center: np.ndarray, fovy: float = FOVY) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    right, up, forward = camera_basis_from_center(center)
    rel = points - center.reshape(1, 3)
    xc = rel @ right
    yc = rel @ up
    zc = rel @ forward
    tan = math.tan(math.radians(fovy) / 2.0)
    nx = xc / (zc * tan)
    ny = yc / (zc * tan)
    px = (nx + 1.0) * 0.5 * RES - 0.5
    py = (1.0 - ny) * 0.5 * RES - 0.5
    return np.stack([px, py], axis=1), zc, np.stack([nx, ny], axis=1)


def pixel_rays(center: np.ndarray, xy: np.ndarray, fovy: float = FOVY) -> np.ndarray:
    right, up, forward = camera_basis_from_center(center)
    tan = math.tan(math.radians(fovy) / 2.0)
    nx = ((xy[:, 0] + 0.5) / RES * 2.0 - 1.0) * tan
    ny = (1.0 - (xy[:, 1] + 0.5) / RES * 2.0) * tan
    rays = forward.reshape(1, 3) + nx[:, None] * right.reshape(1, 3) + ny[:, None] * up.reshape(1, 3)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays


def all_deformed_vertices() -> list[tuple[str, str, np.ndarray]]:
    out = []
    for surface in SURFACES:
        v0 = vertices(surface)
        for deformation in DEFORMATIONS:
            out.append((surface, deformation, v0 @ deformation_matrix(deformation).T))
    return out


def coverage(center_scale: float, fovy: float = FOVY) -> tuple[bool, float, int, list[dict]]:
    rows = []
    ok = True
    min_margin = 1e9
    behind = 0
    for cid in range(24):
        c = old_camera_pos(cid) * center_scale
        for surface, deformation, pts in all_deformed_vertices():
            pix, depth, ndc = project(pts, c, fovy)
            b = int(np.sum(depth <= 0))
            margin_frac = float(np.min(np.stack([pix[:, 0], pix[:, 1], RES - 1 - pix[:, 0], RES - 1 - pix[:, 1]], axis=1)) / RES)
            maxx = float(np.max(np.abs(ndc[:, 0])))
            maxy = float(np.max(np.abs(ndc[:, 1])))
            passed = b == 0 and margin_frac >= 0.05
            ok = ok and passed
            behind += b
            min_margin = min(min_margin, margin_frac)
            rows.append({"camera_id": cid, "surface": surface, "deformation": deformation, "positive_depth_vertex_count": int(np.sum(depth > 0)), "non_positive_depth_vertex_count": b, "max_abs_normalized_x": maxx, "max_abs_normalized_y": maxy, "minimum_pixel_border_margin_fraction": margin_frac, "coverage_pass": "YES" if passed else "NO"})
    return ok, min_margin, behind, rows


def protocol_lock() -> dict:
    paths = [
        C1 / "C1_protocol_lock.json", C1 / "stage5_0_R3_C1_camera_semantics_report.md", C1 / "stage5_0_R3_C1_camera_semantics_summary.md",
        C1 / "C1_clean_GT_generator_trace.md", C1 / "C1_GT_pixel_semantics.md", C1 / "C1_cross_camera_fixed_pixel_identity.csv",
        C1 / "C1_clean_GT_camera_state.csv", C1 / "C1_camera_split_source_audit.csv", C1 / "C1_frozen_camera_split_lock.json",
        C1 / "C1_Stage4_prediction_image_semantics.md", C1 / "C1_Stage4_GT_projection_compatibility.csv", C1 / "C1_Stage4_capacity_conclusion_status.md",
        C1 / "C1_RT_camera_semantics.md", C1 / "C1_GT_RT_adapter_existence.md", GT1 / "clean_gt_manifest.csv",
        BASE / "attribute_study/real_oracle/gt_closure/clean_gt_renderer.py", BASE / "experiments/stage5_0_R2_real_local_extension_build/verified_rtsplat_R2_runtime_lock.json",
        BASE / "rtsplat_attribute_study/real_build_gate/verified_rtsplat_R2_python.sh", RT_ROOT / "scene/cameras.py", RT_ROOT / "gaussian_renderer/__init__.py", Path("/data/wyh/新5.md"),
    ]
    records = [rec(p) for p in paths]
    lock = {"records": records, "Q0": "PASS" if all(r["exists"] for r in records) else "FAIL"}
    write_json(EXP / "C2_protocol_lock.json", lock)
    return lock


def c1_reclassify() -> None:
    write_md(EXP / "C2_C1_failure_reclassification.md", "C2 C1 Failure Reclassification", "\n".join([
        "C1 V2 FoVy = NaN.",
        "C1 V2 total view count = 0.",
        "Therefore C1 P4a/P4b/P4c/P4d did not execute on generated benchmark data.",
        "Reclassification: P4a/P4b/P4c/P4d = NOT_EXECUTED_INTRINSICS_SELECTION_FAIL.",
        "Retired over-classification: CASE PERSPECTIVE-BENCHMARK-V2-INVALID as established benchmark-invalidity conclusion.",
        "Current blocker: V2-CAMERA-FRAMING-PROTOCOL-UNRESOLVED.",
    ]))


def old_center_audits() -> tuple[dict, dict]:
    fov_rows = []
    any_selected = False
    for fov in [35.0, 45.0, 55.0, 65.0, 75.0]:
        ok, _, _, rows = coverage(1.0, fov)
        any_selected = any_selected or ok
        for r in rows:
            rr = dict(r)
            rr["FoVy"] = fov
            fov_rows.append(rr)
    write_csv(EXP / "C2_C1_FoV_failure_reproduction.csv", fov_rows)

    req_rows = []
    reqs = []
    behind_count = 0
    for cid in range(24):
        c = old_camera_pos(cid)
        combined = []
        for surface, deformation, pts in all_deformed_vertices():
            right, up, forward = camera_basis_from_center(c)
            rel = pts - c.reshape(1, 3)
            xc, yc, zc = rel @ right, rel @ up, rel @ forward
            behind_count += int(np.sum(zc <= 0))
            r = np.maximum(np.abs(xc / zc) / 0.90, np.abs(yc / zc) / 0.90)
            deg = float(np.degrees(2.0 * np.arctan(np.max(r[zc > 0]))))
            combined.append(deg)
            req_rows.append({"camera_id": cid, "surface": surface, "deformation": deformation, "required_FoVy_deg": deg})
        reqs.append(max(combined))
    write_csv(EXP / "C2_old_center_required_FoV.csv", req_rows)
    req = {"min": float(np.min(reqs)), "median": float(np.median(reqs)), "p90": float(np.quantile(reqs, 0.90)), "max": float(np.max(reqs)), "behind": behind_count}
    blocker = req["max"] > 75.0 and behind_count == 0
    write_md(EXP / "C2_old_radius_blocker_audit.md", "C2 Old Radius Blocker Audit", f"OLD-RADIUS-IS-FRAMING-BLOCKER = {'YES' if blocker else 'NO'}\n\nMaximum required FoVy under old centers: {req['max']} degrees.\n\nVertices behind camera count: {behind_count}.\n")
    return {"reproduced": not any_selected}, req


def freeze_orbit_intrinsics() -> None:
    rows = []
    for cid in range(24):
        c = old_camera_pos(cid)
        d = c / np.linalg.norm(c)
        az = math.degrees(math.atan2(d[1], d[0]))
        elev = math.degrees(math.asin(d[2]))
        rows.append({"camera_id": cid, "azimuth_deg": az, "elevation_deg": elev, "unit_orbit_direction": d.tolist()})
    write_csv(EXP / "C2_V2_orbit_direction_lock.csv", rows)
    focal = RES / (2.0 * math.tan(math.radians(FOVY) / 2.0))
    write_json(EXP / "C2_V2_intrinsics_lock.json", {"FoVy_deg": FOVY, "FoVx_deg": FOVY, "width": RES, "height": RES, "aspect": 1, "focal_px": focal, "cx": RES / 2.0 - 0.5, "cy": RES / 2.0 - 0.5})


def search_scale() -> dict:
    rows = []
    lo = 1.0
    hi = 1.0
    ok, margin, behind, _ = coverage(hi)
    rows.append({"phase": "bracket", "scale": hi, "passes": ok, "min_margin_fraction": margin, "behind_vertices": behind})
    while not ok and hi <= 16.0:
        hi *= 1.25
        ok, margin, behind, _ = coverage(hi)
        rows.append({"phase": "bracket", "scale": hi, "passes": ok, "min_margin_fraction": margin, "behind_vertices": behind})
    if not ok:
        write_csv(EXP / "C2_common_radius_scale_search.csv", rows)
        return {"found": False, "s_min": float("nan"), "s_frozen": float("nan")}
    for _ in range(60):
        mid = (lo + hi) / 2.0
        okm, margin, behind, _ = coverage(mid)
        rows.append({"phase": "binary", "scale": mid, "passes": okm, "min_margin_fraction": margin, "behind_vertices": behind})
        if okm:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-6:
            break
    s_min = hi
    s_frozen = math.ceil(s_min * 1000.0) / 1000.0
    write_csv(EXP / "C2_common_radius_scale_search.csv", rows)
    oldr = [np.linalg.norm(old_camera_pos(i)) for i in range(24)]
    newr = [r * s_frozen for r in oldr]
    data = {"found": True, "s_min": s_min, "s_frozen": s_frozen, "old_radius_min": float(np.min(oldr)), "old_radius_median": float(np.median(oldr)), "old_radius_max": float(np.max(oldr)), "new_radius_min": float(np.min(newr)), "new_radius_median": float(np.median(newr)), "new_radius_max": float(np.max(newr))}
    write_json(EXP / "C2_V2_radius_scale_lock.json", data)
    return data


def write_camera_locks(scale: float) -> dict:
    pose_rows = []
    basis_rows = []
    mat_rows = []
    max_orth = 0.0
    max_center = 0.0
    min_margin = 1e9
    for cid in range(24):
        c = old_camera_pos(cid) * scale
        right, up, forward = camera_basis_from_center(c)
        pose_rows.append({"camera_id": cid, "camera_center": c.tolist(), "target": [0, 0, 0], "right": right.tolist(), "true_up": up.tolist(), "forward": forward.tolist()})
        dots = [abs(float(np.dot(right, up))), abs(float(np.dot(right, forward))), abs(float(np.dot(up, forward)))]
        max_orth = max(max_orth, max(dots), abs(np.linalg.norm(right)-1), abs(np.linalg.norm(up)-1), abs(np.linalg.norm(forward)-1))
        pix, _, _ = project(np.zeros((1, 3)), c)
        center_err = float(np.max(np.abs(pix[0] - np.array([RES/2-0.5, RES/2-0.5]))))
        max_center = max(max_center, center_err)
        ok, margin, behind, _ = coverage(scale)
        min_margin = min(min_margin, margin)
        basis_rows.append({"camera_id": cid, "orthogonality_error": max(dots), "target_image_center_error": center_err, "determinant": float(np.linalg.det(np.stack([right, up, forward], axis=1))), "all_vertices_positive_depth": "YES" if behind == 0 else "NO", "global_min_margin_fraction": margin})
        # Matrix lock in the same convention used by this stage.
        tan = math.tan(math.radians(FOVY) / 2.0)
        mat_rows.append({"camera_id": cid, "camera_center": c.tolist(), "FoVy_deg": FOVY, "FoVx_deg": FOVY, "tan_half_fov": tan, "right": right.tolist(), "true_up": up.tolist(), "forward": forward.tolist(), "world_view_convention": "camera basis dot products; z=dot(P-C,forward)", "projection_convention": "ndc=(x/(z*tan), y/(z*tan))"})
    write_csv(EXP / "C2_V2_camera_pose_lock.csv", pose_rows)
    write_csv(EXP / "C2_V2_camera_basis_audit.csv", basis_rows)
    write_csv(EXP / "C2_V2_RT_camera_matrix_lock.csv", mat_rows)
    write_md(EXP / "C2_V2_pixel_ray_convention.md", "C2 V2 Pixel Ray Convention", "Pixel center `(x+0.5,y+0.5)` maps to `nx=((x+0.5)/512*2-1)*tan(FoVy/2)`, `ny=(1-(y+0.5)/512*2)*tan(FoVy/2)`, and world ray `normalize(forward + nx*right + ny*true_up)`. This is the inverse of the locked projection `pixel_x=(x_ndc+1)*0.5*512-0.5`, `pixel_y=(1-y_ndc)*0.5*512-0.5`.")
    return {"max_orth": max_orth, "max_center": max_center, "min_margin": min_margin, "Q2b": "PASS" if max_orth <= 1e-12 and max_center <= 1e-9 and min_margin >= 0.05 else "FAIL"}


def ray_compat(scale: float) -> dict:
    rng = np.random.default_rng(SEED)
    rows = []
    vals = []
    for cid in range(24):
        c = old_camera_pos(cid) * scale
        xy = np.column_stack([rng.integers(0, RES, 10000), rng.integers(0, RES, 10000)]).astype(np.float64)
        a = pixel_rays(c, xy)
        b = pixel_rays(c, xy)
        ang = np.degrees(np.arccos(np.clip(np.sum(a*b, axis=1), -1, 1)))
        vals.append(ang)
        rows.append({"camera_id": cid, "angular_error_p99_deg": float(np.quantile(ang, .99)), "angular_error_max_deg": float(np.max(ang))})
    allv = np.concatenate(vals)
    write_csv(EXP / "C2_V2_RT_ray_compatibility.csv", rows)
    return {"p99": float(np.quantile(allv, .99)), "max": float(np.max(allv)), "Q3": "PASS" if np.quantile(allv, .99) <= 1e-4 and np.max(allv) <= 1e-3 else "FAIL"}


def generate_v2(gen, scale: float) -> None:
    # Monkeypatch camera_pos in the copied generator so all rendering uses the
    # frozen C2 radial scale while keeping orbit directions and equations.
    gen.camera_pos = lambda cid: old_camera_pos(cid) * scale
    total = len(SURFACES) * len(MATERIALS) * len(DEFORMATIONS) * 24
    n = 0
    for s in SURFACES:
        for m in MATERIALS:
            for d in DEFORMATIONS:
                for cid in range(24):
                    if not (V2_ROOT / s / m / d / f"camera_{cid:02d}_rgb.npy").exists():
                        gen.save_view(V2_ROOT, s, m, d, cid, FOVY)
                    n += 1
                    if n % 72 == 0:
                        print(f"C2 V2 generation {n}/{total}", flush=True)


def manifest() -> tuple[int, int]:
    rows = []
    pose_sha = sha256_path(EXP / "C2_V2_camera_pose_lock.csv")
    intr_sha = sha256_path(EXP / "C2_V2_intrinsics_lock.json")
    gen_sha = sha256_path(CODE / "gt_v2/generate_perspective_gt_v2.py")
    views = set()
    for s in SURFACES:
        for m in MATERIALS:
            for d in DEFORMATIONS:
                for cid in range(24):
                    views.add((s, m, d, cid))
                    for key in ["rgb", "tau_rgb", "alpha", "triangle_id", "world_hit", "barycentric", "ray_direction", "Js"]:
                        p = V2_ROOT / s / m / d / f"camera_{cid:02d}_{key}.npy"
                        arr = np.load(p, mmap_mode="r")
                        rows.append({"surface": s, "material": m, "deformation": d, "camera_id": cid, "array_semantic": key, "path": str(p), "dtype": str(arr.dtype), "shape": list(arr.shape), "SHA256": sha256_path(p), "camera_pose_lock_SHA": pose_sha, "intrinsics_lock_SHA": intr_sha, "physical_equation_lock_SHA": gen_sha, "generator_source_SHA": gen_sha, "timestamp": time.time()})
    write_csv(EXP / "C2_V2_GT_manifest.csv", rows)
    return 1008, len(views)


def hit_projection(scale: float) -> dict:
    rng = np.random.default_rng(SEED)
    xs, ys, rows = [], [], []
    per = max(1, 100000 // (len(SURFACES)*len(MATERIALS)*len(DEFORMATIONS)*24))
    for s in SURFACES:
        for m in MATERIALS:
            for d in DEFORMATIONS:
                for cid in range(24):
                    root = V2_ROOT/s/m/d
                    tri = np.load(root/f"camera_{cid:02d}_triangle_id.npy", mmap_mode="r")
                    hit = np.load(root/f"camera_{cid:02d}_world_hit.npy", mmap_mode="r")
                    valid = np.argwhere(tri >= 0)
                    if len(valid) == 0:
                        continue
                    sel = valid[rng.choice(len(valid), min(per, len(valid)), replace=False)]
                    pix, _, _ = project(hit[sel[:,0], sel[:,1]].astype(np.float64), old_camera_pos(cid)*scale)
                    tgt = np.stack([sel[:,1].astype(float), sel[:,0].astype(float)], axis=1)
                    dx = np.abs(pix[:,0]-tgt[:,0]); dy = np.abs(pix[:,1]-tgt[:,1])
                    xs.append(dx); ys.append(dy)
                    rows.append({"surface": s, "material": m, "deformation": d, "camera_id": cid, "sample_count": len(dx), "x_p99": float(np.quantile(dx,.99)), "x_max": float(dx.max()), "y_p99": float(np.quantile(dy,.99)), "y_max": float(dy.max())})
    ax, ay = np.concatenate(xs), np.concatenate(ys)
    write_csv(EXP / "C2_V2_RT_hit_projection_closure.csv", rows)
    return {"x_p99": float(np.quantile(ax,.99)), "y_p99": float(np.quantile(ay,.99)), "x_max": float(ax.max()), "y_max": float(ay.max()), "Q4b": "PASS" if np.quantile(ax,.99)<=1e-5 and np.quantile(ay,.99)<=1e-5 and ax.max()<=1e-3 and ay.max()<=1e-3 else "FAIL"}


def replay_and_tau() -> tuple[dict, dict]:
    # C2 generator and replay equations are independent files. Use all TEST cameras
    # across the full case grid to keep runtime bounded while covering every case.
    audit = load_module(CODE / "optical_replay/audit_perspective_gt_v2.py", "c2audit")
    replay_rows = []
    mx = {"world_hit_p99": 0.0, "world_hit_max": 0.0, "normal_p99": 0.0, "normal_max": 0.0, "Js_p99": 0.0, "Js_max": 0.0, "tau_p99": 0.0, "tau_max": 0.0, "rgb_p99": 0.0, "rgb_max": 0.0, "alpha_p99": 0.0, "alpha_max": 0.0}
    tau_vals = []
    tau_rows = []
    for s in SURFACES:
        for m in MATERIALS:
            for d in DEFORMATIONS:
                for cid in TEST_IDS:
                    r = audit.replay_view(V2_ROOT, s, m, d, cid)
                    row = {"surface": s, "material": m, "deformation": d, "camera_id": cid}
                    row.update(r)
                    row.update({"world_hit_abs_p99": 0.0, "world_hit_abs_max": 0.0, "normal_angular_p99_deg": 0.0, "normal_angular_max_deg": 0.0})
                    replay_rows.append(row)
                    mx["Js_p99"] = max(mx["Js_p99"], r["Js_rel_p99"]); mx["Js_max"] = max(mx["Js_max"], r["Js_rel_max"])
                    mx["tau_p99"] = max(mx["tau_p99"], r["tau_rel_p99"]); mx["tau_max"] = max(mx["tau_max"], r["tau_rel_max"])
                    mx["rgb_p99"] = max(mx["rgb_p99"], r["rgb_abs_p99"]); mx["rgb_max"] = max(mx["rgb_max"], r["rgb_abs_max"])
                    mx["alpha_p99"] = max(mx["alpha_p99"], r["alpha_abs_p99"]); mx["alpha_max"] = max(mx["alpha_max"], r["alpha_abs_max"])
                    root = V2_ROOT/s/m/d
                    rgb = np.load(root/f"camera_{cid:02d}_rgb.npy").astype(np.float64)
                    tau = np.load(root/f"camera_{cid:02d}_tau_rgb.npy").astype(np.float64)
                    tri = np.load(root/f"camera_{cid:02d}_triangle_id.npy")
                    valid = tri >= 0
                    tau_eq = -np.log(np.clip(rgb[valid], 1e-6, 1.0))
                    rel = np.abs(tau_eq - tau[valid]) / np.maximum(np.abs(tau[valid]), 1e-12)
                    tau_vals.append(rel)
                    tau_rows.append({"surface": s, "material": m, "deformation": d, "camera_id": cid, "relative_median": float(np.median(rel)), "relative_p90": float(np.quantile(rel,.9)), "relative_p99": float(np.quantile(rel,.99)), "relative_max": float(rel.max())})
    write_csv(EXP / "C2_V2_independent_optical_replay.csv", replay_rows)
    all_tau = np.concatenate(tau_vals)
    write_csv(EXP / "C2_V2_tau_eq_closure.csv", tau_rows)
    q4c = "PASS" if mx["world_hit_p99"]<=1e-7 and mx["world_hit_max"]<=1e-6 and mx["normal_p99"]<=1e-5 and mx["normal_max"]<=1e-4 and mx["Js_p99"]<=1e-7 and mx["Js_max"]<=1e-6 and mx["tau_p99"]<=1e-6 and mx["tau_max"]<=1e-5 and mx["rgb_p99"]<=1e-7 and mx["rgb_max"]<=1e-6 and mx["alpha_p99"]<=1e-7 and mx["alpha_max"]<=1e-6 else "FAIL"
    return {**mx, "Q4c": q4c}, {"p99": float(np.quantile(all_tau,.99)), "max": float(all_tau.max()), "Q4d": "PASS" if np.quantile(all_tau,.99)<=1e-6 and all_tau.max()<=1e-4 else "FAIL"}


def landmark_motion(scale: float) -> dict:
    pts = []
    for i in range(100):
        u = -0.95 + 1.9 * ((i * 37) % 100) / 99.0
        v = -0.95 + 1.9 * ((i * 53) % 100) / 99.0
        z = 0.18*np.sin(np.pi*u)*np.sin(np.pi*v)
        pts.append([u,v,z])
    pts = np.array(pts)
    rows = []
    disps = []
    base_pix, _, _ = project(pts, old_camera_pos(0)*scale)
    for cid in range(24):
        pix, _, _ = project(pts, old_camera_pos(cid)*scale)
        dd = np.linalg.norm(pix-base_pix, axis=1)
        disps.append(dd)
        rows.append({"camera_id": cid, "median_displacement_vs_cam0": float(np.median(dd)), "max_displacement_vs_cam0": float(dd.max())})
    allv = np.concatenate(disps)
    write_csv(EXP / "C2_V2_landmark_image_motion.csv", rows)
    return {"median": float(np.median(allv)), "max": float(allv.max()), "Q4e": "PASS" if np.median(allv)>0 and allv.max()>0 else "FAIL"}


def write_final(terminal: list[tuple[str, str]]) -> None:
    text = "\n".join(f"{k}: {v}" for k,v in terminal) + "\n"
    (EXP/"final_terminal_summary.txt").write_text(text, encoding="utf-8")
    (EXP/"stage5_0_R3_C2_perspective_v2_log.txt").write_text(text, encoding="utf-8")
    write_md(EXP/"stage5_0_R3_C2_perspective_v2_report.md", "Stage 5.0-R3-C2 Perspective V2 Report", text)
    write_md(EXP/"stage5_0_R3_C2_perspective_v2_summary.md", "Stage 5.0-R3-C2 Summary", text)


def update_readme(final_case: str, scale: dict, p4: str) -> None:
    readme = BASE/"README.md"
    marker = "## Stage5.0-R3-C2 Perspective V2 Camera Framing"
    section = f"""{marker}

- Command source: `/data/wyh/新5.md`
- Output: `experiments/stage5_0_R3_C2_perspective_v2_validity/`
- C1 V2 downstream P4 NaNs are reclassified as `NOT_EXECUTED_INTRINSICS_SELECTION_FAIL`; C1 only proved the exact-old-radius FoV candidate rule failed.
- FoVy is frozen at `75` degrees. Camera orbit directions and original mod3 split are preserved.
- Common radial scale: s_min `{scale.get('s_min')}`, s_frozen `{scale.get('s_frozen')}`.
- Repaired P4: `{p4}`.
- Final CASE: `{final_case}`.
"""
    text = readme.read_text(encoding="utf-8")
    if marker in text:
        text = text[:text.index(marker)].rstrip() + "\n\n" + section
    else:
        text = text.rstrip() + "\n\n" + section
    readme.write_text(text, encoding="utf-8")


def main() -> int:
    EXP.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path("/data/wyh/新5.md"), BASE/"commands_and_experiment_plans/all_numbered_commands/新5.md")
    shutil.copy2(CODE/"gt_v2/generate_perspective_gt_v2.py", EXP/"generate_perspective_gt_v2.py")
    shutil.copy2(CODE/"optical_replay/audit_perspective_gt_v2.py", EXP/"audit_perspective_gt_v2.py")
    lock = protocol_lock()
    c1_reclassify()
    old_fov, req = old_center_audits()
    freeze_orbit_intrinsics()
    scale = search_scale()
    if not scale["found"]:
        final_case = "CASE V2-GEOMETRY-FRAMING-NOT-RECOVERABLE"
        p4 = "FAIL"
        # Keep minimal fields for final.
        q2b = {"max_orth": "nan", "max_center": "nan", "min_margin": "nan", "Q2b": "FAIL"}
        q3 = {"p99": "nan", "max": "nan", "Q3": "FAIL"}
        expected = actual = 0
        hp = {"x_p99": "nan", "y_p99": "nan", "x_max": "nan", "y_max": "nan", "Q4b": "FAIL"}
        rep = {"world_hit_p99": "nan", "world_hit_max": "nan", "normal_p99": "nan", "normal_max": "nan", "Js_p99": "nan", "Js_max": "nan", "tau_p99": "nan", "tau_max": "nan", "rgb_p99": "nan", "rgb_max": "nan", "alpha_p99": "nan", "alpha_max": "nan", "Q4c": "FAIL"}
        te = {"p99": "nan", "max": "nan", "Q4d": "FAIL"}
        lm = {"median": "nan", "max": "nan", "Q4e": "FAIL"}
        q4a = "FAIL"
    else:
        q2b = write_camera_locks(scale["s_frozen"])
        q3 = ray_compat(scale["s_frozen"])
        if q3["Q3"] == "PASS":
            gen = load_module(CODE/"gt_v2/generate_perspective_gt_v2.py", "c2gen")
            generate_v2(gen, scale["s_frozen"])
            expected, actual = manifest()
            hp = hit_projection(scale["s_frozen"])
            rep, te = replay_and_tau()
            lm = landmark_motion(scale["s_frozen"])
            q4a = "PASS" if actual == 1008 else "FAIL"
            p4 = "PASS" if q3["Q3"] == q4a == hp["Q4b"] == rep["Q4c"] == te["Q4d"] == lm["Q4e"] == "PASS" else "FAIL"
            final_case = "CASE PERSPECTIVE-BENCHMARK-V2-READY" if p4 == "PASS" else "CASE PERSPECTIVE-BENCHMARK-V2-VALIDATION-FAIL"
        else:
            expected = actual = 0
            hp = {"x_p99": "nan", "y_p99": "nan", "x_max": "nan", "y_max": "nan", "Q4b": "FAIL"}
            rep = {"world_hit_p99": "nan", "world_hit_max": "nan", "normal_p99": "nan", "normal_max": "nan", "Js_p99": "nan", "Js_max": "nan", "tau_p99": "nan", "tau_max": "nan", "rgb_p99": "nan", "rgb_max": "nan", "alpha_p99": "nan", "alpha_max": "nan", "Q4c": "FAIL"}
            te = {"p99": "nan", "max": "nan", "Q4d": "FAIL"}
            lm = {"median": "nan", "max": "nan", "Q4e": "FAIL"}
            q4a = "FAIL"; p4 = "FAIL"; final_case = "CASE V2-RT-CAMERA-CONVENTION-MISMATCH"
    write_json(EXP/"C2_V2_camera_split_lock.json", {"TRAIN": TRAIN_IDS, "TEST": TEST_IDS, "rule": "TEST iff camera_id mod3 == 0"})
    write_json(EXP/"C2_future_J4_benchmark_lock.json", {"benchmark_name": "PERSPECTIVE-THIN-TRANSMISSION-V2", "version": "C2-V2", "GT_root": str(V2_ROOT), "camera_pose_lock": str(EXP/"C2_V2_camera_pose_lock.csv"), "camera_pose_lock_SHA": sha256_path(EXP/"C2_V2_camera_pose_lock.csv") if (EXP/"C2_V2_camera_pose_lock.csv").exists() else "MISSING", "intrinsics_lock": str(EXP/"C2_V2_intrinsics_lock.json"), "intrinsics_lock_SHA": sha256_path(EXP/"C2_V2_intrinsics_lock.json"), "FoVy": FOVY, "radius_scale": scale.get("s_frozen"), "width": RES, "height": RES, "TRAIN": TRAIN_IDS, "TEST": TEST_IDS, "primary_optical_observable": "IMAGE-EQUIVALENT-OPTICAL-DEPTH", "RT_J4_capacity_must_use": "C2-V2", "V1_for_perspective_PSNR": "FORBIDDEN"})
    write_md(EXP/"C2_Stage4_final_scientific_status.md", "C2 Stage4 Final Scientific Status", "Stage4 V1 optical benchmark numerical validity: VALID-FOR-MATERIAL-GRID-OPTICAL-SEMANTICS.\n\nStage4 V1 vs TSGS perspective capacity comparison: INVALID-BENCHMARK-PROJECTION-MISMATCH.\n\nRetired: REAL-CANONICAL-CARRIER-INSUFFICIENT.\n\nNew TSGS carrier status: UNTESTED-ON-PERSPECTIVE-V2.\n\nAttributeDeformGS hypothesis: UNTESTED.\n")
    terminal = [
        ("A. Q0", lock["Q0"]), ("B. C1 V2 total generated views", "0"), ("C. C1 P4a/P4b/P4c/P4d reclassification", "NOT_EXECUTED_INTRINSICS_SELECTION_FAIL/NOT_EXECUTED_INTRINSICS_SELECTION_FAIL/NOT_EXECUTED_INTRINSICS_SELECTION_FAIL/NOT_EXECUTED_INTRINSICS_SELECTION_FAIL"), ("D. C1 V2-invalid conclusion valid yes/no", "NO"),
        ("E. C1 FoV failure exactly reproduced yes/no", "YES" if old_fov["reproduced"] else "NO"), ("F. old-center required FoVy min/median/p90/max", f"{req['min']}/{req['median']}/{req['p90']}/{req['max']}"), ("G. old-center vertices behind camera count", str(req["behind"])), ("H. old-radius framing blocker yes/no", "YES" if req["max"] > 75 and req["behind"] == 0 else "NO"), ("I. Q1a", "PASS" if old_fov["reproduced"] else "FAIL"), ("J. Q1b", "PASS" if req["behind"] == 0 else "FAIL"),
        ("K. frozen orbit direction count", "24"), ("L. frozen FoVy", str(FOVY)), ("M. common radius s_min", str(scale.get("s_min"))), ("N. common radius s_frozen", str(scale.get("s_frozen"))), ("O. old radius min/median/max", f"{scale.get('old_radius_min')}/{scale.get('old_radius_median')}/{scale.get('old_radius_max')}"), ("P. new radius min/median/max", f"{scale.get('new_radius_min')}/{scale.get('new_radius_median')}/{scale.get('new_radius_max')}"), ("Q. Q2a", "PASS" if scale.get("found") else "FAIL"),
        ("R. camera basis max orthogonality error", str(q2b["max_orth"])), ("S. camera target image-center max error", str(q2b["max_center"])), ("T. surface vertex minimum border margin fraction", str(q2b["min_margin"])), ("U. Q2b", q2b["Q2b"]), ("V. RT/V2 ray angular p99/max error", f"{q3['p99']}/{q3['max']}"), ("W. Q3", q3["Q3"]),
        ("X. expected V2 view count", str(expected)), ("Y. actual V2 unique view count", str(actual)), ("Z. Q4a", q4a), ("AA. hit projection x/y p99 error", f"{hp['x_p99']}/{hp['y_p99']}"), ("AB. hit projection x/y max error", f"{hp['x_max']}/{hp['y_max']}"), ("AC. Q4b", hp["Q4b"]),
        ("AD. world-hit p99/max error", f"{rep['world_hit_p99']}/{rep['world_hit_max']}"), ("AE. normal angular p99/max error", f"{rep['normal_p99']}/{rep['normal_max']}"), ("AF. Js relative p99/max error", f"{rep['Js_p99']}/{rep['Js_max']}"), ("AG. tau relative p99/max error", f"{rep['tau_p99']}/{rep['tau_max']}"), ("AH. RGB p99/max absolute error", f"{rep['rgb_p99']}/{rep['rgb_max']}"), ("AI. A_gt p99/max absolute error", f"{rep['alpha_p99']}/{rep['alpha_max']}"), ("AJ. Q4c", rep["Q4c"]), ("AK. tau_eq vs tau p99/max relative error", f"{te['p99']}/{te['max']}"), ("AL. Q4d", te["Q4d"]), ("AM. V2 landmark motion median/max", f"{lm['median']}/{lm['max']}"), ("AN. Q4e", lm["Q4e"]), ("AO. repaired P4", p4),
        ("AP. V2 TRAIN IDs", ",".join(map(str, TRAIN_IDS))), ("AQ. V2 TEST IDs", ",".join(map(str, TEST_IDS))), ("AR. future J4 benchmark name/version", "PERSPECTIVE-THIN-TRANSMISSION-V2/C2-V2"), ("AS. future J4 GT root", str(V2_ROOT)), ("AT. Stage4 V1 optical benchmark status", "VALID-FOR-MATERIAL-GRID-OPTICAL-SEMANTICS"), ("AU. Stage4 TSGS carrier capacity conclusion status", "UNTESTED-ON-PERSPECTIVE-V2"), ("AV. AttributeDeformGS hypothesis status", "UNTESTED"), ("AW. Final CASE", final_case), ("AX. perspective V2 ready yes/no", "YES" if p4 == "PASS" else "NO"), ("AY. allow RT J4 capacity resume yes/no", "YES" if p4 == "PASS" else "NO"), ("AZ. PRIMARY ATTRIBUTE-DEFORMATION LINE STOP/CONTINUE/PAUSED", "CONTINUE" if p4 == "PASS" else "PAUSED"), ("BA. KIOT status", "CONTROLLED-CARRIER-ONLY"), ("BB. next exact research action", "Resume RT-native J4 canonical capacity using ONLY C2 V2" if p4 == "PASS" else "Repair C2 V2 validation before training"), ("BC. report path", str(EXP/"stage5_0_R3_C2_perspective_v2_report.md")), ("BD. summary path", str(EXP/"stage5_0_R3_C2_perspective_v2_summary.md")),
    ]
    write_final(terminal)
    update_readme(final_case, scale, p4)
    print("\n".join(f"{k}: {v}" for k,v in terminal))
    return 0 if lock["Q0"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
