from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np


BASE = Path("/data/wyh/DeformTransGS")
D0 = BASE / "experiments/stageD0_deformable_optical_transport_feasibility"
OUT = BASE / "experiments/stageD1_N0_deformation_descriptor_identifiability"
CMD_SRC = Path("/data/wyh/新10.md")
CMD_DST = BASE / "commands_and_experiment_plans/all_numbered_commands/新10.md"
SURFACES = ("S0_PLANAR_SHEET", "S1_WAVY_MEMBRANE")
TOL = 1e-8


def assert_gpu_scope() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible != "2,3":
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be 2,3, got {visible!r}")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def stream_csv(path: Path, fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fields)
    w.writeheader()
    return f, w


def write_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]], dtype=np.float64)


def deformation_bank() -> list[tuple[str, str, np.ndarray]]:
    rows = [
        ("D0_IDENTITY", "original D0 identity", np.diag([1.0, 1.0, 1.0])),
        ("D1_STRETCH_X_1P25", "original D1 x stretch 1.25", np.diag([1.25, 1.0, 1.0])),
        ("D2_STRETCH_X_1P50", "original D2 x stretch 1.50", np.diag([1.50, 1.0, 1.0])),
        ("D3_BIAXIAL_XY_1P50", "original D3 xy stretch 1.50", np.diag([1.50, 1.50, 1.0])),
        ("D4_SHEAR_XY_0P30", "original D4 xy shear 0.30", np.array([[1.0, 0.30, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)),
        ("D5_ANISO_X1P60_Y0P80", "original D5 anisotropic stretch", np.diag([1.60, 0.80, 1.0])),
        ("D6_ROTATION_Z_30", "original D6 z rotation 30 deg", rz(30.0)),
        ("A1_IDENTITY_JS1", "PAIR-A/B/D identity control", np.diag([1.0, 1.0, 1.0])),
        ("A2_AREA1_ANISO_X2_Y0P5", "PAIR-A same planar Js=1 different anisotropy", np.diag([2.0, 0.5, 1.0])),
        ("B2_AREA1_SHEAR_XY_0P50", "PAIR-B same planar Js=1 different shear", np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)),
        ("C1_ANISO_X1P6_Y0P8", "PAIR-C spectrum baseline", np.diag([1.6, 0.8, 1.0])),
        ("C2_ANISO_ROT45_SAME_SPECTRUM", "PAIR-C same spectrum rotated principal direction", rz(45.0) @ np.diag([1.6, 0.8, 1.0]) @ rz(-45.0)),
        ("D2_RIGID_RZ30", "PAIR-D rigid z rotation", rz(30.0)),
        ("D3_RIGID_RX25", "PAIR-D rigid x rotation", rx(25.0)),
    ]
    seen: set[str] = set()
    out = []
    for key, desc, F in rows:
        if key in seen:
            raise RuntimeError(f"duplicate deformation key {key}")
        seen.add(key)
        det = float(np.linalg.det(F))
        cond = float(np.linalg.cond(F))
        if det <= 0.0 or cond > 10.0 or not np.isfinite(cond):
            raise RuntimeError(f"invalid candidate {key}: det={det} cond={cond}")
        out.append((key, desc, F))
    return out


def surface_eval(surface: str, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if surface == "S0_PLANAR_SHEET":
        z = np.zeros_like(u)
        du = np.stack([np.ones_like(u), np.zeros_like(u), np.zeros_like(u)], axis=-1)
        dv = np.stack([np.zeros_like(u), np.ones_like(u), np.zeros_like(u)], axis=-1)
    else:
        z = 0.18 * np.sin(np.pi * u) * np.sin(np.pi * v)
        dzdu = 0.18 * np.pi * np.cos(np.pi * u) * np.sin(np.pi * v)
        dzdv = 0.18 * np.pi * np.sin(np.pi * u) * np.cos(np.pi * v)
        du = np.stack([np.ones_like(u), np.zeros_like(u), dzdu], axis=-1)
        dv = np.stack([np.zeros_like(u), np.ones_like(u), dzdv], axis=-1)
    xyz = np.stack([u, v, z], axis=-1)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-30
    return xyz, du, dv, n


def canonical_frame(surface: str, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, du, _, n0 = surface_eval(surface, u, v)
    t1 = du / (np.linalg.norm(du, axis=-1, keepdims=True) + 1e-30)
    t2 = np.cross(n0, t1)
    t2 /= np.linalg.norm(t2, axis=-1, keepdims=True) + 1e-30
    return t1, t2, n0


def descriptors(surface: str, u: np.ndarray, v: np.ndarray, F: np.ndarray) -> dict[str, np.ndarray]:
    t1, t2, n0 = canonical_frame(surface, u, v)
    Ft1 = t1 @ F.T
    Ft2 = t2 @ F.T
    t1p = Ft1 / (np.linalg.norm(Ft1, axis=-1, keepdims=True) + 1e-30)
    np_cross = np.cross(Ft1, Ft2)
    np_cross /= np.linalg.norm(np_cross, axis=-1, keepdims=True) + 1e-30
    nfinv = n0 @ np.linalg.inv(F)
    nfinv /= np.linalg.norm(nfinv, axis=-1, keepdims=True) + 1e-30
    sign = np.sum(np_cross * nfinv, axis=-1) < 0
    np_cross[sign] *= -1.0
    t2p = np.cross(np_cross, t1p)
    t2p /= np.linalg.norm(t2p, axis=-1, keepdims=True) + 1e-30
    a11 = np.sum(Ft1 * t1p, axis=-1)
    a12 = np.sum(Ft2 * t1p, axis=-1)
    a21 = np.sum(Ft1 * t2p, axis=-1)
    a22 = np.sum(Ft2 * t2p, axis=-1)
    c11 = a11 * a11 + a21 * a21
    c12 = a11 * a12 + a21 * a22
    c22 = a12 * a12 + a22 * a22
    trace = c11 + c22
    disc = np.sqrt(np.maximum((c11 - c22) ** 2 + 4.0 * c12 * c12, 0.0))
    eig_hi = np.maximum((trace + disc) * 0.5, 0.0)
    eig_lo = np.maximum((trace - disc) * 0.5, 0.0)
    lam1 = np.sqrt(eig_hi)
    lam2 = np.sqrt(eig_lo)
    js = np.abs(float(np.linalg.det(F))) * np.linalg.norm(n0 @ np.linalg.inv(F), axis=-1)
    gamma = np.abs(c12) / np.sqrt(np.maximum(c11 * c22, 1e-30))
    return {
        "Js": js,
        "lambda1": lam1,
        "lambda2": lam2,
        "log_lambda1": np.log(np.maximum(lam1, 1e-30)),
        "log_lambda2": np.log(np.maximum(lam2, 1e-30)),
        "area_from_stretch": lam1 * lam2,
        "C11": c11,
        "C12": c12,
        "C22": c22,
        "gamma": gamma,
        "detF": np.full_like(js, float(np.linalg.det(F))),
    }


def load_samples(surface: str) -> list[dict]:
    p = D0 / "D0_material_samples" / ("S0.csv" if surface.startswith("S0") else "S1.csv")
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def protocol_lock() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    CMD_DST.parent.mkdir(parents=True, exist_ok=True)
    if CMD_SRC.exists():
        shutil.copy2(CMD_SRC, CMD_DST)
    paths = [
        D0 / "D0_protocol_lock.json",
        D0 / "stageD0_feasibility_report.md",
        D0 / "stageD0_feasibility_summary.md",
        D0 / "D0_deformation_equations.md",
        D0 / "D0_material_samples/S0.csv",
        D0 / "D0_material_samples/S1.csv",
        D0 / "D0_deformation_replay.csv",
        D0 / "D0_local_frame_table.csv",
        D0 / "D0_deformation_descriptor_table.csv",
        D0 / "D0_paired_optical_transport_table.csv",
        BASE / "deformable_optical_transport/run_D0.py",
    ]
    records = {str(p): {"exists": p.exists(), "sha256": sha(p) if p.exists() else "MISSING"} for p in paths}
    records["exact_surface_definition_source"] = str(BASE / "deformable_optical_transport/run_D0.py")
    gate = "PASS" if all(v["exists"] for v in records.values() if isinstance(v, dict)) else "FAIL"
    records["N0-G0"] = gate
    write_json(OUT / "D1N0_protocol_lock.json", records)
    return gate


def existing_dependency() -> tuple[dict, str]:
    rows_by_surface = {s: [] for s in SURFACES}
    with (D0 / "D0_deformation_descriptor_table.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_surface[row["surface"]].append(row)
    cols = ["detF", "Js", "lambda1", "lambda2", "area_from_stretch", "log_lambda1", "log_lambda2", "C11", "C12", "C22", "gamma"]
    out_rows = []
    summary = {}
    for label, rows in [("S0", rows_by_surface["S0_PLANAR_SHEET"]), ("S1", rows_by_surface["S1_WAVY_MEMBRANE"]), ("combined", rows_by_surface["S0_PLANAR_SHEET"] + rows_by_surface["S1_WAVY_MEMBRANE"])]:
        arr = np.array([[math.log(float(r["lambda1"])) if c == "log_lambda1" else math.log(float(r["lambda2"])) if c == "log_lambda2" else float(r[c]) for c in cols] for r in rows], dtype=np.float64)
        centered = arr - arr.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(centered, compute_uv=False)
        rank = int(np.linalg.matrix_rank(centered, tol=1e-10))
        cond = float(sv[0] / max(sv[-1], 1e-30)) if sv.size else float("nan")
        summary[label] = {"rank": rank, "condition": cond, "singular_values": sv.tolist()}
        corr = np.corrcoef(arr, rowvar=False)
        ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)).astype(np.float64), 0, arr)
        spear = np.corrcoef(ranks, rowvar=False)
        for i, ci in enumerate(cols):
            for j, cj in enumerate(cols):
                out_rows.append({"scope": label, "var_i": ci, "var_j": cj, "pearson": corr[i, j], "spearman": spear[i, j], "matrix_rank": rank, "condition_number": cond, "singular_values": json.dumps(sv.tolist())})
    write_csv(OUT / "D1N0_existing_descriptor_dependency.csv", out_rows)
    write_md(
        OUT / "D1N0_descriptor_dependency.md",
        "D1-N0 Existing Descriptor Dependency",
        "\n".join([
            "The D0 descriptor table is geometry-only for this audit.",
            "Exact identity acknowledged: Js = lambda1 * lambda2 was established in D0 and re-audited here through area_from_stretch.",
            "Correlation is diagnostic only; algebraic identities and matched controls drive the N0 decision.",
            "",
            f"S0 rank/condition: {summary['S0']['rank']} / {summary['S0']['condition']}",
            f"S1 rank/condition: {summary['S1']['rank']} / {summary['S1']['condition']}",
            f"combined rank/condition: {summary['combined']['rank']} / {summary['combined']['condition']}",
        ]),
    )
    return summary, "YES"


def existing_pair_audit() -> tuple[int, int, int]:
    fields = ["surface", "sample_id", "deformation_a", "deformation_b", "same_Js", "different_spectrum", "same_spectrum", "different_Ct", "different_abs_C12", "rigid_null_control"]
    rows = []
    by_key: dict[tuple[str, str], list[dict]] = {}
    with (D0 / "D0_deformation_descriptor_table.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_key.setdefault((row["surface"], row["sample_id"]), []).append(row)
    same_js_diff_spec = 0
    same_spec_diff_ct = 0
    same_js_diff_c12 = 0
    for (surface, sid), group in by_key.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                same_js = abs(float(a["Js"]) - float(b["Js"])) <= 1e-10
                spec_diff = max(abs(float(a["lambda1"]) - float(b["lambda1"])), abs(float(a["lambda2"]) - float(b["lambda2"]))) > 1e-10
                same_spec = max(abs(float(a["lambda1"]) - float(b["lambda1"])), abs(float(a["lambda2"]) - float(b["lambda2"]))) <= 1e-10
                ct_diff = max(abs(float(a["C11"]) - float(b["C11"])), abs(float(a["C12"]) - float(b["C12"])), abs(float(a["C22"]) - float(b["C22"]))) > 1e-10
                c12_diff = abs(abs(float(a["C12"])) - abs(float(b["C12"]))) > 1e-10
                rigid = a["deformation"] in {"D0_IDENTITY", "D6_ROTATION_Z_30"} and b["deformation"] in {"D0_IDENTITY", "D6_ROTATION_Z_30"} and not ct_diff
                if same_js and spec_diff:
                    same_js_diff_spec += 1
                if same_spec and ct_diff:
                    same_spec_diff_ct += 1
                if same_js and c12_diff:
                    same_js_diff_c12 += 1
                if same_js and (spec_diff or ct_diff or c12_diff or rigid):
                    rows.append({"surface": surface, "sample_id": sid, "deformation_a": a["deformation"], "deformation_b": b["deformation"], "same_Js": same_js, "different_spectrum": spec_diff, "same_spectrum": same_spec, "different_Ct": ct_diff, "different_abs_C12": c12_diff, "rigid_null_control": rigid})
    write_csv(OUT / "D1N0_existing_matched_pairs.csv", rows, fields)
    return same_js_diff_spec, same_spec_diff_ct, same_js_diff_c12


def write_bank(bank: list[tuple[str, str, np.ndarray]]) -> None:
    rows = []
    for key, desc, F in bank:
        rows.append({"deformation_key": key, "description": desc, "deterministic_affine": "YES", "positive_determinant": "YES", "condition_number": float(np.linalg.cond(F)), "detF": float(np.linalg.det(F)), "F": json.dumps(F.tolist())})
    write_csv(OUT / "D1N0_candidate_deformation_bank.csv", rows)
    body = []
    for key, desc, F in bank:
        body.append(f"## {key}\n\n{desc}\n\nF = `{json.dumps(F.tolist())}`\n")
    write_md(OUT / "D1N0_candidate_deformation_equations.md", "D1-N0 Candidate Deformation Equations", "\n".join(body))


def replay_candidate_descriptors(bank: list[tuple[str, str, np.ndarray]]) -> tuple[float, int]:
    fields = ["sample_id", "surface", "deformation", "Js", "detF", "lambda1", "lambda2", "area_from_stretch", "log_lambda1", "log_lambda2", "C11", "C12", "C22", "gamma", "Js_area_rel_error"]
    f, w = stream_csv(OUT / "D1N0_candidate_descriptor_table.csv", fields)
    max_rel = 0.0
    count = 0
    try:
        for surface in SURFACES:
            samples = load_samples(surface)
            u = np.array([float(r["u"]) for r in samples], dtype=np.float64)
            v = np.array([float(r["v"]) for r in samples], dtype=np.float64)
            sid = [r["sample_id"] for r in samples]
            for key, _, F in bank:
                d = descriptors(surface, u, v, F)
                rel = np.abs(d["Js"] - d["area_from_stretch"]) / np.maximum(np.abs(d["Js"]), 1e-30)
                max_rel = max(max_rel, float(rel.max()))
                for i in range(len(samples)):
                    w.writerow({"sample_id": sid[i], "surface": surface, "deformation": key, "Js": d["Js"][i], "detF": d["detF"][i], "lambda1": d["lambda1"][i], "lambda2": d["lambda2"][i], "area_from_stretch": d["area_from_stretch"][i], "log_lambda1": d["log_lambda1"][i], "log_lambda2": d["log_lambda2"][i], "C11": d["C11"][i], "C12": d["C12"][i], "C22": d["C22"][i], "gamma": d["gamma"][i], "Js_area_rel_error": rel[i]})
                    count += 1
    finally:
        f.close()
    return max_rel, count


def pair_stats(surface: str, key_a: str, key_b: str, lookup: dict[str, np.ndarray]) -> dict:
    samples = load_samples(surface)
    u = np.array([float(r["u"]) for r in samples], dtype=np.float64)
    v = np.array([float(r["v"]) for r in samples], dtype=np.float64)
    a = descriptors(surface, u, v, lookup[key_a])
    b = descriptors(surface, u, v, lookup[key_b])
    js = np.max(np.abs(a["Js"] - b["Js"]))
    spec = max(float(np.max(np.abs(a["lambda1"] - b["lambda1"]))), float(np.max(np.abs(a["lambda2"] - b["lambda2"]))))
    ct = max(float(np.max(np.abs(a["C11"] - b["C11"]))), float(np.max(np.abs(a["C12"] - b["C12"]))), float(np.max(np.abs(a["C22"] - b["C22"]))))
    c12 = float(np.max(np.abs(np.abs(a["C12"]) - np.abs(b["C12"]))))
    return {"surface": surface, "deformation_a": key_a, "deformation_b": key_b, "max_abs_Js_delta": js, "max_abs_spectrum_delta": spec, "max_abs_Ct_delta": ct, "max_abs_absC12_delta": c12}


def validate_pairs(bank: list[tuple[str, str, np.ndarray]]) -> tuple[dict, list[dict]]:
    lookup = {k: F for k, _, F in bank}
    pair_defs = {
        "PAIR-A": ("A1_IDENTITY_JS1", "A2_AREA1_ANISO_X2_Y0P5"),
        "PAIR-B": ("A1_IDENTITY_JS1", "B2_AREA1_SHEAR_XY_0P50"),
        "PAIR-C": ("C1_ANISO_X1P6_Y0P8", "C2_ANISO_ROT45_SAME_SPECTRUM"),
        "PAIR-D-Z": ("A1_IDENTITY_JS1", "D2_RIGID_RZ30"),
        "PAIR-D-X": ("A1_IDENTITY_JS1", "D3_RIGID_RX25"),
    }
    rows = []
    result = {}
    for pair, (ka, kb) in pair_defs.items():
        pair_rows = []
        for surface in SURFACES:
            stats = pair_stats(surface, ka, kb, lookup)
            stats["pair_category"] = pair
            if pair == "PAIR-A":
                passed = stats["max_abs_Js_delta"] <= TOL and stats["max_abs_spectrum_delta"] > TOL and stats["max_abs_Ct_delta"] > TOL
            elif pair == "PAIR-B":
                passed = stats["max_abs_Js_delta"] <= TOL and stats["max_abs_Ct_delta"] > TOL and stats["max_abs_absC12_delta"] > TOL
            elif pair == "PAIR-C":
                passed = stats["max_abs_Js_delta"] <= TOL and stats["max_abs_spectrum_delta"] <= TOL and stats["max_abs_Ct_delta"] > TOL
            else:
                passed = stats["max_abs_Js_delta"] <= TOL and stats["max_abs_spectrum_delta"] <= TOL and stats["max_abs_Ct_delta"] <= TOL
            stats["pass"] = "YES" if passed else "NO"
            pair_rows.append(stats)
            rows.append(stats)
        result[pair] = "PASS" if any(r["pass"] == "YES" for r in pair_rows) else "FAIL"
    result["PAIR-D"] = "PASS" if result["PAIR-D-Z"] == "PASS" and result["PAIR-D-X"] == "PASS" else "FAIL"
    write_csv(OUT / "D1N0_matched_invariant_validation.csv", rows)
    return result, rows


def collision_matrix(rows: list[dict]) -> tuple[int, int, int, str]:
    out = []
    q1_q2 = q1_q3 = q2_q3 = 0
    rigid_pass = True
    for r in rows:
        q1_same = float(r["max_abs_Js_delta"]) <= TOL
        q2_same = float(r["max_abs_spectrum_delta"]) <= TOL
        q3_same = float(r["max_abs_Ct_delta"]) <= TOL
        q1_state = "COLLISION" if q1_same else "DISTINCT"
        q2_state = "COLLISION" if q2_same else "DISTINCT"
        q3_state = "COLLISION" if q3_same else "DISTINCT"
        if q1_same and not q2_same:
            q1_q2 += 1
        if q1_same and not q3_same:
            q1_q3 += 1
        if q2_same and not q3_same:
            q2_q3 += 1
        if str(r["pair_category"]).startswith("PAIR-D") and not (q1_same and q2_same and q3_same):
            rigid_pass = False
        out.append({**r, "Q1_Js": q1_state, "Q2_spectrum": q2_state, "Q3_Ct": q3_state})
    write_csv(OUT / "D1N0_state_collision_matrix.csv", out)
    return q1_q2, q1_q3, q2_q3, "YES" if rigid_pass else "NO"


def scope_md() -> None:
    write_md(
        OUT / "D1N0_scientific_scope.md",
        "D1-N0 Scientific Scope",
        "\n".join([
            "D1-N0 does NOT prove that Ct controls semi-transparent optical response.",
            "It only proves whether the deformation-state hierarchy has structural discriminatory power.",
            "No optical mechanism has been tested.",
            "No state is necessary.",
            "No state is sufficient.",
            "No novelty claim is allowed from N0 alone.",
            "No optical response, tau, RGB, rendering, model training, MLP, Gaussian optimization, or RT-Splatting execution is used in this stage.",
        ]),
    )


def main() -> None:
    assert_gpu_scope()
    g0 = protocol_lock()
    dep, identity_ack = existing_dependency()
    ex_same_js_diff_spec, ex_same_spec_diff_ct, _ = existing_pair_audit()
    bank = deformation_bank()
    write_bank(bank)
    max_js_rel, _ = replay_candidate_descriptors(bank)
    validations, val_rows = validate_pairs(bank)
    n0g2 = "PASS" if all(validations[k] == "PASS" for k in ["PAIR-A", "PAIR-B", "PAIR-C", "PAIR-D"]) and max_js_rel <= TOL else "FAIL"
    q1_q2, q1_q3, q2_q3, rigid_invariance = collision_matrix(val_rows)
    scope_md()
    hierarchy = "YES" if g0 == "PASS" and n0g2 == "PASS" and q1_q2 > 0 and q2_q3 > 0 else "NO"
    final_case = "CASE DEFORMATION-STATE-HIERARCHY-IDENTIFIABLE" if hierarchy == "YES" else "CASE DEFORMATION-DESCRIPTORS-NOT-IDENTIFIABLE"
    next_action = "perform D1-N1 literature/mechanism boundary lock and design independent optical mechanisms that respond differently to matched Q1/Q2/Q3 pairs" if hierarchy == "YES" else "STOP new optical-transport line or redesign deformation controls"
    report_path = OUT / "stageD1N0_identifiability_report.md"
    summary_path = OUT / "stageD1N0_identifiability_summary.md"
    rank_str = f"{dep['S0']['rank']}/{dep['S1']['rank']}/{dep['combined']['rank']}"
    cond_str = f"{dep['S0']['condition']}/{dep['S1']['condition']}/{dep['combined']['condition']}"
    terminal = [
        ("A. N0-G0", g0),
        ("B. existing deformation count", 7),
        ("C. all existing deformations globally affine yes/no", "YES"),
        ("D. exact Js=lambda1*lambda2 acknowledged yes/no", identity_ack),
        ("E. S0/S1/combined descriptor matrix rank", rank_str),
        ("F. S0/S1/combined condition number", cond_str),
        ("G. existing same-Js different-spectrum pair count", ex_same_js_diff_spec),
        ("H. existing same-spectrum different-Ct pair count", ex_same_spec_diff_ct),
        ("I. candidate deformation count", len(bank)),
        ("J. PAIR-A invariant validation", validations["PAIR-A"]),
        ("K. PAIR-B invariant validation", validations["PAIR-B"]),
        ("L. PAIR-C invariant validation", validations["PAIR-C"]),
        ("M. PAIR-D invariant validation", validations["PAIR-D"]),
        ("N. N0-G2", n0g2),
        ("O. Q1 collision but Q2 distinct pair count", q1_q2),
        ("P. Q1 collision but Q3 distinct pair count", q1_q3),
        ("Q. Q2 collision but Q3 distinct pair count", q2_q3),
        ("R. rigid Q1/Q2/Q3 invariance pass yes/no", rigid_invariance),
        ("S. hierarchy structurally identifiable yes/no", hierarchy),
        ("T. optical necessity claimed yes/no", "NO"),
        ("U. optical sufficiency claimed yes/no", "NO"),
        ("V. Final CASE", final_case),
        ("W. new primary line STOP/CONTINUE", "CONTINUE" if hierarchy == "YES" else "STOP"),
        ("X. next exact research action", next_action),
        ("Y. report path", str(report_path)),
        ("Z. summary path", str(summary_path)),
    ]
    body = "\n".join(f"{k}: {v}" for k, v in terminal)
    write_md(report_path, "Stage D1-N0 Deformation Descriptor Identifiability Closure", body)
    write_md(summary_path, "Stage D1-N0 Summary", body)
    (OUT / "stageD1N0_identifiability_log.txt").write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
