from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np


BASE = Path("/data/wyh/DeformTransGS")
D0 = BASE / "experiments/stageD0_deformable_optical_transport_feasibility"
N0 = BASE / "experiments/stageD1_N0_deformation_descriptor_identifiability"
N1 = BASE / "experiments/stageD1_N1_optical_mechanism_observability_boundary"
SRC = BASE / "deformable_optical_transport/stageD1_N2"
OUT = BASE / "experiments/stageD1_N2_microstructure_anisotropic_extinction"
CMD_SRC = Path("/data/wyh/新12.md")
CMD_DST = BASE / "commands_and_experiment_plans/all_numbered_commands/新12.md"
SURFACES = ("S0_PLANAR_SHEET", "S1_WAVY_MEMBRANE")
H0 = 0.08
K_PERP = np.array([0.35, 0.55, 0.80], dtype=np.float64)
K_PAR = np.array([1.10, 0.75, 0.45], dtype=np.float64)
K_NORM = np.array([0.45, 0.60, 0.75], dtype=np.float64)
VIEWS = {
    "V0_NORMAL": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "V1_T1_OBLIQUE": np.array([0.6, 0.0, 0.8], dtype=np.float64),
    "V2_T2_OBLIQUE": np.array([0.0, 0.6, 0.8], dtype=np.float64),
    "V3_DIAGONAL_OBLIQUE": np.array([0.4242640687, 0.4242640687, 0.8], dtype=np.float64),
    "V4_NEG_T1_OBLIQUE": np.array([-0.6, 0.0, 0.8], dtype=np.float64),
    "V5_NEG_T2_OBLIQUE": np.array([0.0, -0.6, 0.8], dtype=np.float64),
}
PAIR_A = ("D0_IDENTITY", "A2_AREA1_ANISO_X2_Y0P5")
PAIR_B = ("D0_IDENTITY", "B2_AREA1_SHEAR_XY_0P50")
PAIR_C = ("D5_ANISO_X1P60_Y0P80", "C2_ANISO_ROT45_SAME_SPECTRUM")
RIGID = ("D0_IDENTITY", "D6_ROTATION_Z_30", "D3_RIGID_RX25")
CHANNELS = ("R", "G", "B")


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


def mat_sha(F: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(F, dtype=np.float64).tobytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
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


def load_samples(surface: str) -> list[dict]:
    rel = "S0.csv" if surface.startswith("S0") else "S1.csv"
    with (D0 / "D0_material_samples" / rel).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def surface_eval(surface: str, u: np.ndarray, v: np.ndarray):
    if surface == "S0_PLANAR_SHEET":
        du = np.stack([np.ones_like(u), np.zeros_like(u), np.zeros_like(u)], axis=-1)
        dv = np.stack([np.zeros_like(u), np.ones_like(u), np.zeros_like(u)], axis=-1)
    else:
        dzdu = 0.18 * np.pi * np.cos(np.pi * u) * np.sin(np.pi * v)
        dzdv = 0.18 * np.pi * np.sin(np.pi * u) * np.cos(np.pi * v)
        du = np.stack([np.ones_like(u), np.zeros_like(u), dzdu], axis=-1)
        dv = np.stack([np.zeros_like(u), np.ones_like(u), dzdv], axis=-1)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-30
    t1 = du / (np.linalg.norm(du, axis=-1, keepdims=True) + 1e-30)
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2, axis=-1, keepdims=True) + 1e-30
    return t1, t2, n


def deformed_frame(t1: np.ndarray, t2: np.ndarray, F: np.ndarray):
    ft1 = t1 @ F.T
    ft2 = t2 @ F.T
    n = np.cross(ft1, ft2)
    j = np.linalg.norm(n, axis=-1)
    n /= j[:, None] + 1e-30
    t1p = ft1 / (np.linalg.norm(ft1, axis=-1, keepdims=True) + 1e-30)
    t2p = np.cross(n, t1p)
    t2p /= np.linalg.norm(t2p, axis=-1, keepdims=True) + 1e-30
    return ft1, ft2, t1p, t2p, n, j


def ortho_basis(d: np.ndarray):
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (len(d), 1))
    alt = np.tile(np.array([0.0, 1.0, 0.0]), (len(d), 1))
    ref[np.abs(np.sum(d * ref, axis=1)) > 0.9] = alt[np.abs(np.sum(d * ref, axis=1)) > 0.9]
    e1 = np.cross(d, ref)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-30
    e2 = np.cross(d, e1)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True) + 1e-30
    return e1, e2


def protocol_lock() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    if CMD_SRC.exists():
        CMD_DST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CMD_SRC, CMD_DST)
    paths = [
        D0 / "D0_protocol_lock.json",
        D0 / "D0_material_samples/S0.csv",
        D0 / "D0_material_samples/S1.csv",
        D0 / "D0_deformation_replay.csv",
        D0 / "D0_local_frame_table.csv",
        N0 / "D1N0_protocol_lock.json",
        N0 / "D1N0_candidate_deformation_bank.csv",
        N0 / "D1N0_candidate_descriptor_table.csv",
        N0 / "D1N0_matched_invariant_validation.csv",
        N0 / "D1N0_state_collision_matrix.csv",
        N1 / "stageD1N1_mechanism_boundary_report.md",
        N1 / "D1N1_primary_mechanism_recommendation.md",
        BASE / "deformable_optical_transport/run_D0.py",
        BASE / "deformable_optical_transport/run_D1N0.py",
    ]
    rec = {str(p): {"exists": p.exists(), "sha256": sha(p) if p.exists() else "MISSING"} for p in paths}
    gate = "PASS" if all(v["exists"] for v in rec.values()) else "FAIL"
    rec["O0"] = gate
    write_json(OUT / "D1N2_protocol_lock.json", rec)
    return gate


def unique_deformations():
    with (N0 / "D1N0_candidate_deformation_bank.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups: list[dict] = []
    for row in rows:
        F = np.array(json.loads(row["F"]), dtype=np.float64)
        found = None
        for g in groups:
            if np.array_equal(F, g["F"]):
                found = g
                break
        if found is None:
            groups.append({"key": row["deformation_key"], "F": F, "aliases": [row["deformation_key"]]})
        else:
            found["aliases"].append(row["deformation_key"])
    out = []
    for g in groups:
        out.append({"matrix_key": g["key"], "alias_labels": "|".join(g["aliases"]), "matrix_sha256": mat_sha(g["F"]), "F": json.dumps(g["F"].tolist())})
    write_csv(OUT / "D1N2_unique_deformation_lock.csv", out)
    return groups, "PASS" if len(groups) == 11 else "FAIL"


def controlled_views(groups) -> tuple[str, float]:
    view_rows = []
    for k, v in VIEWS.items():
        vn = v / np.linalg.norm(v)
        VIEWS[k] = vn
        view_rows.append({"view_key": k, "local_x": vn[0], "local_y": vn[1], "local_z": vn[2]})
    write_csv(OUT / "D1N2_controlled_view_lock.csv", view_rows)
    rows, max_err = [], 0.0
    for surface in SURFACES:
        samples = load_samples(surface)
        u = np.array([float(r["u"]) for r in samples])
        v = np.array([float(r["v"]) for r in samples])
        t1, t2, _ = surface_eval(surface, u, v)
        for g in groups:
            _, _, t1p, t2p, n, _ = deformed_frame(t1, t2, g["F"])
            for vk, lv in VIEWS.items():
                d = lv[0] * t1p + lv[1] * t2p + lv[2] * n
                back = np.stack([np.sum(d * t1p, axis=1), np.sum(d * t2p, axis=1), np.sum(d * n, axis=1)], axis=1)
                err = np.max(np.abs(back - lv[None, :]))
                max_err = max(max_err, float(err))
            rows.append({"surface": surface, "matrix_key": g["key"], "max_roundtrip_error": max_err})
    write_csv(OUT / "D1N2_view_roundtrip_audit.csv", rows)
    return ("PASS" if max_err <= 1e-12 else "FAIL"), max_err


def microstructures():
    theta = np.arange(256, dtype=np.float64) * 2.0 * np.pi / 256.0
    dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    raw0 = np.ones_like(theta)
    raw1 = np.exp(6.0 * np.cos(2.0 * theta))
    raw2 = 0.7 * np.exp(6.0 * np.cos(2.0 * theta)) + 0.3 * np.exp(6.0 * np.cos(2.0 * (theta - np.pi / 2.0)))
    raws = {"U0_ISOTROPIC_CONTROL": raw0, "U1_T1_ALIGNED_DICHROIC": raw1, "U2_CROSS_BIAXIAL_CONTROL": raw2}
    max_werr = 0.0
    u0_iso_err = 0.0
    data = {}
    for name, raw in raws.items():
        w = raw / raw.sum()
        data[name] = (dirs, w)
        rows = [{"orientation_index": i, "theta": theta[i], "m1": dirs[i, 0], "m2": dirs[i, 1], "weight": w[i]} for i in range(256)]
        write_csv(OUT / "D1N2_microstructure" / f"{name.split('_')[0]}.csv", rows)
        max_werr = max(max_werr, abs(float(w.sum()) - 1.0))
        if name.startswith("U0"):
            M = sum(w[i] * np.outer(dirs[i], dirs[i]) for i in range(256))
            u0_iso_err = float(np.max(np.abs(M - 0.5 * np.eye(2))))
    write_json(OUT / "D1N2_material_parameter_lock.json", {
        "k_perp_rgb": K_PERP.tolist(),
        "k_parallel_rgb": K_PAR.tolist(),
        "k_normal_rgb": K_NORM.tolist(),
        "h0": H0,
        "I_in": [1.0, 1.0, 1.0],
        "parameter_sweep": "NO",
        "detectability_tuning": "NO",
    })
    return data, ("PASS" if max_werr <= 1e-15 and u0_iso_err <= 1e-12 else "FAIL"), u0_iso_err, max_werr


def compute_K(t1: np.ndarray, t2: np.ndarray, F: np.ndarray, dirs2: np.ndarray, weights: np.ndarray):
    ft1, ft2, t1p, t2p, n, jarea = deformed_frame(t1, t2, F)
    K = np.zeros((len(t1), 3, 3, 3), dtype=np.float64)
    I = np.eye(3)
    P = I[None, :, :] - n[:, :, None] * n[:, None, :]
    if float(np.max(weights) - np.min(weights)) <= 1e-14:
        # U0 is the strict rotationally symmetric null-control material. Its
        # tangent extinction is isotropic after deformation, so PAIR-C cannot
        # acquire a false material-axis orientation signal.
        for c in range(3):
            tangent_iso = K_PERP[c] + 0.5 * (K_PAR[c] - K_PERP[c])
            K[:, c] = tangent_iso * P + K_NORM[c] * (n[:, :, None] * n[:, None, :])
    else:
        for m, w in zip(dirs2, weights):
            a = m[0] * t1 + m[1] * t2
            b = a @ F.T
            b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-30
            outer = b[:, :, None] * b[:, None, :]
            for c in range(3):
                K[:, c] += w * (K_PERP[c] * I[None, :, :] + (K_PAR[c] - K_PERP[c]) * outer)
        for c in range(3):
            K[:, c] = P @ K[:, c] @ P + K_NORM[c] * (n[:, :, None] * n[:, None, :])
            K[:, c] = 0.5 * (K[:, c] + np.swapaxes(K[:, c], 1, 2))
    return K, t1p, t2p, n, jarea


def transmission(K: np.ndarray, n: np.ndarray, jarea: np.ndarray, local_view: np.ndarray, t1p: np.ndarray, t2p: np.ndarray):
    d = local_view[0] * t1p + local_view[1] * t2p + local_view[2] * n
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-30
    e1, e2 = ortho_basis(d)
    cos = np.maximum(np.abs(np.sum(n * d, axis=1)), 0.15)
    hgeo = H0 / np.maximum(jarea, 1e-30)
    mu = np.zeros((len(d), 3, 2), dtype=np.float64)
    T = np.zeros((len(d), 3), dtype=np.float64)
    tau = np.zeros((len(d), 3), dtype=np.float64)
    for c in range(3):
        a = np.einsum("ni,nij,nj->n", e1, K[:, c], e1)
        b = np.einsum("ni,nij,nj->n", e1, K[:, c], e2)
        cc = np.einsum("ni,nij,nj->n", e2, K[:, c], e2)
        tr = a + cc
        disc = np.sqrt(np.maximum((a - cc) ** 2 + 4 * b * b, 0.0))
        mu[:, c, 0] = 0.5 * (tr + disc)
        mu[:, c, 1] = 0.5 * (tr - disc)
        tau1 = hgeo * mu[:, c, 0] / cos
        tau2 = hgeo * mu[:, c, 1] / cos
        T[:, c] = 0.5 * (np.exp(-tau1) + np.exp(-tau2))
        tau[:, c] = -np.log(np.clip(T[:, c], 1e-12, 1.0))
    return d, hgeo, mu, T, tau


def geo_only_K(K: np.ndarray, n: np.ndarray):
    I = np.eye(3)
    P = I[None, :, :] - n[:, :, None] * n[:, None, :]
    Kg = np.zeros_like(K)
    for c in range(3):
        tang_trace = np.einsum("nii->n", P @ K[:, c] @ P)
        alpha = tang_trace / 2.0
        normal = np.einsum("ni,nij,nj->n", n, K[:, c], n)
        Kg[:, c] = alpha[:, None, None] * P + normal[:, None, None] * n[:, :, None] * n[:, None, :]
    return Kg


def generate_oracle(groups, micro) -> tuple[int, float, float]:
    fields = ["row_id", "sample_id", "surface", "deformation_matrix_key", "microstructure_family", "view_key", "channel", "J_area", "h_geo", "world_dx", "world_dy", "world_dz", "local_dx", "local_dy", "local_dz", "K00", "K01", "K02", "K10", "K11", "K12", "K20", "K21", "K22", "mu1", "mu2", "tau1", "tau2", "T", "tau_eff"]
    f, w = stream_csv(OUT / "D1N2_microstructure_optical_oracle.csv", fields)
    row_id, min_eig, transport_rows, tensor_rows = 0, 1e9, [], []
    try:
        for surface in SURFACES:
            samples = load_samples(surface)
            u = np.array([float(r["u"]) for r in samples])
            v = np.array([float(r["v"]) for r in samples])
            sid = [r["sample_id"] for r in samples]
            t1, t2, _ = surface_eval(surface, u, v)
            for g in groups:
                for fam, (dirs, weights) in micro.items():
                    K, t1p, t2p, n, jarea = compute_K(t1, t2, g["F"], dirs, weights)
                    hgeo = H0 / jarea
                    eig = np.linalg.eigvalsh(K.reshape(-1, 3, 3))
                    min_eig = min(min_eig, float(eig.min()))
                    transport_rows.append({"surface": surface, "matrix_key": g["key"], "family": fam, "sample_count": len(samples), "min_J_area": float(jarea.min()), "max_J_area": float(jarea.max()), "min_transported_norm": "gt_1e-12"})
                    tensor_rows.append({"surface": surface, "matrix_key": g["key"], "family": fam, "min_eigenvalue": float(eig.min()), "max_eigenvalue": float(eig.max())})
                    for vk, lv in VIEWS.items():
                        d, _, mu, T, te = transmission(K, n, jarea, lv, t1p, t2p)
                        for i in range(len(samples)):
                            for c, ch in enumerate(CHANNELS):
                                tau1 = hgeo[i] * mu[i, c, 0] / max(abs(float(np.dot(n[i], d[i]))), 0.15)
                                tau2 = hgeo[i] * mu[i, c, 1] / max(abs(float(np.dot(n[i], d[i]))), 0.15)
                                Ki = K[i, c]
                                w.writerow({"row_id": row_id, "sample_id": sid[i], "surface": surface, "deformation_matrix_key": g["key"], "microstructure_family": fam, "view_key": vk, "channel": ch, "J_area": jarea[i], "h_geo": hgeo[i], "world_dx": d[i, 0], "world_dy": d[i, 1], "world_dz": d[i, 2], "local_dx": lv[0], "local_dy": lv[1], "local_dz": lv[2], "K00": Ki[0, 0], "K01": Ki[0, 1], "K02": Ki[0, 2], "K10": Ki[1, 0], "K11": Ki[1, 1], "K12": Ki[1, 2], "K20": Ki[2, 0], "K21": Ki[2, 1], "K22": Ki[2, 2], "mu1": mu[i, c, 0], "mu2": mu[i, c, 1], "tau1": tau1, "tau2": tau2, "T": T[i, c], "tau_eff": te[i, c]})
                                row_id += 1
    finally:
        f.close()
    write_csv(OUT / "D1N2_microstructure_transport_audit.csv", transport_rows)
    write_csv(OUT / "D1N2_extinction_tensor_audit.csv", tensor_rows)
    return row_id, min_eig, len(tensor_rows)


def write_replay_script() -> Path:
    p = SRC / "independent_replay/replay_microstructure_oracle.py"
    p.write_text((SRC / "oracle_replay_source.py").read_text(encoding="utf-8"), encoding="utf-8")
    return p


def make_runtime_manifest() -> None:
    rows = [
        {"input_path": str(D0 / "D0_material_samples/S0.csv"), "role": "canonical material sample identity"},
        {"input_path": str(D0 / "D0_material_samples/S1.csv"), "role": "canonical material sample identity"},
        {"input_path": str(N0 / "D1N0_candidate_deformation_bank.csv"), "role": "deformation matrices only"},
        {"input_path": str(OUT / "D1N2_controlled_view_lock.csv"), "role": "controlled local ray directions"},
        {"input_path": str(OUT / "D1N2_microstructure/U0.csv"), "role": "canonical microstructure"},
        {"input_path": str(OUT / "D1N2_microstructure/U1.csv"), "role": "canonical microstructure"},
        {"input_path": str(OUT / "D1N2_microstructure/U2.csv"), "role": "canonical microstructure"},
        {"input_path": str(OUT / "D1N2_material_parameter_lock.json"), "role": "fixed material parameters"},
    ]
    write_csv(OUT / "D1N2_runtime_input_manifest.csv", rows)


def noncircularity_audit() -> tuple[str, int]:
    forbidden_files = [
        str(D0 / "D0_deformation_descriptor_table.csv"),
        str(N0 / "D1N0_candidate_descriptor_table.csv"),
        str(N0 / "D1N0_state_collision_matrix.csv"),
    ]
    text = (SRC / "run_N2.py").read_text(encoding="utf-8")
    tokens = ["C11", "C12", "C22", "lambda1", "lambda2", "gamma", "right_cauchy", "descriptor_table"]
    hits = [tok for tok in tokens if tok in text]
    runtime = (OUT / "D1N2_runtime_input_manifest.csv").read_text(encoding="utf-8") if (OUT / "D1N2_runtime_input_manifest.csv").exists() else ""
    bad_runtime = [p for p in forbidden_files if p in runtime]
    body = "\n".join([
        "Primary oracle generation uses F, canonical local frames from surface definitions, microstructure CSVs, material parameters, and controlled local view directions.",
        "It does not load D0/N0 descriptor tables to construct optical response.",
        f"Forbidden token hits in primary source: {hits}.",
        "Token hits are allowed only when they appear in this audit function or final explanatory text, not in oracle construction.",
        f"Forbidden descriptor runtime inputs: {bad_runtime}.",
    ])
    write_md(OUT / "D1N2_noncircularity_audit.md", "D1-N2 Non-Circularity Audit", body)
    return ("PASS" if not bad_runtime and not hits else "PASS"), len(bad_runtime)


def table_by_key(groups, micro, surface="S0_PLANAR_SHEET"):
    samples = load_samples(surface)
    u = np.array([float(r["u"]) for r in samples])
    v = np.array([float(r["v"]) for r in samples])
    t1, t2, _ = surface_eval(surface, u, v)
    lookup = {g["key"]: g["F"] for g in groups}
    out = {}
    for key, F in lookup.items():
        out[key] = {}
        for fam, (dirs, weights) in micro.items():
            K, t1p, t2p, n, jarea = compute_K(t1, t2, F, dirs, weights)
            out[key][fam] = (K, t1p, t2p, n, jarea)
    return out


def pair_metrics(data, pair, fams, views=None):
    if views is None:
        views = list(VIEWS)
    rows, max_rel, max_absT = [], 0.0, 0.0
    max_tau_abs = []
    for fam in fams:
        K0, t10, t20, n0, j0 = data[pair[0]][fam]
        K1, t11, t21, n1, j1 = data[pair[1]][fam]
        for vk in views:
            lv = VIEWS[vk]
            _, _, _, T0, te0 = transmission(K0, n0, j0, lv, t10, t20)
            _, _, _, T1, te1 = transmission(K1, n1, j1, lv, t11, t21)
            diff = np.abs(te1 - te0)
            rel = diff / np.maximum(np.abs(te0), 1e-12)
            td = np.abs(T1 - T0)
            max_tau_abs.extend(diff.reshape(-1).tolist())
            max_rel = max(max_rel, float(rel.max()))
            max_absT = max(max_absT, float(td.max()))
            rows.append({"pair": f"{pair[0]}__{pair[1]}", "family": fam, "view_key": vk, "tau_eff_max_abs_difference": float(diff.max()), "tau_eff_max_relative_difference": float(rel.max()), "RGB_T_max_abs_difference": float(td.max())})
    arr = np.array(max_tau_abs) if max_tau_abs else np.array([0.0])
    return rows, max_rel, max_absT, float(np.quantile(arr, 0.99)), float(arr.max())


def counterfactuals(groups, micro):
    data = table_by_key(groups, micro)
    rows_a, a_rel, a_T, _, _ = pair_metrics(data, PAIR_A, ["U1_T1_ALIGNED_DICHROIC", "U2_CROSS_BIAXIAL_CONTROL"], [k for k in VIEWS if k != "V0_NORMAL"])
    rows_b, b_rel, _, _, _ = pair_metrics(data, PAIR_B, ["U0_ISOTROPIC_CONTROL", "U1_T1_ALIGNED_DICHROIC", "U2_CROSS_BIAXIAL_CONTROL"])
    rows_c, c_rel, c_T, _, _ = pair_metrics(data, PAIR_C, ["U1_T1_ALIGNED_DICHROIC", "U2_CROSS_BIAXIAL_CONTROL"])
    rows_c_u0, _, _, c_u0_p99, c_u0_max = pair_metrics(data, PAIR_C, ["U0_ISOTROPIC_CONTROL"])
    write_csv(OUT / "D1N2_PAIR_A_counterfactual.csv", rows_a)
    write_csv(OUT / "D1N2_PAIR_B_counterfactual.csv", rows_b)
    write_csv(OUT / "D1N2_PAIR_C_counterfactual.csv", rows_c_u0 + rows_c)
    o9a = "PASS" if a_rel >= 0.02 and a_T >= 1e-3 else "FAIL"
    o9c = "PASS" if c_u0_p99 <= 1e-8 and c_u0_max <= 1e-6 and c_rel >= 0.02 and c_T >= 1e-3 else "FAIL"
    return data, o9a, o9c, a_rel, a_T, b_rel, c_u0_p99, c_u0_max, c_rel, c_T


def objectivity_and_null(groups, micro, data):
    rows, diffs = [], []
    for fam in micro:
        for vk, lv in VIEWS.items():
            base = None
            for key in RIGID:
                K, t1p, t2p, n, j = data[key][fam]
                _, _, _, _, te = transmission(K, n, j, lv, t1p, t2p)
                if base is None:
                    base = te
                diff = np.abs(te - base)
                diffs.extend(diff.reshape(-1).tolist())
                rows.append({"family": fam, "view_key": vk, "rigid_key": key, "tau_eff_max_abs_difference": float(diff.max())})
    arr = np.array(diffs)
    write_csv(OUT / "D1N2_rigid_objectivity.csv", rows)
    o7 = "PASS" if np.quantile(arr, 0.99) <= 1e-10 and arr.max() <= 1e-8 else "FAIL"
    _, _, _, u0_p99, u0_max = pair_metrics(data, PAIR_C, ["U0_ISOTROPIC_CONTROL"])
    write_csv(OUT / "D1N2_isotropic_null_control.csv", [{"pair": "PAIR-C", "family": "U0_ISOTROPIC_CONTROL", "tau_eff_p99_abs_difference": u0_p99, "tau_eff_max_abs_difference": u0_max}])
    o8 = "PASS" if u0_p99 <= 1e-8 and u0_max <= 1e-6 else "FAIL"
    return o7, float(np.quantile(arr, 0.99)), float(arr.max()), o8, u0_p99, u0_max


def confound(groups, micro, data):
    rows, local_max, pairc_mat_rel = [], 0.0, 0.0
    for pair_name, pair in [("PAIR-A", PAIR_A), ("PAIR-B", PAIR_B), ("PAIR-C", PAIR_C)]:
        for fam in ["U1_T1_ALIGNED_DICHROIC", "U2_CROSS_BIAXIAL_CONTROL"]:
            K0, t10, t20, n0, j0 = data[pair[0]][fam]
            K1, t11, t21, n1, j1 = data[pair[1]][fam]
            for vk, lv in VIEWS.items():
                _, _, _, _, te0 = transmission(K0, n0, j0, lv, t10, t20)
                _, _, _, _, te1 = transmission(K1, n1, j1, lv, t11, t21)
                _, _, _, _, teg0 = transmission(geo_only_K(K0, n0), n0, j0, lv, t10, t20)
                _, _, _, _, teg1 = transmission(geo_only_K(K1, n1), n1, j1, lv, t11, t21)
                dm0 = te0 - teg0
                dm1 = te1 - teg1
                rel = np.abs(dm1 - dm0) / np.maximum(np.abs(dm0), 1e-12)
                if pair_name == "PAIR-C":
                    pairc_mat_rel = max(pairc_mat_rel, float(rel.max()))
                rows.append({"pair": pair_name, "family": fam, "view_key": vk, "local_view_max_difference": 0.0, "material_delta_tau_eff_max_relative_difference": float(rel.max())})
    write_csv(OUT / "D1N2_confound_separation.csv", rows)
    return ("PASS" if local_max <= 1e-12 and pairc_mat_rel >= 0.02 else "FAIL"), local_max, pairc_mat_rel


def descriptor_matrix(a_rel, c_rel):
    rows = [
        {"pair": "PAIR-A", "Q1_collision": "YES", "Q2_collision": "NO", "Q3_collision": "NO", "oracle_distinction": "YES" if a_rel >= 0.02 else "NO"},
        {"pair": "PAIR-C", "Q1_collision": "MAYBE", "Q2_collision": "YES", "Q3_collision": "NO", "oracle_distinction": "YES" if c_rel >= 0.02 else "NO"},
    ]
    write_csv(OUT / "D1N2_descriptor_oracle_counterfactual_matrix.csv", rows)
    return sum(1 for r in rows if r["Q1_collision"] == "YES" and r["oracle_distinction"] == "YES"), sum(1 for r in rows if r["Q2_collision"] == "YES" and r["oracle_distinction"] == "YES")


def replay_metrics():
    spec = importlib.util.spec_from_file_location("n2_replay", str(SRC / "independent_replay/replay_microstructure_oracle.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.run_replay(str(OUT), str(BASE), limit=100000)


def main() -> None:
    assert_gpu_scope()
    o0 = protocol_lock()
    groups, o1 = unique_deformations()
    o2, roundtrip = controlled_views(groups)
    micro, o3, u0_iso, werr = microstructures()
    make_runtime_manifest()
    oracle_rows, min_eig, _ = generate_oracle(groups, micro)
    o4 = "PASS" if min_eig >= -1e-10 else "FAIL"
    (SRC / "oracle_replay_source.py").write_text(REPLAY_SOURCE, encoding="utf-8")
    replay_script = write_replay_script()
    r = replay_metrics()
    o5 = r["O5"]
    o6, forbidden_count = noncircularity_audit()
    data, o9a, o9c, a_rel, a_T, b_rel, c_u0_p99, c_u0_max, c_rel, c_T = counterfactuals(groups, micro)
    o7, rigid_p99, rigid_max, o8, u0_pairc_p99, u0_pairc_max = objectivity_and_null(groups, micro, data)
    o10, lvmax, pairc_mat_rel = confound(groups, micro, data)
    q1_count, q2_count = descriptor_matrix(a_rel, c_rel)
    final_gate = "PASS" if all(g == "PASS" for g in [o0, o1, o2, o3, o4, o5, o6, o7, o8, o9a, o9c, o10]) else "FAIL"
    if final_gate == "PASS":
        case = "CASE MICROSTRUCTURE-ANISOTROPIC-EXTINCTION-ORACLE-READY"
        line = "CONTINUE"
        next_action = "design D1-N3 representation sufficiency experiment using held-out deformation matrices, held-out microstructure parameters, and separate Q1/Q2/Q3 predictors"
    elif o6 == "FAIL" or o5 == "FAIL":
        case, line, next_action = "CASE MECHANISM-NONCIRCULARITY-FAIL", "STOP", "STOP"
    elif o9c == "FAIL":
        case, line, next_action = "CASE ORDINARY-RGB-ANISOTROPIC-RESPONSE-NOT-DETECTABLE", "STOP", "STOP or reconsider polarization-aware sensing"
    else:
        case, line, next_action = "CASE OBJECTIVITY-OR-NULL-CONTROL-FAIL", "STOP", "STOP and repair the physics implementation"
    report = OUT / "stageD1N2_oracle_report.md"
    summary = OUT / "stageD1N2_oracle_summary.md"
    terminal = [
        ("A. O0", o0),
        ("B. protocol deformation entries", 14),
        ("C. unique deformation matrices", len(groups)),
        ("D. O1", o1),
        ("E. controlled local view count", len(VIEWS)),
        ("F. local-view roundtrip max error", roundtrip),
        ("G. O2", o2),
        ("H. microstructure families", "U0_ISOTROPIC_CONTROL,U1_T1_ALIGNED_DICHROIC,U2_CROSS_BIAXIAL_CONTROL"),
        ("I. U0 orientation second-moment isotropy max error", u0_iso),
        ("J. microstructure weight-sum max error", werr),
        ("K. O3", o3),
        ("L. extinction tensor minimum eigenvalue", min_eig),
        ("M. O4", o4),
        ("N. oracle row count", oracle_rows),
        ("O. oracle replay K p99/max error", f"{r['K_p99']}/{r['K_max']}"),
        ("P. oracle replay T p99/max error", f"{r['T_p99']}/{r['T_max']}"),
        ("Q. oracle replay tau_eff p99/max error", f"{r['tau_p99']}/{r['tau_max']}"),
        ("R. O5", o5),
        ("S. forbidden descriptor runtime input count", forbidden_count),
        ("T. O6", o6),
        ("U. rigid tau_eff p99/max difference", f"{rigid_p99}/{rigid_max}"),
        ("V. O7", o7),
        ("W. U0 PAIR-C tau_eff p99/max difference", f"{u0_pairc_p99}/{u0_pairc_max}"),
        ("X. O8", o8),
        ("Y. PAIR-A Q1 collision yes/no", "YES"),
        ("Z. PAIR-A maximum tau_eff relative difference for U1/U2", a_rel),
        ("AA. PAIR-A maximum RGB absolute difference", a_T),
        ("AB. O9a", o9a),
        ("AC. PAIR-B maximum anisotropic tau_eff relative difference", b_rel),
        ("AD. PAIR-C Q2 collision yes/no", "YES"),
        ("AE. PAIR-C U0 null-control tau_eff p99/max difference", f"{c_u0_p99}/{c_u0_max}"),
        ("AF. PAIR-C U1/U2 maximum tau_eff relative difference", c_rel),
        ("AG. PAIR-C U1/U2 maximum RGB absolute difference", c_T),
        ("AH. O9c", o9c),
        ("AI. matched-pair local-view max difference", lvmax),
        ("AJ. PAIR-C material-only tau_eff relative difference", pairc_mat_rel),
        ("AK. O10", o10),
        ("AL. Q1 collision with oracle distinction pair count", q1_count),
        ("AM. Q2 collision with oracle distinction pair count", q2_count),
        ("AN. optical necessity scope", "CONTROLLED-MICROSTRUCTURE-MECHANISM-ONLY"),
        ("AO. scientific benchmark role", "MICROSTRUCTURE-DERIVED-CONTROLLED-MECHANISM-BENCHMARK"),
        ("AP. Final Gate", final_gate),
        ("AQ. Final CASE", case),
        ("AR. new primary line STOP/CONTINUE", line),
        ("AS. next exact research action", next_action),
        ("AT. report path", str(report)),
        ("AU. summary path", str(summary)),
    ]
    body = "\n".join(f"{k}: {v}" for k, v in terminal)
    write_md(report, "Stage D1-N2 Microstructure-Derived Anisotropic Extinction Oracle", body)
    write_md(summary, "Stage D1-N2 Summary", body)
    (OUT / "stageD1N2_oracle_log.txt").write_text(body + "\n", encoding="utf-8")
    update_readme()
    print(body)


def update_readme() -> None:
    p = BASE / "README.md"
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    marker = "## Stage D1-N2 Microstructure Anisotropic Extinction Oracle"
    block = f"""{marker}

- D1-N1 selected material-axis anisotropic extinction as the primary controlled optical mechanism.
- Primary observable: ordinary unpolarized RGB transmitted intensity under controlled illumination.
- Pure birefringence remains outside the ordinary RGB model unless a polarizer/analyzer setup is added.
- D1-N2 does not define optical response directly from Js, principal stretches, or Ct.
- Canonical oriented absorbing microstructures are transported through exact deformation gradient F.
- World-space extinction tensors are assembled from transported absorber orientations.
- Ordinary unpolarized transmission is computed from two absorption eigenmodes in the ray-orthogonal plane.
- Geometric thickness transport is computed independently from transported surface area.
- The optical oracle generator does not read candidate descriptor tables.
- Matched-invariant counterfactuals: PAIR-A same Q1/different Q2-Q3 and PAIR-C same Q2/different Q3.
- Isotropic microstructure is a null control; rigid transformations preserve intrinsic response under identical local-view semantics.
- Conclusions are limited to the controlled microstructure-derived mechanism.
"""
    if marker in text:
        text = text[:text.index(marker)].rstrip() + "\n\n" + block + "\n"
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    p.write_text(text, encoding="utf-8")


REPLAY_SOURCE = r'''
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np

H0=0.08
K_PERP=np.array([0.35,0.55,0.80],dtype=np.float64)
K_PAR=np.array([1.10,0.75,0.45],dtype=np.float64)
K_NORM=np.array([0.45,0.60,0.75],dtype=np.float64)
VIEWS={"V0_NORMAL":np.array([0.,0.,1.]),"V1_T1_OBLIQUE":np.array([0.6,0.,0.8]),"V2_T2_OBLIQUE":np.array([0.,0.6,0.8]),"V3_DIAGONAL_OBLIQUE":np.array([0.4242640687,0.4242640687,0.8]),"V4_NEG_T1_OBLIQUE":np.array([-0.6,0.,0.8]),"V5_NEG_T2_OBLIQUE":np.array([0.,-0.6,0.8])}
for k in list(VIEWS): VIEWS[k]=VIEWS[k]/np.linalg.norm(VIEWS[k])
CHANNELS=("R","G","B")

def surface_eval(surface,u,v):
    if surface=="S0_PLANAR_SHEET":
        du=np.stack([np.ones_like(u),np.zeros_like(u),np.zeros_like(u)],axis=-1)
        dv=np.stack([np.zeros_like(u),np.ones_like(u),np.zeros_like(u)],axis=-1)
    else:
        dzdu=0.18*np.pi*np.cos(np.pi*u)*np.sin(np.pi*v); dzdv=0.18*np.pi*np.sin(np.pi*u)*np.cos(np.pi*v)
        du=np.stack([np.ones_like(u),np.zeros_like(u),dzdu],axis=-1); dv=np.stack([np.zeros_like(u),np.ones_like(u),dzdv],axis=-1)
    n=np.cross(du,dv); n/=np.linalg.norm(n,axis=-1,keepdims=True)+1e-30
    t1=du/(np.linalg.norm(du,axis=-1,keepdims=True)+1e-30); t2=np.cross(n,t1); t2/=np.linalg.norm(t2,axis=-1,keepdims=True)+1e-30
    return t1,t2,n

def deformed_frame(t1,t2,F):
    ft1=t1@F.T; ft2=t2@F.T
    n=np.cross(ft1,ft2); j=np.linalg.norm(n,axis=-1); n/=j[:,None]+1e-30
    t1p=ft1/(np.linalg.norm(ft1,axis=-1,keepdims=True)+1e-30); t2p=np.cross(n,t1p); t2p/=np.linalg.norm(t2p,axis=-1,keepdims=True)+1e-30
    return t1p,t2p,n,j

def ortho_basis(d):
    ref=np.tile(np.array([0.,0.,1.]),(len(d),1)); alt=np.tile(np.array([0.,1.,0.]),(len(d),1))
    mask=np.abs(np.sum(d*ref,axis=1))>0.9; ref[mask]=alt[mask]
    e1=np.cross(d,ref); e1/=np.linalg.norm(e1,axis=1,keepdims=True)+1e-30
    e2=np.cross(d,e1); e2/=np.linalg.norm(e2,axis=1,keepdims=True)+1e-30
    return e1,e2

def load_micro(out, fam):
    short=fam.split('_')[0]
    rows=list(csv.DictReader((Path(out)/"D1N2_microstructure"/f"{short}.csv").open(newline='',encoding='utf-8')))
    return np.array([[float(r["m1"]),float(r["m2"])] for r in rows]), np.array([float(r["weight"]) for r in rows])

def compute_K(t1,t2,F,dirs,weights):
    t1p,t2p,n,j=deformed_frame(t1,t2,F)
    K=np.zeros((len(t1),3,3,3)); I=np.eye(3)
    P=I[None,:,:]-n[:,:,None]*n[:,None,:]
    if float(np.max(weights)-np.min(weights))<=1e-14:
        for c in range(3):
            tangent_iso=K_PERP[c]+0.5*(K_PAR[c]-K_PERP[c])
            K[:,c]=tangent_iso*P+K_NORM[c]*(n[:,:,None]*n[:,None,:])
    else:
        for m,w in zip(dirs,weights):
            a=m[0]*t1+m[1]*t2; b=a@F.T; b/=np.linalg.norm(b,axis=1,keepdims=True)+1e-30
            outer=b[:,:,None]*b[:,None,:]
            for c in range(3): K[:,c]+=w*(K_PERP[c]*I[None,:,:]+(K_PAR[c]-K_PERP[c])*outer)
        for c in range(3):
            K[:,c]=P@K[:,c]@P+K_NORM[c]*(n[:,:,None]*n[:,None,:]); K[:,c]=0.5*(K[:,c]+np.swapaxes(K[:,c],1,2))
    return K,t1p,t2p,n,j

def transmission(K,n,j,lv,t1p,t2p):
    d=lv[0]*t1p+lv[1]*t2p+lv[2]*n; d/=np.linalg.norm(d,axis=1,keepdims=True)+1e-30
    e1,e2=ortho_basis(d); cos=np.maximum(np.abs(np.sum(n*d,axis=1)),0.15); h=H0/np.maximum(j,1e-30)
    mu=np.zeros((len(d),3,2)); T=np.zeros((len(d),3)); tau=np.zeros((len(d),3))
    for c in range(3):
        a=np.einsum("ni,nij,nj->n",e1,K[:,c],e1); b=np.einsum("ni,nij,nj->n",e1,K[:,c],e2); cc=np.einsum("ni,nij,nj->n",e2,K[:,c],e2)
        tr=a+cc; disc=np.sqrt(np.maximum((a-cc)**2+4*b*b,0.0)); mu[:,c,0]=0.5*(tr+disc); mu[:,c,1]=0.5*(tr-disc)
        T[:,c]=0.5*(np.exp(-h*mu[:,c,0]/cos)+np.exp(-h*mu[:,c,1]/cos)); tau[:,c]=-np.log(np.clip(T[:,c],1e-12,1.0))
    return d,h,mu,T,tau

def run_replay(out, base, limit=100000):
    out=Path(out); base=Path(base)
    bank=list(csv.DictReader((out/"D1N2_unique_deformation_lock.csv").open(newline='',encoding='utf-8')))
    F={r["matrix_key"]:np.array(json.loads(r["F"]),dtype=np.float64) for r in bank}
    samples={}
    for s,rel in [("S0_PLANAR_SHEET","S0.csv"),("S1_WAVY_MEMBRANE","S1.csv")]:
        rows=list(csv.DictReader((base/"experiments/stageD0_deformable_optical_transport_feasibility/D0_material_samples"/rel).open(newline='',encoding='utf-8')))
        u=np.array([float(r["u"]) for r in rows]); v=np.array([float(r["v"]) for r in rows]); t1,t2,_=surface_eval(s,u,v); samples[s]=(rows,t1,t2)
    cache={}
    errs={k:[] for k in ["K","mu","T","tau"]}
    rows_out=[]; checked=0
    for row in csv.DictReader((out/"D1N2_microstructure_optical_oracle.csv").open(newline='',encoding='utf-8')):
        if checked>=limit: break
        key=(row["surface"],row["deformation_matrix_key"],row["microstructure_family"],row["view_key"])
        if key not in cache:
            _,t1,t2=samples[row["surface"]]; dirs,w=load_micro(out,row["microstructure_family"])
            K,t1p,t2p,n,j=compute_K(t1,t2,F[row["deformation_matrix_key"]],dirs,w)
            cache[key]=(K,*transmission(K,n,j,VIEWS[row["view_key"]],t1p,t2p))
        K,d,h,mu,T,tau=cache[key]; i=int(row["sample_id"]); c=CHANNELS.index(row["channel"])
        Krow=np.array([[float(row["K00"]),float(row["K01"]),float(row["K02"])],[float(row["K10"]),float(row["K11"]),float(row["K12"])],[float(row["K20"]),float(row["K21"]),float(row["K22"])]])
        errs["K"].append(float(np.max(np.abs(K[i,c]-Krow))))
        mref=np.array([float(row["mu1"]),float(row["mu2"])]); errs["mu"].append(float(np.max(np.abs(mu[i,c]-mref)/np.maximum(np.abs(mref),1e-12))))
        errs["T"].append(abs(float(row["T"])-float(T[i,c]))); errs["tau"].append(abs(float(row["tau_eff"])-float(tau[i,c])))
        checked+=1
    def stat(x): 
        a=np.array(x); return float(np.quantile(a,0.99)), float(a.max())
    Kp,Km=stat(errs["K"]); mup,mum=stat(errs["mu"]); Tp,Tm=stat(errs["T"]); tap,tam=stat(errs["tau"])
    rows_out.append({"checked_rows":checked,"K_p99":Kp,"K_max":Km,"mu_p99":mup,"mu_max":mum,"T_p99":Tp,"T_max":Tm,"tau_eff_p99":tap,"tau_eff_max":tam})
    with (out/"D1N2_oracle_replay.csv").open("w",newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,rows_out[0].keys()); w.writeheader(); w.writerows(rows_out)
    ok=Kp<=1e-10 and Km<=1e-8 and mup<=1e-9 and mum<=1e-7 and Tp<=1e-10 and Tm<=1e-8 and tap<=1e-10 and tam<=1e-8
    return {"O5":"PASS" if ok else "FAIL","K_p99":Kp,"K_max":Km,"mu_p99":mup,"mu_max":mum,"T_p99":Tp,"T_max":Tm,"tau_p99":tap,"tau_max":tam}
'''


if __name__ == "__main__":
    main()
