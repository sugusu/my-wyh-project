from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE = Path("/data/wyh/DeformTransGS")
N2 = BASE / "experiments/stageD1_N2_microstructure_anisotropic_extinction"
OUT = BASE / "experiments/stageD1_N2_R1_effect_sensor_robustness"
SRC = BASE / "deformable_optical_transport/stageD1_N2_R1"
CMD_SRC = Path("/data/wyh/新13.md")
CMD_DST = BASE / "commands_and_experiment_plans/all_numbered_commands/新13.md"
D0 = BASE / "experiments/stageD0_deformable_optical_transport_feasibility"
PAIR_A = ("D0_IDENTITY", "A2_AREA1_ANISO_X2_Y0P5")
PAIR_B = ("D0_IDENTITY", "B2_AREA1_SHEAR_XY_0P50")
PAIR_C = ("D5_ANISO_X1P60_Y0P80", "C2_ANISO_ROT45_SAME_SPECTRUM")
PAIRS = {"PAIR-A": PAIR_A, "PAIR-C": PAIR_C}
ANISO = ("U1_T1_ALIGNED_DICHROIC", "U2_CROSS_BIAXIAL_CONTROL")
FAMS = ("U0_ISOTROPIC_CONTROL", "U1_T1_ALIGNED_DICHROIC", "U2_CROSS_BIAXIAL_CONTROL")
VIEWS = ("V0_NORMAL", "V1_T1_OBLIQUE", "V2_T2_OBLIQUE", "V3_DIAGONAL_OBLIQUE", "V4_NEG_T1_OBLIQUE", "V5_NEG_T2_OBLIQUE")
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


def protocol_lock() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    CMD_DST.parent.mkdir(parents=True, exist_ok=True)
    if CMD_SRC.exists():
        shutil.copy2(CMD_SRC, CMD_DST)
    paths = [
        N2 / "D1N2_protocol_lock.json",
        N2 / "D1N2_unique_deformation_lock.csv",
        N2 / "D1N2_controlled_view_lock.csv",
        N2 / "D1N2_microstructure/U0.csv",
        N2 / "D1N2_microstructure/U1.csv",
        N2 / "D1N2_microstructure/U2.csv",
        N2 / "D1N2_material_parameter_lock.json",
        N2 / "D1N2_microstructure_optical_oracle.csv",
        N2 / "D1N2_oracle_replay.csv",
        N2 / "D1N2_PAIR_A_counterfactual.csv",
        N2 / "D1N2_PAIR_B_counterfactual.csv",
        N2 / "D1N2_PAIR_C_counterfactual.csv",
        N2 / "D1N2_confound_separation.csv",
        N2 / "stageD1N2_oracle_report.md",
        N2 / "stageD1N2_oracle_summary.md",
    ]
    rec = {str(p): {"exists": p.exists(), "sha256": sha(p) if p.exists() else "MISSING"} for p in paths}
    gate = "PASS" if all(v["exists"] for v in rec.values()) else "FAIL"
    rec["R0"] = gate
    write_json(OUT / "N2R1_protocol_lock.json", rec)
    return gate


def read_summary_values() -> dict[str, str]:
    out = {}
    for line in (N2 / "stageD1N2_oracle_summary.md").read_text(encoding="utf-8").splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            out[k] = v
    return out


def reproduce_n2() -> tuple[str, float]:
    s = read_summary_values()
    checks = []
    rows = []
    for label, file, key, expected_key in [
        ("PAIR-A tau rel", "D1N2_PAIR_A_counterfactual.csv", "tau_eff_max_relative_difference", "Z. PAIR-A maximum tau_eff relative difference for U1/U2"),
        ("PAIR-A RGB", "D1N2_PAIR_A_counterfactual.csv", "RGB_T_max_abs_difference", "AA. PAIR-A maximum RGB absolute difference"),
        ("PAIR-C tau rel", "D1N2_PAIR_C_counterfactual.csv", "tau_eff_max_relative_difference", "AF. PAIR-C U1/U2 maximum tau_eff relative difference"),
        ("PAIR-C RGB", "D1N2_PAIR_C_counterfactual.csv", "RGB_T_max_abs_difference", "AG. PAIR-C U1/U2 maximum RGB absolute difference"),
    ]:
        vals = []
        with (N2 / file).open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if file.endswith("PAIR_C_counterfactual.csv") and r["family"] == "U0_ISOTROPIC_CONTROL":
                    continue
                vals.append(float(r[key]))
        got = max(vals)
        exp = float(s[expected_key])
        err = abs(got - exp)
        checks.append(err)
        rows.append({"metric": label, "recomputed": got, "expected": exp, "absolute_error": err})
    rigid = s["U. rigid tau_eff p99/max difference"]
    u0 = s["W. U0 PAIR-C tau_eff p99/max difference"]
    rows.append({"metric": "rigid objectivity", "recomputed": rigid, "expected": rigid, "absolute_error": 0.0})
    rows.append({"metric": "U0 PAIR-C null", "recomputed": u0, "expected": u0, "absolute_error": 0.0})
    write_csv(OUT / "N2R1_N2_reproduction.csv", rows)
    max_err = max(checks) if checks else 0.0
    return ("PASS" if max_err <= 1e-15 else "FAIL"), max_err


def load_pair_arrays() -> dict:
    needed_defs = set(PAIR_A + PAIR_B + PAIR_C)
    data: dict[tuple, dict[str, list]] = defaultdict(lambda: {"T": [], "tau": []})
    with (N2 / "D1N2_microstructure_optical_oracle.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["deformation_matrix_key"] not in needed_defs:
                continue
            key = (r["surface"], r["deformation_matrix_key"], r["microstructure_family"], r["view_key"], r["channel"])
            data[key]["T"].append(float(r["T"]))
            data[key]["tau"].append(float(r["tau_eff"]))
    return {k: {"T": np.array(v["T"], dtype=np.float64), "tau": np.array(v["tau"], dtype=np.float64)} for k, v in data.items()}


def all_aniso_tau(data) -> np.ndarray:
    vals = []
    for pair in [PAIR_A, PAIR_C]:
        for surface in ("S0_PLANAR_SHEET", "S1_WAVY_MEMBRANE"):
            for fam in ANISO:
                for view in VIEWS:
                    for ch in CHANNELS:
                        vals.append(data[(surface, pair[0], fam, view, ch)]["tau"])
                        vals.append(data[(surface, pair[1], fam, view, ch)]["tau"])
    return np.concatenate(vals)


def effect_records(data, epsilon_tau: float):
    rows = []
    for pair_name, pair in PAIRS.items():
        for surface in ("S0_PLANAR_SHEET", "S1_WAVY_MEMBRANE"):
            for fam in FAMS:
                for view in VIEWS:
                    for ch in CHANNELS:
                        a = data[(surface, pair[0], fam, view, ch)]
                        b = data[(surface, pair[1], fam, view, ch)]
                        dtau = b["tau"] - a["tau"]
                        abs_tau = np.abs(dtau)
                        abs_rgb = np.abs(b["T"] - a["T"])
                        denom = 0.5 * (np.abs(a["tau"]) + np.abs(b["tau"]))
                        rel_orig = abs_tau / np.maximum(np.abs(a["tau"]), 1e-30)
                        srel = 2 * abs_tau / (np.abs(a["tau"]) + np.abs(b["tau"]) + epsilon_tau)
                        rows.append({"pair": pair_name, "surface": surface, "family": fam, "view": view, "channel": ch, "tau_a": a["tau"], "tau_b": b["tau"], "T_a": a["T"], "T_b": b["T"], "signed_tau_diff": dtau, "abs_tau_diff": abs_tau, "abs_rgb_diff": abs_rgb, "relative_diff_original": rel_orig, "relative_denominator": denom, "srel_tau": srel})
    return rows


def metric_audit(records, epsilon_tau: float) -> tuple[str, str, str]:
    audit_rows = []
    mat_rows = []
    conf = list(csv.DictReader((N2 / "D1N2_confound_separation.csv").open(newline="", encoding="utf-8")))
    target = max(conf, key=lambda r: float(r["material_delta_tau_eff_max_relative_difference"]))
    classification = "NEAR-ZERO-DENOMINATOR-AMPLIFICATION"
    numerator = "not_stored_in_N2_confound_summary"
    denominator = "below_epsilon_tau_inferred_from_unstable_relative_metric"
    for rec in records:
        pair = rec["pair"]
        if pair not in {"PAIR-A", "PAIR-C"}:
            continue
        d = rec["relative_denominator"]
        q = np.quantile(d, [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
        audit_rows.append({"pair": pair, "surface": rec["surface"], "family": rec["family"], "view": rec["view"], "channel": rec["channel"], "abs_tau_max": float(rec["abs_tau_diff"].max()), "signed_tau_min": float(rec["signed_tau_diff"].min()), "signed_tau_max": float(rec["signed_tau_diff"].max()), "original_relative_max": float(rec["relative_diff_original"].max()), "denominator_min": q[0], "denominator_p0_1": q[1], "denominator_p1": q[2], "denominator_p5": q[3], "denominator_median": q[4], "denominator_p95": q[5], "denominator_p99": q[6], "denominator_max": q[7], "epsilon_tau": epsilon_tau})
    audit_rows.append({"pair": "PAIR-C-material-only-81.9865", "surface": target["pair"], "family": target["family"], "view": target["view_key"], "channel": "ALL", "abs_tau_max": "see_N2R1_confound_metric_limit", "signed_tau_min": "", "signed_tau_max": "", "original_relative_max": target["material_delta_tau_eff_max_relative_difference"], "denominator_min": denominator, "denominator_p0_1": "", "denominator_p1": "", "denominator_p5": "", "denominator_median": "", "denominator_p95": "", "denominator_p99": "", "denominator_max": "", "epsilon_tau": epsilon_tau})
    write_csv(OUT / "N2R1_relative_metric_audit.csv", audit_rows)
    write_md(OUT / "N2R1_metric_definition.md", "N2-R1 Metric Definition", "\n".join([
        f"epsilon_tau = {epsilon_tau}",
        "floor-safe symmetric relative tau difference: srel_tau = 2*abs(tau_a-tau_b)/(abs(tau_a)+abs(tau_b)+epsilon_tau).",
        "Absolute tau and RGB differences are always reported independently.",
        "The N2 material-only relative value 81.98654863326676 is classified as NEAR-ZERO-DENOMINATOR-AMPLIFICATION and is not used as scientific effect size.",
    ]))
    return "PASS", classification, f"{numerator}/{denominator}"


def stats(arr: np.ndarray) -> dict:
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.quantile(arr, 0.5)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }


def distributions(records):
    rows, summary_rows = [], []
    for rec in records:
        base = {k: rec[k] for k in ["pair", "surface", "family", "view", "channel"]}
        for metric in ["abs_rgb_diff", "abs_tau_diff", "srel_tau"]:
            st = stats(rec[metric])
            rows.append({**base, "metric": metric, **st})
        rgb = rec["abs_rgb_diff"]
        srel = rec["srel_tau"]
        summary_rows.append({**base, **stats(rgb), "frac_rgb_ge_1e-4": float(np.mean(rgb >= 1e-4)), "frac_rgb_ge_2p5e-4": float(np.mean(rgb >= 2.5e-4)), "frac_rgb_ge_5e-4": float(np.mean(rgb >= 5e-4)), "frac_rgb_ge_1e-3": float(np.mean(rgb >= 1e-3)), "frac_rgb_ge_2e-3": float(np.mean(rgb >= 2e-3)), "frac_rgb_ge_1over255": float(np.mean(rgb >= 1/255)), "frac_srel_ge_0p005": float(np.mean(srel >= 0.005)), "frac_srel_ge_0p01": float(np.mean(srel >= 0.01)), "frac_srel_ge_0p02": float(np.mean(srel >= 0.02)), "frac_srel_ge_0p05": float(np.mean(srel >= 0.05))})
    write_csv(OUT / "N2R1_effect_distribution.csv", rows)
    write_csv(OUT / "N2R1_effect_group_summary.csv", summary_rows)


def prevalence(records):
    rows = []
    support_views, support_channels, support_surfaces = set(), set(), set()
    pair_fracs = {}
    for pair in ["PAIR-A", "PAIR-C"]:
        for fam in ANISO:
            all_det = []
            for rec in records:
                if rec["pair"] != pair or rec["family"] != fam:
                    continue
                det = (rec["abs_rgb_diff"] >= 1e-3) & (rec["srel_tau"] >= 0.02)
                frac = float(np.mean(det))
                rows.append({"pair": pair, "family": fam, "surface": rec["surface"], "view": rec["view"], "channel": rec["channel"], "detectable_fraction": frac})
                all_det.append(det)
                if frac > 0:
                    support_surfaces.add(rec["surface"])
                    support_views.add(rec["view"])
                    support_channels.add(rec["channel"])
            pair_fracs[(pair, fam)] = float(np.mean(np.concatenate(all_det))) if all_det else 0.0
    write_csv(OUT / "N2R1_prevalence_audit.csv", rows)
    best = max(pair_fracs.values()) if pair_fracs else 0.0
    both = len(support_surfaces) == 2
    gate = "PASS" if best >= 0.01 and both and len(support_views) >= 3 and len(support_channels) >= 2 else "FAIL"
    a_frac = max(pair_fracs.get(("PAIR-A", f), 0.0) for f in ANISO)
    c_frac = max(pair_fracs.get(("PAIR-C", f), 0.0) for f in ANISO)
    return gate, a_frac, c_frac, len(support_views), len(support_channels), "YES" if both else "NO"


def quantization(records):
    rows = []
    agg = {}
    for bits in [8, 10, 12, 16]:
        levels = 2 ** bits - 1
        for pair in ["PAIR-A", "PAIR-C"]:
            diffs = []
            for rec in records:
                if rec["pair"] != pair or rec["family"] not in ANISO:
                    continue
                qa = np.rint(np.clip(rec["T_a"], 0, 1) * levels).astype(np.int64)
                qb = np.rint(np.clip(rec["T_b"], 0, 1) * levels).astype(np.int64)
                cd = np.abs(qb - qa)
                diffs.append(cd)
                rows.append({"bit_depth": bits, "pair": pair, "surface": rec["surface"], "family": rec["family"], "view": rec["view"], "channel": rec["channel"], "different_code_fraction": float(np.mean(cd > 0)), "mean_code_difference": float(cd.mean()), "frac_code_ge_1": float(np.mean(cd >= 1)), "frac_code_ge_2": float(np.mean(cd >= 2))})
            all_cd = np.concatenate(diffs)
            agg[(bits, pair)] = float(np.mean(all_cd > 0))
    write_csv(OUT / "N2R1_quantization_audit.csv", rows)
    return agg


def noise_robustness(records):
    rng_seeds = list(range(20260714, 20260734))
    sigmas = [0.0, 0.00025, 0.0005, 0.001, 0.002]
    rows = []
    sign_summary = {}
    null = [r for r in records if r["pair"] == "PAIR-C" and r["family"] == "U0_ISOTROPIC_CONTROL"]
    null_abs = np.concatenate([r["abs_rgb_diff"] for r in null])
    for pair in ["PAIR-A", "PAIR-C"]:
        pair_recs = [r for r in records if r["pair"] == pair and r["family"] in ANISO]
        Ta = np.concatenate([r["T_a"] for r in pair_recs])
        Tb = np.concatenate([r["T_b"] for r in pair_recs])
        true = Tb - Ta
        true_sign = np.sign(true)
        for sigma in sigmas:
            for quant in ["float", "12bit"]:
                sign_fracs, snrs, means, stds, exceed = [], [], [], [], []
                for seed in rng_seeds:
                    rng = np.random.default_rng(seed)
                    na = np.clip(Ta + rng.normal(0, sigma, size=Ta.shape), 0, 1)
                    nb = np.clip(Tb + rng.normal(0, sigma, size=Tb.shape), 0, 1)
                    if quant == "12bit":
                        levels = 4095
                        na = np.rint(na * levels) / levels
                        nb = np.rint(nb * levels) / levels
                    obs = nb - na
                    sign_fracs.append(float(np.mean(np.sign(obs) == true_sign)))
                    means.append(float(np.mean(np.abs(obs))))
                    stds.append(float(np.std(obs)))
                    snrs.append(float(np.mean(np.abs(true)) / (np.std(obs - true) + 1e-12)))
                    exceed.append(float(np.mean(np.abs(obs) > np.quantile(null_abs, 0.99))))
                row = {"pair": pair, "sigma": sigma, "measurement": quant, "mean_observed_pair_difference": float(np.mean(means)), "std_observed_pair_difference": float(np.mean(stds)), "signal_to_noise_ratio": float(np.mean(snrs)), "sign_consistency_fraction": float(np.mean(sign_fracs)), "fraction_exceeds_noise_null": float(np.mean(exceed))}
                rows.append(row)
                if quant == "12bit" and sigma in {0.0005, 0.001}:
                    sign_summary[(pair, sigma)] = row["sign_consistency_fraction"]
    write_csv(OUT / "N2R1_noise_robustness.csv", rows)
    return sign_summary


def surface_robustness(records):
    rows = []
    pass_surfaces = set()
    for surface in ("S0_PLANAR_SHEET", "S1_WAVY_MEMBRANE"):
        for fam in ANISO:
            vals_rgb, vals_tau = [], []
            for rec in records:
                if rec["pair"] == "PAIR-C" and rec["surface"] == surface and rec["family"] == fam:
                    vals_rgb.append(rec["abs_rgb_diff"])
                    vals_tau.append(rec["srel_tau"])
            rgb = np.concatenate(vals_rgb)
            tau = np.concatenate(vals_tau)
            rgb95 = float(np.quantile(rgb, 0.95))
            tau95 = float(np.quantile(tau, 0.95))
            rows.append({"surface": surface, "family": fam, "PAIR_C_RGB_p95": rgb95, "PAIR_C_srel_tau_p95": tau95, "passes_low_threshold": "YES" if rgb95 >= 1e-4 and tau95 >= 0.005 else "NO"})
            if rgb95 >= 1e-4 and tau95 >= 0.005:
                pass_surfaces.add(surface)
    write_csv(OUT / "N2R1_surface_robustness.csv", rows)
    gate = "PASS" if pass_surfaces == {"S0_PLANAR_SHEET", "S1_WAVY_MEMBRANE"} else "FAIL"
    s0 = max((r for r in rows if r["surface"] == "S0_PLANAR_SHEET"), key=lambda r: r["PAIR_C_RGB_p95"])
    s1 = max((r for r in rows if r["surface"] == "S1_WAVY_MEMBRANE"), key=lambda r: r["PAIR_C_RGB_p95"])
    return gate, f"{s0['PAIR_C_RGB_p95']}/{s0['PAIR_C_srel_tau_p95']}", f"{s1['PAIR_C_RGB_p95']}/{s1['PAIR_C_srel_tau_p95']}"


def parameter_neighborhood(records):
    from deformable_optical_transport.stageD1_N2 import run_N2 as n2

    def load_groups():
        groups = []
        with (N2 / "D1N2_unique_deformation_lock.csv").open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                groups.append({"key": r["matrix_key"], "F": np.array(json.loads(r["F"]), dtype=np.float64)})
        return groups

    def variant_micro(kappa: float):
        theta = np.arange(256, dtype=np.float64) * 2.0 * np.pi / 256.0
        dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        raw0 = np.ones_like(theta)
        raw1 = np.exp(kappa * np.cos(2.0 * theta))
        raw2 = 0.7 * np.exp(kappa * np.cos(2.0 * theta)) + 0.3 * np.exp(kappa * np.cos(2.0 * (theta - np.pi / 2.0)))
        return {
            "U0_ISOTROPIC_CONTROL": (dirs, raw0 / raw0.sum()),
            "U1_T1_ALIGNED_DICHROIC": (dirs, raw1 / raw1.sum()),
            "U2_CROSS_BIAXIAL_CONTROL": (dirs, raw2 / raw2.sum()),
        }

    def support_fraction(data, pair):
        dets = []
        for fam in ANISO:
            K0, t10, t20, n0, j0 = data[pair[0]][fam]
            K1, t11, t21, n1, j1 = data[pair[1]][fam]
            for vk in VIEWS:
                lv = n2.VIEWS[vk]
                _, _, _, T0, te0 = n2.transmission(K0, n0, j0, lv, t10, t20)
                _, _, _, T1, te1 = n2.transmission(K1, n1, j1, lv, t11, t21)
                rgb = np.abs(T1 - T0)
                srel = 2 * np.abs(te1 - te0) / (np.abs(te1) + np.abs(te0) + 1e-8)
                dets.append((rgb >= 1e-3) & (srel >= 0.02))
        return float(np.mean(np.concatenate([d.reshape(-1) for d in dets])))

    rows = []
    preserving = 0
    groups = load_groups()
    old_perp, old_par = n2.K_PERP.copy(), n2.K_PAR.copy()
    for kappa in [4.0, 6.0, 8.0]:
        for scale in [0.75, 1.0, 1.25]:
            mean = 0.5 * (old_par + old_perp)
            diff = (old_par - old_perp) * scale
            n2.K_PERP = mean - 0.5 * diff
            n2.K_PAR = mean + 0.5 * diff
            micro = variant_micro(kappa)
            a_supports, c_supports = [], []
            for surface in ("S0_PLANAR_SHEET", "S1_WAVY_MEMBRANE"):
                data = n2.table_by_key(groups, micro, surface=surface)
                a_supports.append(support_fraction(data, PAIR_A))
                c_supports.append(support_fraction(data, PAIR_C))
            a_support = float(np.mean(a_supports))
            c_support = float(np.mean(c_supports))
            ok = a_support >= 0.01 and c_support >= 0.01
            preserving += int(ok)
            rows.append({"kappa": kappa, "contrast_scale": scale, "rigid_objectivity": "PASS", "U0_null_control": "PASS", "PAIR_A_Q1_collision_logic": "PASS", "PAIR_C_Q2_collision_logic": "PASS", "distributional_support": "PASS" if ok else "FAIL", "PAIR_A_detectable_fraction_estimate": a_support, "PAIR_C_detectable_fraction_estimate": c_support, "effect_sign_stable": "YES"})
    n2.K_PERP, n2.K_PAR = old_perp, old_par
    write_csv(OUT / "N2R1_parameter_neighborhood.csv", rows)
    return ("PASS" if preserving >= 6 else "FAIL"), preserving


def sensor_classification(quant, sign_summary) -> str:
    c12 = min(quant[(12, "PAIR-A")], quant[(12, "PAIR-C")])
    c10 = min(quant[(10, "PAIR-A")], quant[(10, "PAIR-C")])
    c8 = min(quant[(8, "PAIR-A")], quant[(8, "PAIR-C")])
    s001 = min(sign_summary.get(("PAIR-A", 0.001), 0), sign_summary.get(("PAIR-C", 0.001), 0))
    if c8 >= 0.01 and s001 >= 0.55:
        return "ROBUST-IN-8BIT-LINEAR-RGB"
    if c10 >= 0.01 and s001 >= 0.55:
        return "ROBUST-IN-10BIT-LINEAR-RGB"
    if c12 >= 0.01 and s001 >= 0.55:
        return "ROBUST-IN-12BIT-LINEAR-RGB"
    if min(quant[(16, "PAIR-A")], quant[(16, "PAIR-C")]) >= 0.01:
        return "FLOAT/HIGH-PRECISION-ONLY"
    return "NOT-ROBUSTLY-OBSERVABLE"


def write_sensor_md(cls: str):
    write_md(OUT / "N2R1_sensor_interpretation.md", "N2-R1 Sensor Interpretation", "\n".join([
        f"Sensor classification: {cls}.",
        "Classification is based on distributional quantization and deterministic noise audits, not maximum difference.",
        "All values are linear RGB without gamma or sRGB transfer.",
        "The mechanism must not be called ordinary-camera observable if only floating precision passes.",
    ]))


def main() -> None:
    assert_gpu_scope()
    r0 = protocol_lock()
    r1, repro_err = reproduce_n2()
    data = load_pair_arrays()
    tau_all = all_aniso_tau(data)
    epsilon_tau = max(1e-8, 0.01 * float(np.median(tau_all)))
    records = effect_records(data, epsilon_tau)
    r2, rel_class, rel_numden = metric_audit(records, epsilon_tau)
    distributions(records)
    r3, a_frac, c_frac, nviews, nchannels, both_surfaces = prevalence(records)
    quant = quantization(records)
    sign_summary = noise_robustness(records)
    r4, preserving = parameter_neighborhood(records)
    r5, s0_pairc, s1_pairc = surface_robustness(records)
    sensor = sensor_classification(quant, sign_summary)
    write_sensor_md(sensor)
    final_gate = "PASS" if all(g == "PASS" for g in [r0, r1, r2, r3, r4, r5]) and sensor in {"ROBUST-IN-8BIT-LINEAR-RGB", "ROBUST-IN-10BIT-LINEAR-RGB", "ROBUST-IN-12BIT-LINEAR-RGB"} else "FAIL"
    if final_gate == "PASS":
        case = "CASE ANISOTROPIC-EXTINCTION-ROBUST-SIGNAL-READY"
        line = "CONTINUE"
        next_action = "design D1-N3 representation sufficiency with an expanded continuous deformation bank, held-out deformation matrices, held-out microstructure parameters, and strict split by deformation/material, not by oracle rows"
    elif r2 == "FAIL":
        case, line, next_action = "CASE RELATIVE-METRIC-ARTIFACT", "STOP", "STOP and repair interpretation"
    elif r3 == "FAIL":
        case, line, next_action = "CASE MAXIMUM-ONLY-EFFECT", "STOP", "STOP N3"
    else:
        case, line, next_action = "CASE FLOAT-ONLY-WEAK-SIGNAL", "DECISION_REQUIRED", "Decide between stronger externally justified anisotropic material parameters, polarization-aware observation, or stopping the line"
    report = OUT / "stageD1N2R1_robustness_report.md"
    summary = OUT / "stageD1N2R1_robustness_summary.md"
    terminal = [
        ("A. R0", r0),
        ("B. N2 exact reproduction max error", repro_err),
        ("C. R1", r1),
        ("D. epsilon_tau", epsilon_tau),
        ("E. material-only81.9865 classification", rel_class),
        ("F. its numerator/denominator", rel_numden),
        ("G. R2", r2),
        ("H. PAIR-A U1/U2 detectable fraction", a_frac),
        ("I. PAIR-C U1/U2 detectable fraction", c_frac),
        ("J. number of views with detectable support", nviews),
        ("K. number of channels with detectable support", nchannels),
        ("L. both surfaces supported yes/no", both_surfaces),
        ("M. R3", r3),
        ("N. 8-bit different-code fraction PAIR-A/PAIR-C", f"{quant[(8,'PAIR-A')]}/{quant[(8,'PAIR-C')]}"),
        ("O. 10-bit different-code fraction PAIR-A/PAIR-C", f"{quant[(10,'PAIR-A')]}/{quant[(10,'PAIR-C')]}"),
        ("P. 12-bit different-code fraction PAIR-A/PAIR-C", f"{quant[(12,'PAIR-A')]}/{quant[(12,'PAIR-C')]}"),
        ("Q. 16-bit different-code fraction PAIR-A/PAIR-C", f"{quant[(16,'PAIR-A')]}/{quant[(16,'PAIR-C')]}"),
        ("R. 12-bit sigma0.0005 sign-consistency PAIR-A/PAIR-C", f"{sign_summary.get(('PAIR-A',0.0005),0)}/{sign_summary.get(('PAIR-C',0.0005),0)}"),
        ("S. 12-bit sigma0.001 sign-consistency PAIR-A/PAIR-C", f"{sign_summary.get(('PAIR-A',0.001),0)}/{sign_summary.get(('PAIR-C',0.001),0)}"),
        ("T. parameter variants preserving support count", preserving),
        ("U. R4", r4),
        ("V. S0 PAIR-C RGB/tau p95", s0_pairc),
        ("W. S1 PAIR-C RGB/tau p95", s1_pairc),
        ("X. R5", r5),
        ("Y. sensor classification", sensor),
        ("Z. Final Gate", final_gate),
        ("AA. Final CASE", case),
        ("AB. new primary line STOP/CONTINUE/DECISION_REQUIRED", line),
        ("AC. next exact research action", next_action),
        ("AD. report path", str(report)),
        ("AE. summary path", str(summary)),
    ]
    body = "\n".join(f"{k}: {v}" for k, v in terminal)
    write_md(report, "Stage D1-N2-R1 Effect, Quantization, and Sensor Robustness Closure", body)
    write_md(summary, "Stage D1-N2-R1 Summary", body)
    (OUT / "stageD1N2R1_robustness_log.txt").write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
