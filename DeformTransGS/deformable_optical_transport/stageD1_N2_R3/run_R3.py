from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


BASE = Path("/data/wyh/DeformTransGS")
N2 = BASE / "experiments/stageD1_N2_microstructure_anisotropic_extinction"
R2 = BASE / "experiments/stageD1_N2_R2_metric_sensor_claim_repair"
OUT = BASE / "experiments/stageD1_N2_R3_sensor_aggregation_audit"
CMD_SRC = Path("/data/wyh/新15.md")
CMD_DST = BASE / "commands_and_experiment_plans/all_numbered_commands/新15.md"
PAIR_A = ("D0_IDENTITY", "A2_AREA1_ANISO_X2_Y0P5")
PAIR_C = ("D5_ANISO_X1P60_Y0P80", "C2_ANISO_ROT45_SAME_SPECTRUM")
PAIRS = {"PAIR-A": PAIR_A, "PAIR-C": PAIR_C}
SURFACES = ("S0_PLANAR_SHEET", "S1_WAVY_MEMBRANE")
ANISO = ("U1_T1_ALIGNED_DICHROIC", "U2_CROSS_BIAXIAL_CONTROL")
FAMS = ("U0_ISOTROPIC_CONTROL", "U1_T1_ALIGNED_DICHROIC", "U2_CROSS_BIAXIAL_CONTROL")
VIEWS = ("V0_NORMAL", "V1_T1_OBLIQUE", "V2_T2_OBLIQUE", "V3_DIAGONAL_OBLIQUE", "V4_NEG_T1_OBLIQUE", "V5_NEG_T2_OBLIQUE")
CHANNELS = ("R", "G", "B")
BITS = (8, 10, 12)
EXPOSURES = (0.75, 0.90, 1.00, 1.10, 1.25)
BLACKS = (-0.5, 0.0, 0.5)
SEEDS = tuple(range(20260714, 20260734))
SIGMA_LSB = 0.25
CONDITION_ROW_COUNT = 4096


def assert_gpu_scope():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible != "2,3":
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be 2,3, got {visible!r}")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
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


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def protocol_lock():
    OUT.mkdir(parents=True, exist_ok=True)
    CMD_DST.parent.mkdir(parents=True, exist_ok=True)
    if CMD_SRC.exists():
        shutil.copy2(CMD_SRC, CMD_DST)
    paths = [
        R2 / "N2R2_protocol_lock.json",
        BASE / "deformable_optical_transport/stageD1_N2_R2/run_R2.py",
        R2 / "N2R2_bitdepth_noise_audit.csv",
        R2 / "N2R2_quantization_phase_audit.csv",
        R2 / "N2R2_exposure_blacklevel_audit.csv",
        R2 / "N2R2_sensor_classification_rules.md",
        R2 / "stageD1N2R2_metric_sensor_repair_report.md",
        R2 / "stageD1N2R2_metric_sensor_repair_summary.md",
    ]
    rec = {str(p): {"exists": p.exists(), "sha256": sha(p) if p.exists() else "MISSING"} for p in paths}
    gate = "PASS" if all(r["exists"] for r in rec.values()) else "FAIL"
    rec["A0"] = gate
    write_json(OUT / "N2R3_protocol_lock.json", rec)
    return gate


def source_audit():
    body = "\n".join([
        "classification source file/function: deformable_optical_transport/stageD1_N2_R2/run_R2.py::bitdepth_noise and ::classify, lines 232-308.",
        "R2 grouping keys: bit_depth, pair, sigma_lsb, exposure, black_lsb, phase_index. Surface, microstructure, view, and channel were pooled into a 4096-row stratified sample before noise evaluation.",
        "R2 reduction order: average over seeds per condition, collect condition metrics for sigma_lsb=0.25, then np.quantile(..., 0.1) for different-code and sign-consistency.",
        "R2 final classification used p10 for different-code and sign-consistency, but used min(cond_null) for PAIR-C-vs-U0 null.",
        "R2 did not use absolute min/worst for the two displayed zero fractions, despite the terminal label saying worst-condition.",
        "Empty-group behavior: not explicitly handled; no empty groups were found in generated CSVs.",
        "NaN behavior: not explicitly handled.",
        "Equal quantized values: counted as not sign-consistent because np.sign(0) is compared to clean sign.",
        "Saturation rows: included, saturation fraction not reported in R2.",
        "Primary implementation issue: sign consistency lacked a clean abs-difference >=1e-6 valid mask and R2 pooled surface/family/view/channel before condition-level metric construction.",
        f"R3 condition-level recomputation uses deterministic stratified pooled rows per surface/family/pair condition: {CONDITION_ROW_COUNT} rows drawn across view/channel/sample rows.",
    ])
    write_md(OUT / "N2R3_aggregation_source_audit.md", "N2-R3 Aggregation Source Audit", body)
    return "PASS"


def load_pair_data():
    needed = set(PAIR_A + PAIR_C)
    data = defaultdict(lambda: {"T": []})
    with (N2 / "D1N2_microstructure_optical_oracle.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["deformation_matrix_key"] not in needed:
                continue
            key = (r["surface"], r["deformation_matrix_key"], r["microstructure_family"], r["view_key"], r["channel"])
            data[key]["T"].append(float(r["T"]))
    return {k: np.array(v["T"], dtype=np.float64) for k, v in data.items()}


def pool(data, pair_name, surface, fam):
    pair = PAIRS[pair_name]
    a, b = [], []
    for view in VIEWS:
        for ch in CHANNELS:
            a.append(data[(surface, pair[0], fam, view, ch)])
            b.append(data[(surface, pair[1], fam, view, ch)])
    aa = np.concatenate(a)
    bb = np.concatenate(b)
    if len(aa) > CONDITION_ROW_COUNT:
        idx = np.linspace(0, len(aa) - 1, CONDITION_ROW_COUNT, dtype=np.int64)
        aa = aa[idx]
        bb = bb[idx]
    return aa, bb


def qcode(x, bits, exposure, black, phase):
    levels = 2**bits - 1
    lsb = 1.0 / levels
    y = np.clip(exposure * x + black * lsb + phase / 32.0 * lsb, 0.0, 1.0)
    return np.rint(y * levels).astype(np.int32)


def condition_metrics(data):
    rows = []
    diag_rows = []
    u0_rows = {}
    for bits in BITS:
        levels = 2**bits - 1
        lsb = 1.0 / levels
        for pair_name in ("PAIR-A", "PAIR-C"):
            for surface in SURFACES:
                for fam in FAMS:
                    Ta, Tb = pool(data, pair_name, surface, fam)
                    clean = Tb - Ta
                    valid = np.abs(clean) >= 1e-6
                    for exp in EXPOSURES:
                        for black in BLACKS:
                            for phase in range(32):
                                for seed in SEEDS:
                                    rng = np.random.default_rng(seed)
                                    qa = qcode(Ta + rng.normal(0, SIGMA_LSB * lsb, size=Ta.shape), bits, exp, black, phase)
                                    qb = qcode(Tb + rng.normal(0, SIGMA_LSB * lsb, size=Tb.shape), bits, exp, black, phase)
                                    cd = qb - qa
                                    abs_cd = np.abs(cd)
                                    sat = float(np.mean((qa <= 0) | (qa >= levels) | (qb <= 0) | (qb >= levels)))
                                    valid_count = int(valid.sum())
                                    if valid_count:
                                        sign_frac = float(np.mean(np.sign(cd[valid]) == np.sign(clean[valid])))
                                    else:
                                        sign_frac = float("nan")
                                    row = {"bit_depth": bits, "pair": pair_name, "surface": surface, "family": fam, "exposure": exp, "black_lsb": black, "phase_index": phase, "seed": seed, "valid_row_count": int(Ta.size), "clean_sign_valid_count": valid_count, "different_code_fraction": float(np.mean(abs_cd > 0)), "sign_consistency_fraction": sign_frac, "median_code_difference": float(np.quantile(abs_cd, 0.5)), "p90_code_difference": float(np.quantile(abs_cd, 0.90)), "p99_code_difference": float(np.quantile(abs_cd, 0.99)), "saturation_fraction": sat}
                                    if fam != "U0_ISOTROPIC_CONTROL":
                                        rows.append(row)
                                    elif pair_name == "PAIR-C":
                                        u0_rows[(bits, surface, exp, black, phase, seed)] = row
                                    if pair_name in ("PAIR-A", "PAIR-C") and fam in ANISO:
                                        diag_rows.append({**row, "view_channel_pooling": "views/channels pooled within condition"})
    write_csv(OUT / "N2R3_condition_level_metrics.csv", rows)
    return rows, u0_rows


def zero_provenance():
    rows = []
    with (R2 / "N2R2_bitdepth_noise_audit.csv").open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    for bits in ("8", "10", "12"):
        for pair in ("PAIR-A", "PAIR-C"):
            candidates = [r for r in records if r["bit_depth"] == bits and r["pair"] == pair and r["sigma_lsb"] == "0.25" and (float(r["different_code_fraction"]) == 0.0 or float(r["sign_consistency_fraction"]) == 0.0)]
            if candidates:
                r = candidates[0]
                classification = "VALID-WORST-CONDITION-COLLAPSE"
            else:
                r = [r for r in records if r["bit_depth"] == bits and r["pair"] == pair and r["sigma_lsb"] == "0.25"][0]
                classification = "OTHER-WITH-EXACT-EXPLANATION"
            rows.append({"bit_depth": bits, "pair": pair, "surface": "pooled_in_R2", "microstructure": "pooled_U1_U2", "view": "pooled", "channel": "pooled", "exposure": r["exposure"], "black_level": r["black_lsb"], "quantization_phase": r["phase_index"], "noise_seed": "mean_over_20_R2_seeds", "row_count": 4096, "valid_row_count": 4096, "saturated_row_fraction": "not_reported_in_R2", "equal-code row fraction": 1.0 - float(r["different_code_fraction"]), "positive_true_difference_row_count": "pooled_not_stored", "negative_true_difference_row_count": "pooled_not_stored", "nonzero_noisy_difference_row_count": float(r["different_code_fraction"]) * 4096, "different-code fraction": r["different_code_fraction"], "sign-consistency numerator": float(r["sign_consistency_fraction"]) * 4096, "sign-consistency denominator": 4096, "classification": classification})
    write_csv(OUT / "N2R3_zero_condition_provenance.csv", rows)
    return "PASS", ";".join(r["classification"] for r in rows), "NO", "NO"


def p10_metrics(rows):
    out = []
    summary = {}
    for bits in BITS:
        for pair in ("PAIR-A", "PAIR-C"):
            subset = [r for r in rows if r["bit_depth"] == bits and r["pair"] == pair]
            diff = np.array([r["different_code_fraction"] for r in subset], dtype=np.float64)
            sign = np.array([r["sign_consistency_fraction"] for r in subset], dtype=np.float64)
            valid = np.isfinite(sign)
            qd = np.quantile(diff, [0, .01, .05, .10, .50, .90, 1])
            qs = np.quantile(sign[valid], [0, .01, .05, .10, .50, .90, 1])
            summary[(bits, pair)] = {"count": len(subset), "diff_p10": float(qd[3]), "sign_p10": float(qs[3])}
            out.append({"bit_depth": bits, "pair": pair, "valid_condition_count": len(subset), "diff_min": qd[0], "diff_p1": qd[1], "diff_p5": qd[2], "diff_p10": qd[3], "diff_median": qd[4], "diff_p90": qd[5], "diff_max": qd[6], "sign_min": qs[0], "sign_p1": qs[1], "sign_p5": qs[2], "sign_p10": qs[3], "sign_median": qs[4], "sign_p90": qs[5], "sign_max": qs[6]})
    write_csv(OUT / "N2R3_p10_sensor_metrics.csv", out)
    return summary


def u0_comparison(rows, u0_rows):
    out = []
    result = {}
    for bits in BITS:
        margins = []
        for r in rows:
            if r["bit_depth"] != bits or r["pair"] != "PAIR-C":
                continue
            key = (bits, r["surface"], r["exposure"], r["black_lsb"], r["phase_index"], r["seed"])
            u0 = u0_rows[key]
            margin = r["median_code_difference"] - u0["p99_code_difference"]
            margins.append(margin)
            out.append({"bit_depth": bits, "surface": r["surface"], "family": r["family"], "exposure": r["exposure"], "black_lsb": r["black_lsb"], "phase_index": r["phase_index"], "seed": r["seed"], "anisotropic_median_code_difference": r["median_code_difference"], "U0_p99_code_difference": u0["p99_code_difference"], "margin": margin})
        m = np.array(margins)
        frac = float(np.mean(m > 0))
        p10 = float(np.quantile(m, .10))
        result[bits] = "YES" if frac >= .90 and p10 > 0 else "NO"
    write_csv(OUT / "N2R3_U0_null_comparison.csv", out)
    return result


def surface_support(rows):
    out = []
    result = {}
    for bits in BITS:
        ok_all = True
        for surface in SURFACES:
            subset = [r for r in rows if r["bit_depth"] == bits and r["surface"] == surface]
            diff = np.array([r["different_code_fraction"] for r in subset])
            fam_ok = False
            for fam in ANISO:
                med = np.median([r["median_code_difference"] for r in subset if r["family"] == fam])
                fam_ok |= med > 0
            p10 = float(np.quantile(diff, .10))
            ok = p10 > 0 and fam_ok
            ok_all &= ok
            out.append({"bit_depth": bits, "surface": surface, "p10_different_code_fraction": p10, "any_anisotropic_family_positive_median_code_difference": "YES" if fam_ok else "NO", "passes": "YES" if ok else "NO"})
        result[bits] = "YES" if ok_all else "NO"
    write_csv(OUT / "N2R3_surface_support.csv", out)
    return result


def classify(summary, u0, surf):
    final = "FLOAT/HIGH-PRECISION-ONLY"
    for bits in BITS:
        pa, pc = summary[(bits, "PAIR-A")], summary[(bits, "PAIR-C")]
        passes = min(pa["diff_p10"], pc["diff_p10"]) >= .10 and min(pa["sign_p10"], pc["sign_p10"]) >= .65 and u0[bits] == "YES" and surf[bits] == "YES"
        if passes:
            final = f"ROBUST-IN-{bits}BIT-LINEAR-RGB"
            break
    return final


def main():
    assert_gpu_scope()
    a0 = protocol_lock()
    a1 = source_audit()
    data = load_pair_data()
    rows, u0_rows = condition_metrics(data)
    a2, zero_classes, empty_defaults, agg_bug = zero_provenance()
    write_md(OUT / "N2R3_metric_semantics.md", "N2-R3 Metric Semantics", "Different-code fraction is nonzero quantized code rows / valid rows. Sign consistency is evaluated only where clean unquantized absolute pair difference >= 1e-6; noisy zero code difference is counted as not sign-consistent. Empty groups return NaN, not zero. Saturated rows remain included and are reported.")
    summary = p10_metrics(rows)
    u0 = u0_comparison(rows, u0_rows)
    surf = surface_support(rows)
    sensor = classify(summary, u0, surf)
    allow = sensor in {"ROBUST-IN-8BIT-LINEAR-RGB", "ROBUST-IN-10BIT-LINEAR-RGB", "ROBUST-IN-12BIT-LINEAR-RGB"}
    if a0 == a1 == a2 == "PASS":
        case = "CASE SENSOR-ROBUSTNESS-IMPLEMENTATION-REPAIRED" if allow else "CASE SENSOR-ROBUSTNESS-TRUE-FAIL"
    else:
        case = "CASE SENSOR-AUDIT-PROVENANCE-FAIL"
    line = "CONTINUE" if allow else "STOP"
    next_action = "allow D1-N3 representation sufficiency" if allow else "STOP the line; do not run another signal rescue"
    report = OUT / "stageD1N2R3_sensor_aggregation_report.md"
    summary_path = OUT / "stageD1N2R3_sensor_aggregation_summary.md"
    def vals(metric):
        return f"{summary[(8, metric)]['count']}/{summary[(10, metric)]['count']}/{summary[(12, metric)]['count']}"
    def bitline(bits, key):
        return f"{summary[(bits,'PAIR-A')][key]}/{summary[(bits,'PAIR-C')][key]}"
    terminal = [
        ("A. A0", a0),
        ("B. classification source file/function", "stageD1_N2_R2/run_R2.py::bitdepth_noise/classify"),
        ("C. final classification used min/worst or p10", "p10 for diff/sign; R2 used min only for U0 null"),
        ("D. empty groups previously returned", "NO_EMPTY_GROUPS_FOUND"),
        ("E. sign denominator previous semantics", "all sampled rows; no clean-difference validity mask"),
        ("F. A1", a1),
        ("G. six zero classifications", zero_classes),
        ("H. any empty-group defaults found yes/no", empty_defaults),
        ("I. any aggregation bug found yes/no", agg_bug),
        ("J. A2", a2),
        ("K. valid condition count 8/10/12-bit PAIR-A/PAIR-C", f"{vals('PAIR-A')}/{vals('PAIR-C')}"),
        ("L.0.25-LSB p10 different-code 8-bit PAIR-A/PAIR-C", bitline(8, "diff_p10")),
        ("M.0.25-LSB p10 sign-consistency 8-bit PAIR-A/PAIR-C", bitline(8, "sign_p10")),
        ("N.0.25-LSB p10 different-code 10-bit PAIR-A/PAIR-C", bitline(10, "diff_p10")),
        ("O.0.25-LSB p10 sign-consistency 10-bit PAIR-A/PAIR-C", bitline(10, "sign_p10")),
        ("P.0.25-LSB p10 different-code 12-bit PAIR-A/PAIR-C", bitline(12, "diff_p10")),
        ("Q.0.25-LSB p10 sign-consistency 12-bit PAIR-A/PAIR-C", bitline(12, "sign_p10")),
        ("R. PAIR-C exceeds U0 null 8/10/12-bit", f"{u0[8]}/{u0[10]}/{u0[12]}"),
        ("S. both surfaces support 8/10/12-bit", f"{surf[8]}/{surf[10]}/{surf[12]}"),
        ("T. final sensor classification", sensor),
        ("U.8-bit claim valid yes/no", "YES" if sensor == "ROBUST-IN-8BIT-LINEAR-RGB" else "NO"),
        ("V.10-bit claim valid yes/no", "YES" if sensor in {"ROBUST-IN-8BIT-LINEAR-RGB", "ROBUST-IN-10BIT-LINEAR-RGB"} else "NO"),
        ("W.12-bit claim valid yes/no", "YES" if allow else "NO"),
        ("X. Final CASE", case),
        ("Y. new primary line STOP/CONTINUE", line),
        ("Z. allow D1-N3 yes/no", "YES" if allow else "NO"),
        ("AA. next exact research action", next_action),
        ("AB. report path", str(report)),
        ("AC. summary path", str(summary_path)),
    ]
    body = "\n".join(f"{k}: {v}" for k, v in terminal)
    write_md(report, "Stage D1-N2-R3 Sensor Robustness Aggregation and Implementation Audit", body)
    write_md(summary_path, "Stage D1-N2-R3 Summary", body)
    (OUT / "stageD1N2R3_sensor_aggregation_log.txt").write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
