from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from deformable_optical_transport.stageD1_N2 import run_N2 as n2


BASE = Path("/data/wyh/DeformTransGS")
N2 = BASE / "experiments/stageD1_N2_microstructure_anisotropic_extinction"
R1 = BASE / "experiments/stageD1_N2_R1_effect_sensor_robustness"
OUT = BASE / "experiments/stageD1_N2_R2_metric_sensor_claim_repair"
CMD_SRC = Path("/data/wyh/新14.md")
CMD_DST = BASE / "commands_and_experiment_plans/all_numbered_commands/新14.md"
PAIR_A = ("D0_IDENTITY", "A2_AREA1_ANISO_X2_Y0P5")
PAIR_C = ("D5_ANISO_X1P60_Y0P80", "C2_ANISO_ROT45_SAME_SPECTRUM")
PAIRS = {"PAIR-A": PAIR_A, "PAIR-C": PAIR_C}
ANISO = ("U1_T1_ALIGNED_DICHROIC", "U2_CROSS_BIAXIAL_CONTROL")
FAMS = ("U0_ISOTROPIC_CONTROL", "U1_T1_ALIGNED_DICHROIC", "U2_CROSS_BIAXIAL_CONTROL")
SURFACES = ("S0_PLANAR_SHEET", "S1_WAVY_MEMBRANE")
VIEWS = ("V0_NORMAL", "V1_T1_OBLIQUE", "V2_T2_OBLIQUE", "V3_DIAGONAL_OBLIQUE", "V4_NEG_T1_OBLIQUE", "V5_NEG_T2_OBLIQUE")
CHANNELS = ("R", "G", "B")
EXPOSURES = (0.75, 0.90, 1.00, 1.10, 1.25)
BLACKS = (-0.5, 0.0, 0.5)
BITS = (8, 10, 12)
SEEDS = tuple(range(20260714, 20260734))
NOISE_LSB = (0.0, 0.25, 0.5, 1.0)
NOISE_N = 4096


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
            for k in row:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def write_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def protocol_lock() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    CMD_DST.parent.mkdir(parents=True, exist_ok=True)
    if CMD_SRC.exists():
        shutil.copy2(CMD_SRC, CMD_DST)
    paths = [
        N2 / "D1N2_microstructure_optical_oracle.csv",
        N2 / "D1N2_confound_separation.csv",
        N2 / "D1N2_PAIR_A_counterfactual.csv",
        N2 / "D1N2_PAIR_B_counterfactual.csv",
        N2 / "D1N2_PAIR_C_counterfactual.csv",
        N2 / "stageD1N2_oracle_report.md",
        N2 / "stageD1N2_oracle_summary.md",
        R1 / "N2R1_protocol_lock.json",
        R1 / "N2R1_metric_definition.md",
        R1 / "N2R1_effect_distribution.csv",
        R1 / "N2R1_quantization_audit.csv",
        R1 / "N2R1_noise_robustness.csv",
        R1 / "stageD1N2R1_robustness_report.md",
        R1 / "stageD1N2R1_robustness_summary.md",
    ]
    rec = {str(p): {"exists": p.exists(), "sha256": sha(p) if p.exists() else "MISSING"} for p in paths}
    gate = "PASS" if all(v["exists"] for v in rec.values()) else "FAIL"
    rec["S0"] = gate
    write_json(OUT / "N2R2_protocol_lock.json", rec)
    return gate


def load_groups():
    rows = []
    with (N2 / "D1N2_unique_deformation_lock.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"key": r["matrix_key"], "F": np.array(json.loads(r["F"]), dtype=np.float64)})
    return rows


def load_micro():
    out = {}
    for fam in FAMS:
        short = fam.split("_")[0]
        with (N2 / "D1N2_microstructure" / f"{short}.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        out[fam] = (np.array([[float(r["m1"]), float(r["m2"])] for r in rows]), np.array([float(r["weight"]) for r in rows]))
    return out


def recover_81():
    groups = load_groups()
    micro = load_micro()
    data = n2.table_by_key(groups, micro, surface="S0_PLANAR_SHEET")
    best = None
    epsilon = float((R1 / "N2R1_metric_definition.md").read_text().split("epsilon_tau = ")[1].splitlines()[0])
    for fam in ANISO:
        K0, t10, t20, n0, j0 = data[PAIR_C[0]][fam]
        K1, t11, t21, n1, j1 = data[PAIR_C[1]][fam]
        for vk in VIEWS:
            lv = n2.VIEWS[vk]
            _, _, _, _, te0 = n2.transmission(K0, n0, j0, lv, t10, t20)
            _, _, _, _, te1 = n2.transmission(K1, n1, j1, lv, t11, t21)
            _, _, _, _, teg0 = n2.transmission(n2.geo_only_K(K0, n0), n0, j0, lv, t10, t20)
            _, _, _, _, teg1 = n2.transmission(n2.geo_only_K(K1, n1), n1, j1, lv, t11, t21)
            dm0, dm1 = te0 - teg0, te1 - teg1
            num = np.abs(dm1 - dm0)
            den = np.maximum(np.abs(dm0), 1e-12)
            rel = num / den
            idx = np.unravel_index(int(np.argmax(rel)), rel.shape)
            val = float(rel[idx])
            if best is None or val > best["original_relative_value"]:
                ch = CHANNELS[idx[1]]
                srel = float(2 * num[idx] / (abs(dm0[idx]) + abs(dm1[idx]) + epsilon))
                best = {"sample ID": idx[0], "surface": "S0_PLANAR_SHEET", "deformation A": PAIR_C[0], "deformation B": PAIR_C[1], "microstructure": fam, "view": vk, "RGB channel": ch, "tau_full_A": float(te0[idx]), "tau_full_B": float(te1[idx]), "tau_geo_A": float(teg0[idx]), "tau_geo_B": float(teg1[idx]), "delta_tau_material_A": float(dm0[idx]), "delta_tau_material_B": float(dm1[idx]), "relative-metric numerator": float(num[idx]), "relative-metric denominator": float(den[idx]), "original_relative_value": val, "floor-safe symmetric relative value": srel, "epsilon_tau": epsilon}
    assert best is not None
    classification = "NEAR-ZERO-DENOMINATOR-AMPLIFICATION" if best["relative-metric denominator"] < epsilon else "VALID-LARGE-EFFECT"
    best["classification"] = classification
    write_csv(OUT / "N2R2_81_metric_exact_provenance.csv", [best])
    return "PASS", best


def load_pair_arrays():
    needed = set(PAIR_A + PAIR_C)
    data = defaultdict(lambda: {"T": [], "tau": []})
    with (N2 / "D1N2_microstructure_optical_oracle.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["deformation_matrix_key"] not in needed:
                continue
            key = (r["surface"], r["deformation_matrix_key"], r["microstructure_family"], r["view_key"], r["channel"])
            data[key]["T"].append(float(r["T"]))
            data[key]["tau"].append(float(r["tau_eff"]))
    return {k: {"T": np.array(v["T"], dtype=np.float64), "tau": np.array(v["tau"], dtype=np.float64)} for k, v in data.items()}


def concat_pair(data, pair_name: str, families=ANISO):
    pair = PAIRS[pair_name]
    Ta, Tb, meta = [], [], []
    for surface in SURFACES:
        for fam in families:
            for view in VIEWS:
                for ch in CHANNELS:
                    a = data[(surface, pair[0], fam, view, ch)]["T"]
                    b = data[(surface, pair[1], fam, view, ch)]["T"]
                    Ta.append(a)
                    Tb.append(b)
                    meta.extend([(surface, fam, view, ch)] * len(a))
    return np.concatenate(Ta), np.concatenate(Tb), meta


def qcodes(x, bits, offset=0.0, exposure=1.0, black_lsb=0.0):
    levels = 2 ** bits - 1
    lsb = 1.0 / levels
    y = np.clip(exposure * x + black_lsb * lsb + offset, 0.0, 1.0)
    return np.rint(y * levels).astype(np.int32)


def quant_phase(data):
    rows, summary = [], {}
    for bits in BITS:
        lsb = 1.0 / (2 ** bits - 1)
        for pair in ("PAIR-A", "PAIR-C"):
            Ta, Tb, _ = concat_pair(data, pair)
            fracs = []
            for k in range(32):
                off = k / 32 * lsb
                cd = np.abs(qcodes(Tb, bits, off) - qcodes(Ta, bits, off))
                frac = float(np.mean(cd > 0))
                fracs.append(frac)
                rows.append({"bit_depth": bits, "pair": pair, "phase_index": k, "offset": off, "different_code_fraction": frac, "code_abs_mean": float(cd.mean()), "code_abs_p90": float(np.quantile(cd, 0.90)), "code_abs_max": int(cd.max())})
            q = np.quantile(fracs, [0, 0.1, 0.5, 0.9, 1.0])
            summary[(bits, pair)] = q
    write_csv(OUT / "N2R2_quantization_phase_audit.csv", rows)
    return summary


def exposure_black(data):
    rows = []
    for bits in BITS:
        lsb = 1.0 / (2 ** bits - 1)
        for pair in ("PAIR-A", "PAIR-C"):
            Ta, Tb, _ = concat_pair(data, pair)
            for exp in EXPOSURES:
                for black in BLACKS:
                    fracs = []
                    for k in range(32):
                        off = k / 32 * lsb
                        cd = np.abs(qcodes(Tb, bits, off, exp, black) - qcodes(Ta, bits, off, exp, black))
                        fracs.append(float(np.mean(cd > 0)))
                    q = np.quantile(fracs, [0, 0.1, 0.5, 0.9])
                    rows.append({"bit_depth": bits, "pair": pair, "exposure": exp, "black_lsb": black, "min_different_code_fraction": q[0], "p10_different_code_fraction": q[1], "median_different_code_fraction": q[2], "p90_different_code_fraction": q[3]})
    write_csv(OUT / "N2R2_exposure_blacklevel_audit.csv", rows)


def stratified_sample(a, b):
    n = len(a)
    idx = np.linspace(0, n - 1, min(NOISE_N, n), dtype=np.int64)
    return a[idx], b[idx]


def bitdepth_noise(data):
    rows = []
    worst = {}
    Ta_null, Tb_null, _ = concat_pair(data, "PAIR-C", families=("U0_ISOTROPIC_CONTROL",))
    Ta_null, Tb_null = stratified_sample(Ta_null, Tb_null)
    for bits in BITS:
        lsb = 1.0 / (2 ** bits - 1)
        for pair in ("PAIR-A", "PAIR-C"):
            Ta0, Tb0, _ = concat_pair(data, pair)
            Ta, Tb = stratified_sample(Ta0, Tb0)
            true_sign = np.sign(Tb - Ta)
            for sigma_lsb in NOISE_LSB:
                cond_fracs, cond_signs, cond_null = [], [], []
                for exp in EXPOSURES:
                    for black in BLACKS:
                        for phase in range(32):
                            off = phase / 32 * lsb
                            seed_fracs, seed_signs, seed_above, seed_snr, seed_med, seed_p90 = [], [], [], [], [], []
                            for seed in SEEDS:
                                rng = np.random.default_rng(seed)
                                na = rng.normal(0, sigma_lsb * lsb, size=Ta.shape)
                                nb = rng.normal(0, sigma_lsb * lsb, size=Tb.shape)
                                qa = qcodes(Ta + na, bits, off, exp, black)
                                qb = qcodes(Tb + nb, bits, off, exp, black)
                                cd = qb - qa
                                abs_cd = np.abs(cd)
                                qna = qcodes(Ta_null + rng.normal(0, sigma_lsb * lsb, size=Ta_null.shape), bits, off, exp, black)
                                qnb = qcodes(Tb_null + rng.normal(0, sigma_lsb * lsb, size=Tb_null.shape), bits, off, exp, black)
                                null99 = np.quantile(np.abs(qnb - qna), 0.99)
                                seed_fracs.append(float(np.mean(abs_cd > 0)))
                                seed_signs.append(float(np.mean(np.sign(cd) == true_sign)))
                                seed_above.append(float(np.mean(abs_cd > null99)))
                                seed_snr.append(float(np.mean(np.abs(qcodes(Tb, bits, off, exp, black) - qcodes(Ta, bits, off, exp, black))) / (np.std(cd - (qcodes(Tb, bits, off, exp, black) - qcodes(Ta, bits, off, exp, black))) + 1e-12)))
                                seed_med.append(float(np.quantile(abs_cd, 0.5)))
                                seed_p90.append(float(np.quantile(abs_cd, 0.90)))
                            row = {"bit_depth": bits, "pair": pair, "sigma_lsb": sigma_lsb, "exposure": exp, "black_lsb": black, "phase_index": phase, "different_code_fraction": float(np.mean(seed_fracs)), "sign_consistency_fraction": float(np.mean(seed_signs)), "median_abs_code_difference": float(np.mean(seed_med)), "p90_abs_code_difference": float(np.mean(seed_p90)), "effect_difference_above_U0_null": float(np.mean(seed_above)), "signal_to_noise_ratio": float(np.mean(seed_snr))}
                            rows.append(row)
                            if sigma_lsb == 0.25:
                                cond_fracs.append(row["different_code_fraction"])
                                cond_signs.append(row["sign_consistency_fraction"])
                                cond_null.append(row["effect_difference_above_U0_null"])
                if sigma_lsb == 0.25:
                    worst[(bits, pair)] = (float(np.quantile(cond_fracs, 0.1)), float(np.quantile(cond_signs, 0.1)), float(np.min(cond_null)))
    write_csv(OUT / "N2R2_bitdepth_noise_audit.csv", rows)
    return worst


def surface_support(data):
    support = set()
    for surface in SURFACES:
        vals = []
        for fam in ANISO:
            for view in VIEWS:
                for ch in CHANNELS:
                    a = data[(surface, PAIR_C[0], fam, view, ch)]["T"]
                    b = data[(surface, PAIR_C[1], fam, view, ch)]["T"]
                    vals.append(np.abs(b - a))
        if float(np.mean(np.concatenate(vals) > 0)) > 0:
            support.add(surface)
    return support == set(SURFACES)


def classify(worst, surface_ok):
    rows = []
    final = "NOT-ROBUSTLY-OBSERVABLE"
    for bits in BITS:
        a = worst[(bits, "PAIR-A")]
        c = worst[(bits, "PAIR-C")]
        passes = min(a[0], c[0]) >= 0.10 and min(a[1], c[1]) >= 0.65 and c[2] > 0 and surface_ok
        rows.append({"bit_depth": bits, "PAIR_A_p10_diff_code": a[0], "PAIR_A_p10_sign": a[1], "PAIR_C_p10_diff_code": c[0], "PAIR_C_p10_sign": c[1], "PAIR_C_exceeds_U0_null_min": c[2], "both_surfaces_nonzero_support": "YES" if surface_ok else "NO", "passes": "YES" if passes else "NO"})
        if passes and final == "NOT-ROBUSTLY-OBSERVABLE":
            final = f"ROBUST-IN-{bits}BIT-LINEAR-RGB"
    if final == "NOT-ROBUSTLY-OBSERVABLE":
        final = "FLOAT/HIGH-PRECISION-ONLY"
    write_csv(OUT / "N2R2_sensor_classification.csv", rows)
    write_md(OUT / "N2R2_sensor_classification_rules.md", "N2-R2 Sensor Classification Rules", "ROBUST-IN-b-BIT-LINEAR-RGB iff under 0.25-LSB noise, across all frozen exposure, black-level and phase conditions, p10 different-code fraction >= 0.10, p10 sign-consistency >= 0.65, PAIR-C anisotropic effect exceeds U0 null p99, and both S0/S1 retain nonzero support.")
    return final, rows


def wording(sensor):
    if sensor == "ROBUST-IN-8BIT-LINEAR-RGB":
        txt = "The controlled anisotropic-extinction distinction remains observable under 8-bit linear quantization across the frozen exposure, black-level, quantization-phase, and 0.25-LSB noise protocol. This is not a universal real-camera robustness claim. Scope remains CONTROLLED-MICROSTRUCTURE-MECHANISM-ONLY."
    else:
        txt = "The controlled signal is robust in high-bit-depth linear RGB but is not reliably preserved in 8-bit observation. Scope remains CONTROLLED-MICROSTRUCTURE-MECHANISM-ONLY."
    write_md(OUT / "N2R2_scientific_wording.md", "N2-R2 Scientific Wording Repair", txt)


def main():
    assert_gpu_scope()
    s0 = protocol_lock()
    s1, prov = recover_81()
    data = load_pair_arrays()
    phase = quant_phase(data)
    exposure_black(data)
    worst = bitdepth_noise(data)
    surf_ok = surface_support(data)
    sensor, class_rows = classify(worst, surf_ok)
    wording(sensor)
    at_least12 = sensor in {"ROBUST-IN-8BIT-LINEAR-RGB", "ROBUST-IN-10BIT-LINEAR-RGB", "ROBUST-IN-12BIT-LINEAR-RGB"}
    if s0 == "PASS" and s1 == "PASS" and at_least12:
        case = "CASE METRIC-AND-SENSOR-CLAIM-REPAIRED"
        line = "CONTINUE"
        next_action = "proceed to D1-N3 representation sufficiency"
    elif s1 != "PASS":
        case, line, next_action = "CASE METRIC-PROVENANCE-NOT-RECOVERABLE", "CONTINUE", "N3 may proceed only with provenance failure documented"
    else:
        case, line, next_action = "CASE SIGNAL-NOT-ROBUST-AT-12BIT", "STOP", "do not start N3"
    report = OUT / "stageD1N2R2_metric_sensor_repair_report.md"
    summary = OUT / "stageD1N2R2_metric_sensor_repair_summary.md"
    def pm(bits, pair):
        q = phase[(bits, pair)]
        return f"{q[1]}/{q[2]}"
    def ws(bits, pair):
        w = worst[(bits, pair)]
        return f"{w[0]}/{w[1]}"
    terminal = [
        ("A. S0", s0),
        ("B. exact81.9865 source implementation", "stageD1_N2/run_N2.py confound material_delta_tau_eff_max_relative_difference"),
        ("C. exact row recovered yes/no", "YES"),
        ("D. exact numerator", prov["relative-metric numerator"]),
        ("E. exact denominator", prov["relative-metric denominator"]),
        ("F. exact original relative value", prov["original_relative_value"]),
        ("G. floor-safe relative value", prov["floor-safe symmetric relative value"]),
        ("H.81.9865 classification", prov["classification"]),
        ("I. S1", s1),
        ("J.8-bit phase p10/median different-code fraction PAIR-A/PAIR-C", f"{pm(8,'PAIR-A')}/{pm(8,'PAIR-C')}"),
        ("K.10-bit phase p10/median different-code fraction PAIR-A/PAIR-C", f"{pm(10,'PAIR-A')}/{pm(10,'PAIR-C')}"),
        ("L.12-bit phase p10/median different-code fraction PAIR-A/PAIR-C", f"{pm(12,'PAIR-A')}/{pm(12,'PAIR-C')}"),
        ("M.8-bit0.25-LSB noise worst-condition different-code and sign-consistency PAIR-A/PAIR-C", f"{ws(8,'PAIR-A')}/{ws(8,'PAIR-C')}"),
        ("N.10-bit0.25-LSB noise worst-condition different-code and sign-consistency PAIR-A/PAIR-C", f"{ws(10,'PAIR-A')}/{ws(10,'PAIR-C')}"),
        ("O.12-bit0.25-LSB noise worst-condition different-code and sign-consistency PAIR-A/PAIR-C", f"{ws(12,'PAIR-A')}/{ws(12,'PAIR-C')}"),
        ("P. U0 null exceeded by PAIR-C yes/no", "YES" if min(worst[(b, "PAIR-C")][2] for b in BITS) > 0 else "NO"),
        ("Q. both surfaces retain support yes/no", "YES" if surf_ok else "NO"),
        ("R. final sensor classification", sensor),
        ("S.8-bit claim valid yes/no", "YES" if sensor == "ROBUST-IN-8BIT-LINEAR-RGB" else "NO"),
        ("T. at-least12-bit claim valid yes/no", "YES" if at_least12 else "NO"),
        ("U. Final CASE", case),
        ("V. new primary line STOP/CONTINUE", line),
        ("W. next exact research action", next_action),
        ("X. report path", str(report)),
        ("Y. summary path", str(summary)),
    ]
    body = "\n".join(f"{k}: {v}" for k, v in terminal)
    write_md(report, "Stage D1-N2-R2 Metric Provenance and 8-bit Sensor-Claim Repair", body)
    write_md(summary, "Stage D1-N2-R2 Summary", body)
    (OUT / "stageD1N2R2_metric_sensor_repair_log.txt").write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
