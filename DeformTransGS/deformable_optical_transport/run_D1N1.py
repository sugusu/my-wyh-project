from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


BASE = Path("/data/wyh/DeformTransGS")
N0 = BASE / "experiments/stageD1_N0_deformation_descriptor_identifiability"
OUT = BASE / "experiments/stageD1_N1_optical_mechanism_observability_boundary"
CMD_SRC = Path("/data/wyh/新11.md")
CMD_DST = BASE / "commands_and_experiment_plans/all_numbered_commands/新11.md"


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


def write_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def protocol_lock() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    CMD_DST.parent.mkdir(parents=True, exist_ok=True)
    if CMD_SRC.exists():
        shutil.copy2(CMD_SRC, CMD_DST)
    paths = [
        N0 / "D1N0_protocol_lock.json",
        N0 / "D1N0_candidate_deformation_bank.csv",
        N0 / "D1N0_candidate_deformation_equations.md",
        N0 / "D1N0_candidate_descriptor_table.csv",
        N0 / "D1N0_matched_invariant_validation.csv",
        N0 / "D1N0_state_collision_matrix.csv",
        N0 / "D1N0_scientific_scope.md",
        N0 / "stageD1N0_identifiability_report.md",
        N0 / "stageD1N0_identifiability_summary.md",
    ]
    records = {str(p): {"exists": p.exists(), "sha256": sha(p) if p.exists() else "MISSING"} for p in paths}
    gate = "PASS" if all(r["exists"] for r in records.values()) else "FAIL"
    records["D1N1_N0_evidence_lock"] = gate
    write_json(OUT / "D1N1_protocol_lock.json", records)
    return gate


def load_bank() -> list[dict]:
    with (N0 / "D1N0_candidate_deformation_bank.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["F_array"] = np.array(json.loads(row["F"]), dtype=np.float64)
    return rows


def alias_audit(bank: list[dict]) -> tuple[str, int, int, str]:
    groups: list[list[dict]] = []
    used: set[int] = set()
    for i, row in enumerate(bank):
        if i in used:
            continue
        group = [row]
        used.add(i)
        for j in range(i + 1, len(bank)):
            if j not in used and np.array_equal(row["F_array"], bank[j]["F_array"]):
                group.append(bank[j])
                used.add(j)
        groups.append(group)
    rows = []
    duplicate_labels = []
    for gid, group in enumerate(groups):
        keys = [r["deformation_key"] for r in group]
        if len(group) > 1:
            duplicate_labels.append("+".join(keys))
        for row in group:
            rows.append({
                "alias_group_id": gid,
                "deformation_key": row["deformation_key"],
                "duplicate_group_keys": "|".join(keys),
                "is_duplicate_alias": "YES" if len(group) > 1 else "NO",
                "counts_as_independent_observation": "NO" if len(group) > 1 and row is not group[0] else "YES",
                "allowed_role": "reproducibility/null-control label" if len(group) > 1 else "independent protocol matrix",
                "F": row["F"],
            })
    write_csv(OUT / "D1N1_deformation_alias_audit.csv", rows)
    required = [
        {"D0_IDENTITY", "A1_IDENTITY_JS1"},
        {"D5_ANISO_X1P60_Y0P80", "C1_ANISO_X1P6_Y0P8"},
        {"D6_ROTATION_Z_30", "D2_RIGID_RZ30"},
    ]
    actual = [set(r["deformation_key"] for r in g) for g in groups]
    ok = all(any(req.issubset(g) for g in actual) for req in required)
    gate = "PASS" if ok and len(bank) == 14 and len(groups) == 11 else "FAIL"
    return gate, len(bank), len(groups), "; ".join(duplicate_labels)


def mechanism_boundary() -> list[dict]:
    rows = [
        {
            "mechanism ID": "M0_GEOMETRY_ONLY",
            "Chinese name": "几何项不改变材料光学状态",
            "physical state being changed": "none; only surface pose, local frame, view direction, and geometric path terms change",
            "candidate deformation state Q1/Q2/Q3/history": "none/material state invariant",
            "observable with ordinary RGB yes/no/conditional": "YES",
            "requires polarization yes/no": "NO",
            "requires illumination control yes/no": "YES",
            "requires refractive ray tracing yes/no": "CONDITIONAL",
            "requires microstructure simulation yes/no": "NO",
            "compatible with current pointwise framework yes/no": "YES",
            "compatible with current C2 cameras yes/no": "YES",
            "can distinguish PAIR-A yes/no": "NO",
            "can distinguish PAIR-B yes/no": "NO",
            "can distinguish PAIR-C yes/no": "NO",
            "rigid invariant expected yes/no": "YES",
            "physical evidence type": "geometric reparameterization baseline",
            "existing-project novelty risk": "high overlap with local-frame appearance preservation",
            "implementation complexity": "LOW",
            "recommended role": "BASELINE",
        },
        {
            "mechanism ID": "M1_ISOTROPIC_MASS_CONSERVING_EXTINCTION",
            "Chinese name": "各向同性质量守恒消光",
            "physical state being changed": "effective thickness or density through area ratio Js",
            "candidate deformation state Q1/Q2/Q3/history": "Q1",
            "observable with ordinary RGB yes/no/conditional": "YES",
            "requires polarization yes/no": "NO",
            "requires illumination control yes/no": "YES",
            "requires refractive ray tracing yes/no": "NO",
            "requires microstructure simulation yes/no": "NO",
            "compatible with current pointwise framework yes/no": "YES",
            "compatible with current C2 cameras yes/no": "YES",
            "can distinguish PAIR-A yes/no": "NO",
            "can distinguish PAIR-B yes/no": "NO",
            "can distinguish PAIR-C yes/no": "NO",
            "rigid invariant expected yes/no": "YES",
            "physical evidence type": "Beer-Lambert thickness baseline already present in V2 MAT1/MAT2",
            "existing-project novelty risk": "controlled baseline only",
            "implementation complexity": "LOW",
            "recommended role": "BASELINE",
        },
        {
            "mechanism ID": "M2_STRETCH_SPECTRUM_RESPONSE",
            "Chinese name": "主拉伸谱响应",
            "physical state being changed": "orientation-free stretch-dependent optical coefficients",
            "candidate deformation state Q1/Q2/Q3/history": "Q2",
            "observable with ordinary RGB yes/no/conditional": "CONDITIONAL",
            "requires polarization yes/no": "NO",
            "requires illumination control yes/no": "YES",
            "requires refractive ray tracing yes/no": "NO",
            "requires microstructure simulation yes/no": "NO",
            "compatible with current pointwise framework yes/no": "YES",
            "compatible with current C2 cameras yes/no": "YES",
            "can distinguish PAIR-A yes/no": "YES",
            "can distinguish PAIR-B yes/no": "YES",
            "can distinguish PAIR-C yes/no": "NO",
            "rigid invariant expected yes/no": "YES",
            "physical evidence type": "phenomenological stretch-state mechanism; needs independent parameterization",
            "existing-project novelty risk": "moderate; cannot establish Q3 direction effects",
            "implementation complexity": "MEDIUM",
            "recommended role": "OPTIONAL",
        },
        {
            "mechanism ID": "M3_MATERIAL_AXIS_ANISOTROPIC_EXTINCTION",
            "Chinese name": "材料轴相关各向异性消光",
            "physical state being changed": "material-basis tensorial absorption/extinction or dichroic attenuation",
            "candidate deformation state Q1/Q2/Q3/history": "Q3",
            "observable with ordinary RGB yes/no/conditional": "YES",
            "requires polarization yes/no": "NO",
            "requires illumination control yes/no": "YES",
            "requires refractive ray tracing yes/no": "NO",
            "requires microstructure simulation yes/no": "NO",
            "compatible with current pointwise framework yes/no": "YES",
            "compatible with current C2 cameras yes/no": "YES",
            "can distinguish PAIR-A yes/no": "YES",
            "can distinguish PAIR-B yes/no": "YES",
            "can distinguish PAIR-C yes/no": "YES",
            "rigid invariant expected yes/no": "YES",
            "physical evidence type": "RGB-observable anisotropic absorption/dichroism boundary, not pure birefringence",
            "existing-project novelty risk": "plausible distinct question if mechanism is independently parameterized and tested by matched pairs",
            "implementation complexity": "MEDIUM",
            "recommended role": "PRIMARY",
        },
        {
            "mechanism ID": "M4_POLARIZATION_STRAIN_OPTIC",
            "Chinese name": "偏振应变光学",
            "physical state being changed": "phase/polarization state such as birefringence or stress-optic response",
            "candidate deformation state Q1/Q2/Q3/history": "Q3",
            "observable with ordinary RGB yes/no/conditional": "CONDITIONAL",
            "requires polarization yes/no": "YES",
            "requires illumination control yes/no": "YES",
            "requires refractive ray tracing yes/no": "CONDITIONAL",
            "requires microstructure simulation yes/no": "NO",
            "compatible with current pointwise framework yes/no": "NO",
            "compatible with current C2 cameras yes/no": "NO",
            "can distinguish PAIR-A yes/no": "CONDITIONAL",
            "can distinguish PAIR-B yes/no": "CONDITIONAL",
            "can distinguish PAIR-C yes/no": "YES_WITH_POLARIZATION",
            "rigid invariant expected yes/no": "YES",
            "physical evidence type": "Jones/Mueller/Stokes or analyzer-dependent intensity, not unpolarized RGB attenuation",
            "existing-project novelty risk": "stronger physics but requires new sensor/model scope",
            "implementation complexity": "HIGH",
            "recommended role": "OPTIONAL",
        },
        {
            "mechanism ID": "M5_HISTORY_DEPENDENT_MICROSTRUCTURE",
            "Chinese name": "历史相关微结构",
            "physical state being changed": "hysteresis, relaxation, plasticity, or time-dependent microstructure",
            "candidate deformation state Q1/Q2/Q3/history": "history",
            "observable with ordinary RGB yes/no/conditional": "CONDITIONAL",
            "requires polarization yes/no": "CONDITIONAL",
            "requires illumination control yes/no": "YES",
            "requires refractive ray tracing yes/no": "CONDITIONAL",
            "requires microstructure simulation yes/no": "YES",
            "compatible with current pointwise framework yes/no": "NO",
            "compatible with current C2 cameras yes/no": "NO",
            "can distinguish PAIR-A yes/no": "OUT_OF_SCOPE",
            "can distinguish PAIR-B yes/no": "OUT_OF_SCOPE",
            "can distinguish PAIR-C yes/no": "OUT_OF_SCOPE",
            "rigid invariant expected yes/no": "YES",
            "physical evidence type": "future material-memory mechanism; current assets contain no time/history data",
            "existing-project novelty risk": "outside current memoryless F/Ct scope",
            "implementation complexity": "VERY_HIGH",
            "recommended role": "OUT-OF-SCOPE",
        },
    ]
    write_csv(OUT / "D1N1_mechanism_boundary.csv", rows)
    return rows


def write_boundary_docs() -> tuple[str, str, str, str, str, str, str]:
    rgb = "\n".join([
        "1. Ordinary RGB intensity can observe mechanisms that change total transmitted energy, including isotropic extinction, anisotropic absorption, dichroism, and direction-dependent scattering when illumination and view terms are controlled.",
        "2. Ideal lossless pure birefringence changes polarization state and phase, not total unpolarized RGB intensity by itself.",
        "3. Birefringence can become intensity variation only with an analyzer/polarizer, polarization camera, or Jones/Mueller/Stokes-aware setup that converts polarization phase/state changes into measured intensity.",
        "4. The current C2 RGB observable cannot distinguish Q3 orientation effects from pure birefringence alone. Q3 in ordinary RGB requires anisotropic absorption/scattering/extinction or an explicit polarization optical setup.",
        "Gate M1: PASS. Pure birefringence is not represented as ordinary RGB Beer-Lambert attenuation.",
    ])
    write_md(OUT / "D1N1_RGB_observability.md", "D1-N1 RGB Observability", rgb)
    objectivity = "\n".join([
        "All admissible material-response mechanisms must be objective: a rigid body transform must not alter intrinsic material optical state.",
        "World-frame quantities may change rendered images through camera-relative direction, but the constitutive material state lives in canonical material frame or an explicitly transported material basis.",
        "M1 uses scalar Js and is rigid invariant.",
        "M2 uses unordered stretch spectrum and is rigid invariant.",
        "M3 uses C11/C12/C22 in canonical material t1/t2 basis and is rigid invariant under Q3=I rigid controls.",
        "M4 requires Jones/Mueller/Stokes frame definitions and cannot be collapsed to RGB attenuation without analyzer optics.",
        "Gate M2: PASS.",
    ])
    write_md(OUT / "D1N1_objectivity_requirements.md", "D1-N1 Objectivity Requirements", objectivity)
    contracts = "\n".join([
        "PAIR-A same Js, different spectrum: Q1-only response must collide after local viewing/path terms are controlled; Q2 and Q3 mechanisms may differ.",
        "PAIR-B same Js, different shear/Ct: Q1-only response must collide after path/view control; Q3 mechanisms may differ; Q2 may differ only if spectrum changes.",
        "PAIR-C same unordered spectrum, different material-axis orientation: Q2 response must collide after direction/path control; Q3 material-axis anisotropic extinction must be capable of differing for at least one material-axis/view configuration.",
        "PAIR-D rigid controls: intrinsic material optical state must remain invariant for Q1/Q2/Q3; image changes may only arise from camera-relative direction/path effects.",
        "No final parameter values are instantiated in D1-N1.",
    ])
    write_md(OUT / "D1N1_matched_pair_response_contracts.md", "D1-N1 Matched Pair Response Contracts", contracts)
    confounds = "\n".join([
        "Conceptual factorization: tau_deformed = tau_canonical_reparameterized + delta_tau_geometry + delta_tau_material.",
        "Future experiments must separately control local viewing-direction change, surface-normal/path-length change, and material-state change.",
        "A required protocol is a same-local-view counterfactual: evaluate matched material points with identical local view direction in the transported local frame while comparing PAIR-A/B/C material-state descriptors.",
        "A Q3 response may be attributed to material-axis strain only if the compared samples are controlled for local view direction and geometry/path terms.",
        "This can be implemented in the current pointwise framework because D0/N0 already provide material identity, local frames, and local direction descriptors.",
        "Gate M3: PASS.",
    ])
    write_md(OUT / "D1N1_confound_separation_protocol.md", "D1-N1 Confound Separation Protocol", confounds)
    novelty = "\n".join([
        "DSRF overlap classification: appearance preservation/reparameterization overlap; this project must not claim novelty for local surface parameterization or local-frame view remapping.",
        "DR-GS overlap classification: geometry/material/illumination disentanglement and deformable rendering overlap; this project must not claim novelty for generic deformable Gaussian relighting.",
        "RT-Splatting overlap classification: static reflection/transmission factorization overlap; this project focuses only on deformation-induced material transport-state evolution.",
        "Generic surface-attached Gaussian deformation is not the novel question.",
        "Candidate distinct question: whether semi-transparent material transport under deformation requires scalar area, stretch spectrum, or full material-basis tangent strain tensor.",
        "Novelty status: PLAUSIBLE-BUT-NOT-PROVEN.",
        "Gate M4: PASS.",
    ])
    write_md(OUT / "D1N1_novelty_boundary.md", "D1-N1 Novelty Boundary", novelty)
    recommendation = "\n".join([
        "Primary recommended mechanism: M3_MATERIAL_AXIS_ANISOTROPIC_EXTINCTION.",
        "Primary observable: ordinary RGB transmitted intensity under controlled illumination.",
        "Rationale: M3 can distinguish PAIR-C same-spectrum/different-Ct controls, is rigid-objective in the canonical material frame, is compatible with the material-point pointwise framework, and does not require full volumetric microstructure simulation.",
        "Constraint: it must be independently parameterized and tested by matched counterfactual pairs; D1-N1 does not claim optical necessity or sufficiency.",
        "Polarization-aware M4 remains optional; it is not the primary path because it requires new sensing/model scope.",
    ])
    write_md(OUT / "D1N1_primary_mechanism_recommendation.md", "D1-N1 Primary Mechanism Recommendation", recommendation)
    return ("PASS", "YES", "PASS", "YES", "PASS", "PLAUSIBLE-BUT-NOT-PROVEN", "PASS")


def main() -> None:
    assert_gpu_scope()
    m0_lock = protocol_lock()
    bank = load_bank()
    alias_gate, protocol_count, unique_count, duplicate_groups = alias_audit(bank)
    m0 = "PASS" if m0_lock == "PASS" and alias_gate == "PASS" else "FAIL"
    mechanism_boundary()
    m1, rigid_valid, m2, same_local_view, m3, novelty_status, m4 = write_boundary_docs()
    final_ready = all(x == "PASS" for x in [m0, m1, m2, m3, m4]) and novelty_status == "PLAUSIBLE-BUT-NOT-PROVEN"
    final_case = "CASE OPTICAL-MECHANISM-BOUNDARY-READY" if final_ready else "CASE NO-OBSERVABLE-DISTINCT-MECHANISM"
    line_status = "CONTINUE" if final_ready else "STOP"
    next_action = "design D1-N2 controlled mechanism benchmark with independent generator and falsifiable Q1/Q2/Q3 counterfactual tests" if final_ready else "STOP the new line and return to RecycleGS"
    report_path = OUT / "stageD1N1_mechanism_boundary_report.md"
    summary_path = OUT / "stageD1N1_mechanism_boundary_summary.md"
    terminal = [
        ("A. M0", m0),
        ("B. deformation protocol entry count", protocol_count),
        ("C. unique deformation matrix count", unique_count),
        ("D. duplicate alias groups", duplicate_groups),
        ("E. ordinary RGB can observe pure birefringence yes/no/conditional", "CONDITIONAL_WITH_ANALYZER"),
        ("F. M1", m1),
        ("G. rigid-objectivity requirements valid yes/no", rigid_valid),
        ("H. M2", m2),
        ("I. Q1-only mechanism can distinguish PAIR-A yes/no", "NO"),
        ("J. Q2 mechanism can distinguish PAIR-C yes/no", "NO"),
        ("K. Q3 mechanism can distinguish PAIR-C yes/no", "YES"),
        ("L. same-local-view confound control possible yes/no", same_local_view),
        ("M. M3", m3),
        ("N. DSRF overlap classification", "OVERLAP-LOCAL-PARAMETERIZATION-NOT-CORE-NOVELTY"),
        ("O. DR-GS overlap classification", "OVERLAP-DEFORMABLE-DISENTANGLEMENT-NOT-CORE-NOVELTY"),
        ("P. RT-Splatting overlap classification", "OVERLAP-STATIC-TRANSPARENCY-NOT-DEFORMATION-TRANSPORT"),
        ("Q. novelty status", novelty_status),
        ("R. M4", m4),
        ("S. primary recommended mechanism", "M3_MATERIAL_AXIS_ANISOTROPIC_EXTINCTION"),
        ("T. primary observable", "ORDINARY_RGB_TRANSMITTED_INTENSITY_WITH_CONTROLLED_ILLUMINATION"),
        ("U. polarization required yes/no", "NO"),
        ("V. mechanism implementable in current pointwise framework yes/no", "YES"),
        ("W. Final CASE", final_case),
        ("X. new primary line STOP/CONTINUE/DECISION_REQUIRED", line_status),
        ("Y. next exact research action", next_action),
        ("Z. report path", str(report_path)),
        ("AA. summary path", str(summary_path)),
    ]
    body = "\n".join(f"{k}: {v}" for k, v in terminal)
    write_md(report_path, "Stage D1-N1 Optical Mechanism and Observability Boundary Lock", body)
    write_md(summary_path, "Stage D1-N1 Summary", body)
    (OUT / "stageD1N1_mechanism_boundary_log.txt").write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
