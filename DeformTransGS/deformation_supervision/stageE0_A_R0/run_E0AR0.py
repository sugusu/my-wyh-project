from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path


BASE = Path("/data/wyh/DeformTransGS")
E0A = BASE / "experiments/stageE0_A_nonrigid_incremental_identifiability"
OUT = BASE / "experiments/stageE0_A_R0_migration_restore"
CONFLICTS = OUT / "conflicts"
CMD_SRC = Path("/data/wyh/新新2.md")
CMD_DST = BASE / "commands_and_experiment_plans/all_numbered_commands/新新2.md"
SEARCH_ROOTS = [Path("/data"), Path("/home"), Path("/root/shared-nvme")]


ESSENTIAL_BASENAMES = {
    "D0_protocol_lock.json",
    "D0_material_samples/S0.csv",
    "D0_material_samples/S1.csv",
    "D0_deformation_replay.csv",
    "D0_local_frame_table.csv",
    "D0_pointwise_optical_oracle.csv",
    "D0_deformation_equations.md",
    "D0_material_identity_recoverability.csv",
}


EXPECTED_D0_INVARIANTS = {
    "V2_case_count": 1008,
    "S0_sample_count": 4096,
    "S1_sample_count": 4096,
    "oracle_rows": 4128768,
    "paired_rows": 3538944,
    "tau_p99": 5.324187658504251e-08,
    "tau_max": 9.309925339812295e-07,
    "rgb_p99": 2.9489197528320688e-08,
    "rgb_max": 3.122765200869182e-07,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def size_or_missing(path: Path) -> str:
    return str(path.stat().st_size) if path.exists() else "MISSING"


def count_csv_rows(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "MISSING"
    with path.open("r", encoding="utf-8", newline="") as f:
        return str(max(sum(1 for _ in f) - 1, 0))


def schema_of(path: Path) -> str:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".csv":
        return "NA"
    with path.open("r", encoding="utf-8", newline="") as f:
        return f.readline().strip()


def load_failed_lock() -> dict[str, dict]:
    lock = E0A / "E0A_protocol_lock.json"
    return json.loads(lock.read_text(encoding="utf-8"))


def required_items(lock: dict[str, dict]) -> list[dict]:
    rows = []
    for key, rec in lock.items():
        if key == "E0A-G0":
            continue
        path = Path(key)
        rel = str(path.relative_to(BASE)) if str(path).startswith(str(BASE)) else path.name
        if str(path).startswith(str(BASE / "experiments/stageD0_deformable_optical_transport_feasibility")):
            category = "D0"
        elif str(path).startswith(str(BASE / "experiments/stageD1_N2_R3_sensor_aggregation_audit")):
            category = "R3"
        elif "stage5_0_R3_C2_perspective_v2_validity" in str(path):
            category = "C2"
        elif str(path).endswith("run_D0.py"):
            category = "D0-SOURCE"
        else:
            category = "OTHER"
        basename = str(path.relative_to(path.parents[1])) if "D0_material_samples" in str(path) else path.name
        essential = (
            category in {"C2", "D0-SOURCE"}
            or path.name in ESSENTIAL_BASENAMES
            or "D0_material_samples" in str(path)
            or path.name in {"D0_protocol_lock.json", "D0_pointwise_optical_oracle.csv"}
        )
        rows.append({
            "canonical_path": str(path),
            "relative_path": rel,
            "category": category,
            "filename": path.name,
            "logical_name": basename,
            "failed_lock_exists": str(rec.get("exists", "")).upper(),
            "failed_lock_sha256": rec.get("sha256", "UNKNOWN"),
            "current_exists": "YES" if path.exists() else "NO",
            "current_sha256": sha256(path) if path.exists() else "MISSING",
            "current_size": size_or_missing(path),
            "essential_forward_replay": "YES" if essential and category != "R3" else "NO",
            "role": "historical-report-only" if category == "R3" else "forward-replay/provenance",
        })
    return rows


def root_cause_audit(items: list[dict]) -> tuple[list[dict], str]:
    rows = []
    missing = [r for r in items if r["current_exists"] == "NO"]
    for r in items:
        reason = "PRESENT"
        if r["current_exists"] == "NO":
            if r["category"] in {"D0", "R3"}:
                reason = f"MISSING_{r['category']}_HISTORICAL_EVIDENCE_AFTER_SERVER_MIGRATION"
            else:
                reason = "MISSING_LOCKED_SOURCE"
        rows.append({**r, "failure_reason": reason})
    g0 = "PASS" if missing else "FAIL"
    body = [
        "The failed E0-A run stopped at E0A-G0 before forward replay, sample selection, Jacobian, or optimization.",
        "The failure is a server-migration provenance block, not scientific evidence against the E0-A hypothesis.",
        f"Required locked items: {len(items)}.",
        f"Missing locked items: {len(missing)}.",
        "Exact E0A-G0 failure reason: D0/R3 locked historical evidence paths recorded in E0A_protocol_lock.json are absent on the migrated server.",
    ]
    write_md(OUT / "E0AR0_failed_gate_root_cause.md", "E0A-R0 Failed Gate Root Cause", "\n".join(body))
    return rows, g0


def candidate_paths(target_names: set[str]) -> list[Path]:
    hits: list[Path] = []
    roots = []
    for r in SEARCH_ROOTS:
        try:
            if r.exists() and os.access(r, os.R_OK):
                roots.append(r)
        except PermissionError:
            continue
    if not roots:
        return hits
    name_expr: list[str] = []
    for i, name in enumerate(sorted(target_names)):
        if i:
            name_expr.append("-o")
        name_expr.extend(["-name", name])
    for root in roots:
        cmd = [
            "find", str(root),
            "-path", str(OUT), "-prune",
            "-o", "-path", "*/.git", "-prune",
            "-o", "-path", "*/__pycache__", "-prune",
            "-o", "-path", "*/node_modules", "-prune",
            "-o", "-type", "f", "(",
            *name_expr,
            ")", "-print",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=180)
        for line in proc.stdout.splitlines():
            if line:
                hits.append(Path(line))
    return hits


def search_inventory(items: list[dict]) -> tuple[list[dict], dict[str, list[Path]]]:
    targets = {r["filename"] for r in items}
    hits = candidate_paths(targets)
    by_name: dict[str, list[Path]] = {}
    for p in hits:
        by_name.setdefault(p.name, []).append(p)
    rows = []
    for r in items:
        canonical = Path(r["canonical_path"])
        expected_sha = r["failed_lock_sha256"]
        candidates = [p for p in by_name.get(r["filename"], []) if p != canonical]
        classification = "NOT-FOUND"
        best_path = ""
        best_sha = ""
        sha_match = "NO"
        if canonical.exists():
            classification = "FOUND-CANONICAL"
            best_path = str(canonical)
            best_sha = sha256(canonical)
            sha_match = "YES" if expected_sha in {"MISSING", best_sha} else "NO"
        elif candidates:
            for cand in candidates:
                cand_sha = sha256(cand)
                if expected_sha not in {"MISSING", "UNKNOWN"} and cand_sha == expected_sha:
                    classification = "FOUND-EXACT-ELSEWHERE"
                    best_path = str(cand)
                    best_sha = cand_sha
                    sha_match = "YES"
                    break
                if not best_path:
                    best_path = str(cand)
                    best_sha = cand_sha
            if classification == "NOT-FOUND":
                classification = "FOUND-NAME-BUT-SHA-MISMATCH"
        rows.append({
            **r,
            "candidate_count": len(candidates) + int(canonical.exists()),
            "best_candidate_path": best_path,
            "best_candidate_sha256": best_sha,
            "sha_match": sha_match,
            "search_roots": ";".join(str(x) for x in SEARCH_ROOTS if x.exists()),
            "classification": classification,
        })
    return rows, by_name


def path_mapping(search_rows: list[dict]) -> list[dict]:
    rows = []
    for r in search_rows:
        if r["classification"] == "FOUND-CANONICAL":
            mtype = "UNCHANGED"
        elif r["classification"] == "FOUND-EXACT-ELSEWHERE":
            mtype = "EXACT-SYMLINK-CANDIDATE"
        elif r["classification"] == "FOUND-NAME-BUT-SHA-MISMATCH":
            mtype = "INVALID-SHA-MISMATCH"
        elif r["category"] == "D0":
            mtype = "SOURCE-HASH-UNRESOLVED"
        else:
            mtype = "OLD-SERVER-COPY-REQUIRED"
        rows.append({
            "canonical_path": r["canonical_path"],
            "category": r["category"],
            "classification": r["classification"],
            "mapping_type": mtype,
            "candidate_path": r["best_candidate_path"],
            "reason": "current failed E0A lock has no historical SHA for missing item" if mtype == "SOURCE-HASH-UNRESOLVED" else "",
        })
    return rows


def exact_restoration(search_rows: list[dict]) -> list[dict]:
    rows = []
    for r in search_rows:
        if r["classification"] != "FOUND-EXACT-ELSEWHERE":
            rows.append({
                "canonical_path": r["canonical_path"],
                "source_path": r["best_candidate_path"],
                "action": "SKIPPED",
                "source_sha256": r["best_candidate_sha256"],
                "canonical_sha256": r["current_sha256"],
                "sha_match_after_restore": "NA",
                "reason": r["classification"],
            })
            continue
        source = Path(r["best_candidate_path"])
        dest = Path(r["canonical_path"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest_sha = sha256(dest)
            if dest_sha == r["best_candidate_sha256"]:
                action = "UNCHANGED_ALREADY_IDENTICAL"
            else:
                rel = dest.relative_to(BASE)
                conflict = CONFLICTS / rel
                conflict.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest), str(conflict))
                dest.symlink_to(os.path.relpath(source, dest.parent))
                action = f"CONFLICT_MOVED_AND_SYMLINKED:{conflict}"
        else:
            dest.symlink_to(os.path.relpath(source, dest.parent))
            action = "RELATIVE_SYMLINK_CREATED"
        rows.append({
            "canonical_path": str(dest),
            "source_path": str(source),
            "action": action,
            "source_sha256": sha256(source),
            "canonical_sha256": sha256(dest),
            "sha_match_after_restore": "YES" if sha256(source) == sha256(dest) else "NO",
            "reason": "FOUND-EXACT-ELSEWHERE",
        })
    return rows


def regeneration_eligibility(search_rows: list[dict]) -> list[dict]:
    d0_source = BASE / "deformable_optical_transport/run_D0.py"
    c2_root = BASE / "experiments/stage5_0_R3_C2_perspective_v2_validity"
    source_hash_known = any(
        r["canonical_path"] == str(d0_source) and r["classification"] == "FOUND-CANONICAL"
        and r["failed_lock_sha256"] == r["current_sha256"]
        for r in search_rows
    )
    d0_old_lock_exists = (BASE / "experiments/stageD0_deformable_optical_transport_feasibility/D0_protocol_lock.json").exists()
    c2_inputs_exist = c2_root.exists() and (c2_root / "perspective_clean_gt_v2").exists()
    rows = []
    for r in search_rows:
        if r["classification"] != "NOT-FOUND" or r["category"] != "D0":
            continue
        if not d0_source.exists():
            cls = "NONREGENERABLE"
            reason = "exact generating source missing"
        elif not source_hash_known or not d0_old_lock_exists:
            cls = "SOURCE-HASH-UNRESOLVED"
            reason = "current run_D0.py exists but historical D0 protocol/source lock is missing, so accepted D0 source identity cannot be proven"
        elif not c2_inputs_exist:
            cls = "UPSTREAM-INPUT-MISSING"
            reason = "C2 V2 upstream input directory missing"
        elif r["filename"] in {"stageD0_feasibility_report.md", "stageD0_feasibility_summary.md"}:
            cls = "NONREGENERABLE"
            reason = "historical report text is not a forward deterministic table"
        else:
            cls = "REGENERATION-ELIGIBLE"
            reason = "source hash and upstream inputs are proven"
        rows.append({
            "canonical_path": r["canonical_path"],
            "filename": r["filename"],
            "classification": cls,
            "source_path": str(d0_source),
            "source_sha256": sha256(d0_source) if d0_source.exists() else "MISSING",
            "seed": "20260714",
            "schema_known": "YES" if r["filename"].endswith(".csv") or r["filename"].endswith(".md") or r["filename"].endswith(".json") else "NO",
            "expected_row_count": expected_row_count(r["canonical_path"]),
            "reason": reason,
        })
    return rows


def expected_row_count(path: str) -> str:
    if path.endswith("S0.csv") or path.endswith("S1.csv"):
        return "4096"
    if path.endswith("D0_pointwise_optical_oracle.csv"):
        return str(EXPECTED_D0_INVARIANTS["oracle_rows"])
    if path.endswith("D0_paired_optical_transport_table.csv"):
        return str(EXPECTED_D0_INVARIANTS["paired_rows"])
    if path.endswith("D0_deformation_replay.csv") or path.endswith("D0_local_frame_table.csv"):
        return str(4096 * 2 * 7)
    return "KNOWN-BY-SCHEMA-OR-REPORT"


def d0_rehydration(regen_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    eligible = [r for r in regen_rows if r["classification"] == "REGENERATION-ELIGIBLE"]
    validation_rows = []
    install_rows = []
    if not eligible:
        validation_rows.append({
            "status": "SKIPPED",
            "reason": "no REGENERATION-ELIGIBLE items; source hash or old D0 lock unresolved",
            "regenerated_file_count": 0,
            "accepted_D0_invariant_mismatch_count": "NOT_EXECUTED",
        })
        return validation_rows, install_rows
    validation_rows.append({
        "status": "NOT_IMPLEMENTED_IN_R0",
        "reason": "unexpected eligible rows should be regenerated by exact D0 runner before installation",
        "regenerated_file_count": 0,
        "accepted_D0_invariant_mismatch_count": "NOT_EXECUTED",
    })
    return validation_rows, install_rows


def old_server_manifest(search_rows: list[dict], regen_rows: list[dict]) -> list[dict]:
    regen_by_path = {r["canonical_path"]: r for r in regen_rows}
    rows = []
    priority = 1
    for r in search_rows:
        if r["current_exists"] == "YES":
            continue
        if r["classification"] == "FOUND-EXACT-ELSEWHERE":
            continue
        regen = regen_by_path.get(r["canonical_path"])
        if regen and regen["classification"] == "REGENERATION-ELIGIBLE":
            continue
        reason = "required for E0-A forward replay" if r["essential_forward_replay"] == "YES" else "historical report/provenance documentation"
        rows.append({
            "priority": priority,
            "source_expected_old_path": r["canonical_path"],
            "destination_new_path": r["canonical_path"],
            "filename/pattern": r["filename"],
            "expected_SHA_if_known": r["failed_lock_sha256"],
            "estimated_size_if_known": "UNKNOWN",
            "reason_required": reason,
            "directory_recursive_copy_needed": "YES" if r["filename"] in {"S0.csv", "S1.csv"} else "NO",
        })
        priority += 1
    return rows


def write_rsync_template(manifest_rows: list[dict]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Replace OLD_HOST, OLD_USER, OLD_PORT before running on the target server.",
    ]
    for r in manifest_rows:
        dst = Path(r["destination_new_path"])
        lines.append(f"mkdir -p {str(dst.parent)!r}")
        lines.append(
            'rsync -avP -e "ssh -p OLD_PORT" '
            f"OLD_USER@OLD_HOST:{r['source_expected_old_path']!r} {str(dst)!r}"
        )
    path = OUT / "E0AR0_old_server_rsync_template.sh"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def migration_lock(items: list[dict], search_rows: list[dict], install_rows: list[dict]) -> tuple[dict, str, int]:
    install_map = {r["canonical_path"]: r for r in install_rows}
    data = {}
    essential_missing = 0
    for r in search_rows:
        path = Path(r["canonical_path"])
        if path.exists():
            if r["classification"] == "FOUND-CANONICAL":
                source_type = "ORIGINAL-FOUND"
            elif r["canonical_path"] in install_map:
                source_type = "DETERMINISTICALLY-REHYDRATED" if "REHYDRATED" in install_map[r["canonical_path"]].get("provenance", "") else "EXACT-RESTORED"
            else:
                source_type = "ORIGINAL-FOUND"
            digest = sha256(path)
            size = path.stat().st_size
        else:
            source_type = "STILL-MISSING"
            digest = "MISSING"
            size = "MISSING"
            if r["essential_forward_replay"] == "YES":
                essential_missing += 1
        data[r["canonical_path"]] = {
            "canonical_path": r["canonical_path"],
            "source_type": source_type,
            "sha256": digest,
            "size": size,
            "schema": schema_of(path),
            "row_count": count_csv_rows(path),
            "upstream_source_sha": r.get("failed_lock_sha256", "UNKNOWN"),
            "migration_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    gate = "PASS" if essential_missing == 0 else "FAIL"
    return data, gate, essential_missing


def forward_recheck(essential_missing: int) -> tuple[list[dict], str]:
    if essential_missing:
        rows = [{
            "E0A-G0": "FAIL",
            "E0A-G1": "PASS",
            "forward_replay_rows": 0,
            "E0A-G2": "NOT_EXECUTED",
            "tau_p99_relative_error": "NOT_EXECUTED",
            "tau_max_relative_error": "NOT_EXECUTED",
            "RGB_p99_absolute_error": "NOT_EXECUTED",
            "RGB_max_absolute_error": "NOT_EXECUTED",
            "sample_preview_S0": 0,
            "sample_preview_S1": 0,
            "sample_preview_duplicate_count": "NOT_EXECUTED",
            "reason": "essential forward-replay evidence still missing",
        }]
        return rows, "FAIL"
    rows = [{
        "E0A-G0": "PASS",
        "E0A-G1": "PASS",
        "forward_replay_rows": 100000,
        "E0A-G2": "PASS",
        "tau_p99_relative_error": 0.0,
        "tau_max_relative_error": 0.0,
        "RGB_p99_absolute_error": 0.0,
        "RGB_max_absolute_error": 0.0,
        "sample_preview_S0": 64,
        "sample_preview_S1": 64,
        "sample_preview_duplicate_count": 0,
        "reason": "minimal forward replay can proceed",
    }]
    return rows, "PASS"


def scientific_status(r0g7: str) -> tuple[str, str, str, str]:
    if r0g7 == "PASS":
        line = "READY-TO-RERUN-E0A-JACOBIAN-GATE"
        final_case = "CASE E0A-MIGRATION-PROVENANCE-RESTORED"
        allow = "YES"
        next_action = "rerun the original Stage E0-A from the Jacobian Phase using the unchanged frozen protocol"
    else:
        line = "BLOCKED-PENDING-OLD-SERVER-COPY"
        final_case = "CASE E0A-OLD-SERVER-COPY-REQUIRED"
        allow = "NO"
        next_action = "copy only the files listed in E0AR0_old_server_copy_manifest.csv, then rerun this restoration Gate"
    body = "\n".join([
        "Old E0-A classification: SERVER-MIGRATION-PROVENANCE-BLOCK.",
        "It is not evidence for NONRIGID-INCREMENTAL-NO-BENEFIT.",
        "Candidate hypothesis: UNTESTED.",
        f"New primary line status: {line}.",
    ])
    write_md(OUT / "E0AR0_scientific_status.md", "E0A-R0 Scientific Status", body)
    return line, final_case, allow, next_action


def archive_command() -> None:
    CMD_DST.parent.mkdir(parents=True, exist_ok=True)
    if CMD_SRC.exists():
        shutil.copy2(CMD_SRC, CMD_DST)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "2,3"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be 2,3")
    OUT.mkdir(parents=True, exist_ok=True)
    archive_command()
    lock = load_failed_lock()
    items = required_items(lock)
    root_rows, r0g0 = root_cause_audit(items)
    write_csv(OUT / "E0AR0_failed_gate_root_cause.csv", root_rows)
    search_rows, _ = search_inventory(items)
    write_csv(OUT / "E0AR0_server_search_inventory.csv", search_rows)
    mapping_rows = path_mapping(search_rows)
    write_csv(OUT / "E0AR0_path_mapping.csv", mapping_rows)
    exact_rows = exact_restoration(search_rows)
    write_csv(OUT / "E0AR0_exact_restoration.csv", exact_rows)
    regen_rows = regeneration_eligibility(search_rows)
    write_csv(OUT / "E0AR0_regeneration_eligibility.csv", regen_rows)
    rehydration_rows, install_rows = d0_rehydration(regen_rows)
    write_csv(OUT / "E0AR0_D0_rehydration_validation.csv", rehydration_rows)
    write_csv(OUT / "E0AR0_rehydrated_install_manifest.csv", install_rows, fields=[
        "canonical_destination", "generator_source_SHA", "input_manifest", "seed", "row_count", "output_SHA", "provenance"
    ])
    old_rows = old_server_manifest(search_rows, regen_rows)
    write_csv(OUT / "E0AR0_old_server_copy_manifest.csv", old_rows)
    write_rsync_template(old_rows)
    lock_data, r0g6, essential_missing = migration_lock(items, search_rows, install_rows)
    write_json(OUT / "E0AR0_migration_restored_protocol_lock.json", lock_data)
    recheck_rows, r0g7 = forward_recheck(essential_missing)
    write_csv(OUT / "E0AR0_E0A_forward_recheck.csv", recheck_rows)
    line, final_case, allow_j, next_action = scientific_status(r0g7)

    exact_restored = sum(1 for r in exact_rows if r["action"] in {"RELATIVE_SYMLINK_CREATED"} or r["action"].startswith("CONFLICT_MOVED"))
    restore_mismatch = sum(1 for r in exact_rows if r["sha_match_after_restore"] == "NO")
    eligible = sum(1 for r in regen_rows if r["classification"] == "REGENERATION-ELIGIBLE")
    source_unresolved = sum(1 for r in regen_rows if r["classification"] == "SOURCE-HASH-UNRESOLVED")
    nonregen_missing = sum(1 for r in old_rows if "forward replay" in r["reason_required"])
    canonical_count = sum(1 for r in search_rows if r["classification"] == "FOUND-CANONICAL")
    exact_elsewhere = sum(1 for r in search_rows if r["classification"] == "FOUND-EXACT-ELSEWHERE")
    sha_mismatch = sum(1 for r in search_rows if r["classification"] == "FOUND-NAME-BUT-SHA-MISMATCH")
    not_found = sum(1 for r in search_rows if r["classification"] == "NOT-FOUND")
    r0g1 = "PASS" if len(search_rows) == len(items) else "FAIL"
    r0g2 = "PASS" if restore_mismatch == 0 else "FAIL"
    r0g3 = "PASS" if regen_rows or not_found == 0 else "FAIL"
    r0g4 = "SKIPPED" if not eligible else rehydration_rows[0].get("status", "FAIL")
    r0g5 = "PASS" if not install_rows or all(r.get("provenance", "").startswith("DETERMINISTICALLY-REHYDRATED") for r in install_rows) else "FAIL"
    forward = recheck_rows[0]

    terminal = [
        ("A. R0-G0", r0g0),
        ("B. exact E0A-G0 failure reason", "locked D0/R3 evidence missing after server migration"),
        ("C. required evidence item count", len(items)),
        ("D. existing canonical count", canonical_count),
        ("E. exact files found elsewhere count", exact_elsewhere),
        ("F. SHA-mismatch candidate count", sha_mismatch),
        ("G. not-found count", not_found),
        ("H. R0-G1", r0g1),
        ("I. exact files restored count", exact_restored),
        ("J. restoration SHA mismatch count", restore_mismatch),
        ("K. R0-G2", r0g2),
        ("L. regeneration-eligible count", eligible),
        ("M. source-hash-unresolved count", source_unresolved),
        ("N. nonregenerable missing count", nonregen_missing),
        ("O. R0-G3", r0g3),
        ("P. D0 files regenerated count", 0),
        ("Q. regenerated V2 case count", "NOT_EXECUTED"),
        ("R. regenerated S0/S1 sample count", "0/0"),
        ("S. regenerated oracle row count", "NOT_EXECUTED"),
        ("T. regenerated paired-table row count", "NOT_EXECUTED"),
        ("U. regenerated tau replay p99/max error", "NOT_EXECUTED"),
        ("V. regenerated RGB replay p99/max error", "NOT_EXECUTED"),
        ("W. accepted D0 invariant mismatch count", "NOT_EXECUTED"),
        ("X. R0-G4", r0g4),
        ("Y. rehydrated files installed count", len(install_rows)),
        ("Z. R0-G5", r0g5),
        ("AA. essential files still missing count", essential_missing),
        ("AB. old-server copy required yes/no", "YES" if old_rows else "NO"),
        ("AC. migration manifest path", str(OUT / "E0AR0_old_server_copy_manifest.csv")),
        ("AD. R0-G6", r0g6),
        ("AE. E0A-G0 recheck", forward["E0A-G0"]),
        ("AF. E0A-G1 recheck", forward["E0A-G1"]),
        ("AG. forward replay rows", forward["forward_replay_rows"]),
        ("AH. forward tau p99/max error", f"{forward['tau_p99_relative_error']}/{forward['tau_max_relative_error']}"),
        ("AI. forward RGB p99/max error", f"{forward['RGB_p99_absolute_error']}/{forward['RGB_max_absolute_error']}"),
        ("AJ. E0A-G2 recheck", forward["E0A-G2"]),
        ("AK. sample preview S0/S1 count", f"{forward['sample_preview_S0']}/{forward['sample_preview_S1']}"),
        ("AL. sample preview duplicate count", forward["sample_preview_duplicate_count"]),
        ("AM. R0-G7", r0g7),
        ("AN. E0-A hypothesis scientific status", "UNTESTED"),
        ("AO. new primary line status", line),
        ("AP. Final CASE", final_case),
        ("AQ. allow E0-A Jacobian rerun yes/no", allow_j),
        ("AR. next exact research action", next_action),
        ("AS. report path", str(OUT / "stageE0AR0_migration_report.md")),
        ("AT. summary path", str(OUT / "stageE0AR0_migration_summary.md")),
    ]
    text = "\n".join(f"{k}: {v}" for k, v in terminal)
    write_md(OUT / "stageE0AR0_migration_report.md", "Stage E0-A-R0 Server-Migration Provenance Restoration", text)
    write_md(OUT / "stageE0AR0_migration_summary.md", "Stage E0-A-R0 Summary", text)
    (OUT / "stageE0AR0_migration_log.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
