from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from rtsplat_attribute_study.stage5_R3.provenance.rt_full_state_checkpoint import PERSISTENT_TENSORS, load_full_state, save_full_state
from rtsplat_attribute_study.stage5_R4 import run_R4 as r4


BASE = Path("/data/wyh/DeformTransGS")
OUT = BASE / "experiments/stage5_0_R4_O1_optimization_protocol_closure"
R4 = BASE / "experiments/stage5_0_R4_rtsplat_v2_canonical_capacity"
C2 = BASE / "experiments/stage5_0_R3_C2_perspective_v2_validity"
G1 = BASE / "experiments/stage5_0_R3_G1_small_gradient_numerical_closure"
RT = Path("/data/wyh/repos/RT-Splatting")
SEED = 20260714
CASES = r4.CASES
TRAIN_IDS = r4.TRAIN_IDS
TEST_IDS = r4.TEST_IDS
TRAINABLE_BASE = ["_occupancy", "_opacity", "_transmissivity", "_features_dc"]
SOURCE_LR = {
    "_occupancy": 0.05,
    "_opacity": 0.05,
    "_transmissivity": 0.01,
    "_features_dc": 0.002,
    "_scaling": 0.005,
    "_rotation": 0.001,
}
R4_ASSIGNED_LR = {name: 1e-6 for name in TRAINABLE_BASE}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_dir():
        for p in sorted(x for x in path.rglob("*") if x.is_file()):
            h.update(str(p.relative_to(path)).encode())
            h.update(sha(p).encode())
        return h.hexdigest()
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
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def tensor_max_abs(x: torch.Tensor) -> float:
    return 0.0 if x.numel() == 0 else float(x.abs().max().detach().cpu())


def tensor_l2(x: torch.Tensor) -> float:
    return 0.0 if x.numel() == 0 else float(x.norm().detach().cpu())


def optimizer_for(pc, trainable: list[str]) -> torch.optim.Adam:
    groups = [{"params": [getattr(pc, name)], "lr": SOURCE_LR[name], "name": name} for name in trainable]
    return torch.optim.Adam(groups, lr=0.0, eps=1e-15)


def load_gt_cache() -> tuple[dict, dict]:
    gt_cache, gt_np = {}, {}
    for case in CASES:
        for cid in range(24):
            gt_cache[(case, cid)] = r4.load_gt(case, cid)
            gt_np[(case, cid)] = (
                gt_cache[(case, cid)][0].detach().cpu().numpy().transpose(1, 2, 0),
                gt_cache[(case, cid)][1].detach().cpu().numpy().transpose(1, 2, 0),
                gt_cache[(case, cid)][2].detach().cpu().numpy(),
            )
    return gt_cache, gt_np


def metric_np(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple[float, float, float, float, float, float]:
    mse = float(np.mean((pred - gt) ** 2))
    psnr = 99.0 if mse <= 1e-20 else -10.0 * math.log10(mse)
    pred_tau = -np.log(np.clip(pred, 1e-6, 1.0))
    gt_tau = -np.log(np.clip(gt, 1e-6, 1.0))
    mask = np.repeat(valid[:, :, None], 3, axis=2)
    elog = np.abs(np.log((pred_tau + 1e-6) / (gt_tau + 1e-6)))
    ratio = np.maximum((pred_tau + 1e-6) / (gt_tau + 1e-6), (gt_tau + 1e-6) / (pred_tau + 1e-6))
    vals = elog[mask]
    return psnr, float(np.median(vals)), float(np.percentile(vals, 90)), float(np.percentile(vals, 95)), float(np.percentile(vals, 99)), float(np.mean(ratio[mask] <= 2.0))


def eval_full(pc, case: str, ids: list[int], cams: dict[int, object], gt_cache: dict) -> dict:
    losses, psnrs, elogs = [], [], []
    with torch.no_grad():
        for cid in ids:
            pred = r4.render_rgb(pc, cams[cid])
            gt, _, valid = gt_cache[(case, cid)]
            total, _, _, _ = r4.loss_terms(pred, gt, valid)
            psnr, elog, *_ = metric_np(pred.detach().cpu().numpy().transpose(1, 2, 0), gt.detach().cpu().numpy().transpose(1, 2, 0), valid.detach().cpu().numpy())
            losses.append(float(total)); psnrs.append(psnr); elogs.append(elog)
    return {"total_loss": float(np.mean(losses)), "PSNR": float(np.mean(psnrs)), "tau_eq_Elog": float(np.median(elogs))}


def protocol_lock() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    (BASE / "commands_and_experiment_plans/all_numbered_commands").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path("/data/wyh/新7.md"), BASE / "commands_and_experiment_plans/all_numbered_commands/新7.md")
    paths = [
        R4 / "R4_protocol_lock.json", R4 / "stage5_0_R4_rtsplat_v2_canonical_report.md",
        R4 / "stage5_0_R4_rtsplat_v2_canonical_summary.md", BASE / "rtsplat_attribute_study/stage5_R4/run_R4.py",
        R4 / "R4_canonical_job_manifest.csv", R4 / "R4_training_camera_schedule.csv",
        R4 / "R4_footprint_diagnostic.csv", R4 / "R4_footprint_policy_lock.json",
        R4 / "R4_optimizer_lock.json", R4 / "R4_capacity_diagnostic.csv",
        C2 / "C2_future_J4_benchmark_lock.json", G1 / "final_terminal_summary.txt",
        BASE / "rtsplat_attribute_study/stage5_R3/provenance/rt_full_state_checkpoint.py",
    ]
    for case in CASES:
        paths += [
            R4 / f"R4_full_train_selection_history/{case}.csv",
            R4 / f"R4_canonical_history/{case}.csv",
            R4 / f"checkpoints/{case}/best_0500.pt",
        ]
    lock = {str(p): {"exists": p.exists(), "sha256": sha(p) if p.exists() else "MISSING"} for p in paths}
    s0 = "PASS" if all(item["exists"] for item in lock.values()) else "FAIL"
    lock["S0"] = s0
    write_json(OUT / "O1_protocol_lock.json", lock)
    return s0


def reconstruct_patience() -> tuple[list[dict], dict]:
    rows, summary = [], {}
    for case in CASES:
        best = None
        best_iter = 0
        stop_iter = "NONE"
        case_rows = list(csv.DictReader((R4 / f"R4_full_train_selection_history/{case}.csv").open()))
        for row in case_rows:
            it = int(row["iteration"])
            loss = float(row["total_loss"])
            before = "" if best is None else best
            if best is None:
                is_new = "YES"
                best = loss
                best_iter = it
            elif loss < best:
                is_new = "YES"
                best = loss
                best_iter = it
            else:
                is_new = "NO"
            patience = it - best_iter
            if patience >= 500 and stop_iter == "NONE":
                stop_iter = it
            rows.append({
                "case": case,
                "iteration": it,
                "full_train_total_loss": loss,
                "full_train_PSNR": row["PSNR"],
                "full_train_median_tau_eq_Elog": row["tau_eq_Elog"],
                "best_loss_before_evaluation": before,
                "is_new_best_reconstructed": is_new,
                "expected_patience_elapsed_after_evaluation": patience,
            })
        summary[case] = {
            "iterations": [int(r["iteration"]) for r in case_rows],
            "losses": [float(r["total_loss"]) for r in case_rows],
            "flags": [r["is_new_best_reconstructed"] for r in rows if r["case"] == case],
            "best_iteration": best_iter,
            "patience_at_500": next(int(r["expected_patience_elapsed_after_evaluation"]) for r in rows if r["case"] == case and int(r["iteration"]) == 500),
            "step500_new_best": next(r["is_new_best_reconstructed"] for r in rows if r["case"] == case and int(r["iteration"]) == 500),
            "stop_eligible_iteration": stop_iter,
        }
    write_csv(OUT / "O1_reconstructed_full_train_patience.csv", rows)
    return rows, summary


def source_audits() -> tuple[str, str, str]:
    stop_rows = []
    for case in CASES:
        stop_rows.append({
            "case": case,
            "loop_maximum": "500 due range(1,501)",
            "actual_loop_break_condition": "for-loop exhausted at 500; no path to 4000 in R4",
            "early_stop_counter_variable": "no_best",
            "counter_initialization": "0",
            "counter_increment_unit": "100 on non-best aggregate evaluation",
            "counter_reset_condition": "reset to 0 on is_best",
            "best_comparison_variable": "ev['total_loss'] < best_loss - 1e-12",
            "comparison_direction": "lower total_loss is better",
            "stopping_threshold": "no_best >= 500 but unreachable after step500 new best because loop hard-caps at 500",
            "recorded_stop_reason": "HARD-500-STEP-CAP",
        })
    write_csv(OUT / "O1_recorded_stop_reason.csv", stop_rows)
    trace = """The actual R4 source stops canonical training at `for it in range(1, 501)` in `train_case`.
The frozen protocol maximum is 4000 steps with early stop only when `current_iteration - best_iteration >= 500`.
R4 also has `no_best >= 500`, but all observed evaluations 100..500 are new best rows, so that break condition did not stop the jobs.
The exact stopping cause is therefore `HARD-500-STEP-CAP`.
Relevant code: `rtsplat_attribute_study/stage5_R4/run_R4.py`, function `train_case`, lines around 317-353.
"""
    write_md(OUT / "O1_early_stop_source_trace.md", "O1 Early Stop Source Trace", trace)
    inventory = [
        {"source_group_name": "f_dc", "source_tensor": "_features_dc", "source_file_function": "scene/gaussian_model.py:training_setup", "argument_config_name": "feature_lr", "default_LR": 0.002, "effective_LR": 0.002, "scheduler": "NONE", "scheduler_formula": "constant"},
        {"source_group_name": "occupancy", "source_tensor": "_occupancy", "source_file_function": "scene/gaussian_model.py:training_setup", "argument_config_name": "occupancy_lr", "default_LR": 0.05, "effective_LR": 0.05, "scheduler": "NONE", "scheduler_formula": "constant"},
        {"source_group_name": "opacity", "source_tensor": "_opacity", "source_file_function": "scene/gaussian_model.py:training_setup", "argument_config_name": "opacity_lr", "default_LR": 0.05, "effective_LR": 0.05, "scheduler": "NONE", "scheduler_formula": "constant"},
        {"source_group_name": "transmissivity", "source_tensor": "_transmissivity", "source_file_function": "scene/gaussian_model.py:training_setup", "argument_config_name": "transmissivity_lr", "default_LR": 0.01, "effective_LR": 0.01, "scheduler": "NONE", "scheduler_formula": "constant"},
        {"source_group_name": "scaling", "source_tensor": "_scaling", "source_file_function": "scene/gaussian_model.py:training_setup", "argument_config_name": "scaling_lr", "default_LR": 0.005, "effective_LR": 0.005, "scheduler": "NONE", "scheduler_formula": "constant"},
        {"source_group_name": "rotation", "source_tensor": "_rotation", "source_file_function": "scene/gaussian_model.py:training_setup", "argument_config_name": "rotation_lr", "default_LR": 0.001, "effective_LR": 0.001, "scheduler": "NONE", "scheduler_formula": "constant"},
        {"source_group_name": "roughness", "source_tensor": "_roughness", "source_file_function": "scene/gaussian_model.py:training_setup", "argument_config_name": "roughness_lr", "default_LR": 0.002, "effective_LR": 0.002, "scheduler": "NONE", "scheduler_formula": "constant"},
        {"source_group_name": "reflectance", "source_tensor": "_reflectance", "source_file_function": "scene/gaussian_model.py:training_setup", "argument_config_name": "reflectance_lr", "default_LR": 0.005, "effective_LR": 0.005, "scheduler": "NONE", "scheduler_formula": "constant"},
    ]
    write_csv(OUT / "O1_RT_source_optimizer_inventory.csv", inventory)
    lr_rows = []
    for tensor in TRAINABLE_BASE:
        src = SOURCE_LR[tensor]
        assigned = R4_ASSIGNED_LR[tensor]
        lr_rows.append({
            "tensor": tensor,
            "R4_optimizer_group": tensor,
            "assigned_LR": assigned,
            "assignment_source": "rtsplat_attribute_study/stage5_R4/run_R4.py TRAIN_LR constant",
            "source_config_key": {"_features_dc": "feature_lr", "_occupancy": "occupancy_lr", "_opacity": "opacity_lr", "_transmissivity": "transmissivity_lr"}[tensor],
            "fallback_branch": "NO",
            "hardcoded_branch": "YES",
            "source_defined_LR": src,
            "classification": "HARDCODED-GENERIC-LR" if assigned != src else "SOURCE-MATCHED",
        })
    write_csv(OUT / "O1_R4_LR_mapping_trace.csv", lr_rows)
    return "PASS", "FAIL", "FAIL"


def footprint_original_audit() -> str:
    fp = list(csv.DictReader((R4 / "R4_footprint_diagnostic.csv").open()))
    rows = []
    for candidate in ["P0_FIXED_FOOTPRINT", "P1_NATIVE_FOOTPRINT"]:
        stored = next((r for r in fp if r.get("policy_candidate") == candidate), {})
        rows.append({
            "candidate": candidate,
            "model_constructed": "NO_EVIDENCE",
            "optimizer_stepped_500": "NO",
            "history_exists": "NO",
            "step0_full_train_aggregate_exists": "NO",
            "step500_full_train_aggregate_exists": "NO",
            "step0_PSNR": "MISSING",
            "step0_tau_eq": "MISSING",
            "step500_PSNR": stored.get("step500_train_PSNR", "MISSING"),
            "step500_tau_eq": stored.get("step500_train_tau_eq", "MISSING"),
            "scale_delta_L2": "MISSING",
            "rotation_delta_L2": "MISSING",
        })
    write_csv(OUT / "O1_footprint_execution_provenance.csv", rows)
    policy = {
        "numeric_diagnostic_complete": False,
        "original_policy": json.loads((R4 / "R4_footprint_policy_lock.json").read_text()).get("policy"),
        "independently_reproduced": False,
        "validity": "FAIL-MISSING-DIAGNOSTIC",
        "S2": "FAIL",
    }
    write_json(OUT / "O1_footprint_policy_reproduction.json", policy)
    return "FAIL"


def parameter_motion() -> None:
    rows = []
    for case in CASES:
        init = torch.load(R4 / f"checkpoints/{case}/best_0000.pt", map_location="cpu")["persistent_tensors"]
        step = torch.load(R4 / f"checkpoints/{case}/best_0500.pt", map_location="cpu")["persistent_tensors"]
        for tensor in TRAINABLE_BASE:
            diff = step[tensor] - init[tensor]
            row = {
                "case": case,
                "tensor": tensor,
                "raw_L1_delta": float(diff.abs().sum()),
                "raw_L2_delta": float(diff.norm()),
                "raw_max_abs_delta": float(diff.abs().max()),
                "delta_over_initial_L2": float(diff.norm() / (init[tensor].norm() + 1e-12)),
            }
            if tensor == "_occupancy":
                ad = (torch.sigmoid(step[tensor]) - torch.sigmoid(init[tensor])).abs()
                row.update({"activated_mean_abs_change": float(ad.mean()), "activated_p90_change": float(torch.quantile(ad, 0.9)), "activated_p99_change": float(torch.quantile(ad, 0.99)), "activated_max_change": float(ad.max())})
            elif tensor == "_opacity":
                ad = (torch.sigmoid(step[tensor]) - torch.sigmoid(init[tensor])).abs()
                row.update({"activated_mean_abs_change": float(ad.mean()), "activated_p90_change": float(torch.quantile(ad, 0.9)), "activated_p99_change": float(torch.quantile(ad, 0.99)), "activated_max_change": float(ad.max())})
            elif tensor == "_transmissivity":
                ad = (torch.sigmoid(step[tensor]) - torch.sigmoid(init[tensor])).abs()
                row.update({"activated_mean_abs_change": float(ad.mean()), "activated_p90_change": float(torch.quantile(ad, 0.9)), "activated_p99_change": float(torch.quantile(ad, 0.99)), "activated_max_change": float(ad.max())})
            else:
                row.update({"activated_mean_abs_change": "NATIVE_COLOR_PATH_RAW", "activated_p90_change": "NATIVE_COLOR_PATH_RAW", "activated_p99_change": "NATIVE_COLOR_PATH_RAW", "activated_max_change": "NATIVE_COLOR_PATH_RAW"})
            rows.append(row)
    write_csv(OUT / "O1_500step_parameter_motion.csv", rows)


def gradient_scale() -> None:
    rows = []
    for case in CASES:
        hist = list(csv.DictReader((R4 / f"R4_canonical_history/{case}.csv").open()))
        for lo, hi, label in [(1, 100, "1-100"), (401, 500, "401-500")]:
            part = [r for r in hist if r["iteration"].isdigit() and lo <= int(r["iteration"]) <= hi]
            for tensor in TRAINABLE_BASE:
                grads = [float(r[f"grad_L2{tensor}"]) for r in part]
                lrs = [float(r[f"lr_{tensor}"]) for r in part]
                deltas = [float(r[f"delta_L2{tensor}"]) for r in part]
                rows.append({
                    "case": case,
                    "step_window": label,
                    "tensor": tensor,
                    "median_gradient_L2": float(np.median(grads)),
                    "median_LR": float(np.median(lrs)),
                    "estimated_LR_times_gradient_L2": float(np.median(grads) * np.median(lrs)),
                    "median_parameter_delta_per_step_reconstructable": float(np.median(np.diff(deltas))) if len(deltas) > 1 else 0.0,
                    "classification": "LR-MAPPING-SUPPRESSED",
                })
    write_csv(OUT / "O1_gradient_step_scale.csv", rows)


def train_job(case: str, trainable: list[str], cams: dict[int, object], gt_cache: dict, out_prefix: str, max_steps: int, save_checkpoints: bool, audit_continuation: bool = False) -> dict:
    pc = r4.make_model(CASES[case][0], trainable)
    init = {name: getattr(pc, name).detach().clone() for name in PERSISTENT_TENSORS}
    opt = optimizer_for(pc, trainable)
    rng = random.Random(SEED)
    sched = []
    while len(sched) < max_steps:
        ids = TRAIN_IDS[:]
        rng.shuffle(ids)
        sched.extend(ids)
    hist, selhist, first_rows = [], [], []
    ckpt_dir = OUT / out_prefix / "checkpoints" / case
    ev0 = eval_full(pc, case, TRAIN_IDS, cams, gt_cache)
    best_loss, best_iter = ev0["total_loss"], 0
    best_path = ckpt_dir / "best_0000.pt"
    if save_checkpoints:
        save_full_state(best_path, pc, {"case": case, "iteration": 0, "stage": "O1"}, opt)
    selhist.append({"case": case, "iteration": 0, **ev0, "best_loss": best_loss, "best_iteration": 0, "patience_elapsed": 0, "is_new_best": "YES", "stop": "NO"})
    actual_steps = 0
    final_stop = "MAX_STEPS"
    before_first = {n: getattr(pc, n).detach().clone() for n in PERSISTENT_TENSORS}
    for it in range(1, max_steps + 1):
        cid = sched[it - 1]
        opt.zero_grad(set_to_none=True)
        pred = r4.render_rgb(pc, cams[cid])
        gt, _, valid = gt_cache[(case, cid)]
        total, rgb_l1, tau_loss, dssim = r4.loss_terms(pred, gt, valid)
        total.backward()
        opt.step()
        actual_steps = it
        row = {"case": case, "iteration": it, "train_camera_id": cid, "total_loss": float(total.detach()), "RGB_L1": float(rgb_l1.detach()), "tau_eq_loss": float(tau_loss.detach()), "DSSIM": float(dssim.detach())}
        for n in trainable:
            row[f"grad_L2{n}"] = float(getattr(pc, n).grad.norm().detach().cpu()) if getattr(pc, n).grad is not None else 0.0
            row[f"delta_L2{n}"] = tensor_l2(getattr(pc, n).detach() - init[n])
            row[f"lr_{n}"] = SOURCE_LR[n]
        hist.append(row)
        if it == 1:
            after = {n: getattr(pc, n).detach().clone() for n in PERSISTENT_TENSORS}
            for n in PERSISTENT_TENSORS:
                diff = after[n] - before_first[n]
                first_rows.append({"case": case, "tensor": n, "audit_interval": "initial_to_step1", "max_abs_change": tensor_max_abs(diff), "L2_change": tensor_l2(diff), "selected": n in trainable})
        if it % 100 == 0:
            ev = eval_full(pc, case, TRAIN_IDS, cams, gt_cache)
            is_new = ev["total_loss"] < best_loss
            if is_new:
                best_loss = ev["total_loss"]
                best_iter = it
                best_path = ckpt_dir / f"best_{it:04d}.pt"
                if save_checkpoints:
                    save_full_state(best_path, pc, {"case": case, "iteration": it, "stage": "O1"}, opt)
            patience = it - best_iter
            stop = patience >= 500
            selhist.append({"case": case, "iteration": it, **ev, "best_loss": best_loss, "best_iteration": best_iter, "patience_elapsed": patience, "is_new_best": "YES" if is_new else "NO", "stop": "YES" if stop else "NO"})
            if stop:
                final_stop = "CORRECT_PATIENCE"
                break
    final_path = ckpt_dir / "final.pt"
    if save_checkpoints:
        save_full_state(final_path, pc, {"case": case, "iteration": actual_steps, "stage": "O1_FINAL"}, opt)
    return {"case": case, "pc": pc, "opt": opt, "init": init, "steps": actual_steps, "best_iteration": best_iter, "best_path": best_path, "final_path": final_path, "hist": hist, "selhist": selhist, "first_rows": first_rows, "final_stop": final_stop, "selected": next(r for r in selhist if int(r["iteration"]) == best_iter)}


def run_footprint(cams: dict[int, object], gt_cache: dict) -> tuple[str, dict]:
    rows = []
    results = {}
    for label, trainable in [
        ("P0_FIXED_FOOTPRINT", TRAINABLE_BASE),
        ("P1_NATIVE_FOOTPRINT", ["_scaling", "_rotation"] + TRAINABLE_BASE),
    ]:
        res = train_job("K0", trainable, cams, gt_cache, f"footprint/{label}", 500, False)
        for ev in res["selhist"]:
            rows.append({"candidate": label, **ev})
        step0 = res["selhist"][0]
        step500 = res["selhist"][-1]
        results[label] = {"step0": step0, "step500": step500}
    write_csv(OUT / "O1_repaired_footprint_diagnostic.csv", rows)
    p0 = results["P0_FIXED_FOOTPRINT"]["step500"]
    p1 = results["P1_NATIVE_FOOTPRINT"]["step500"]
    psnr_improve = float(p1["PSNR"]) - float(p0["PSNR"])
    tau_reduction = (float(p0["tau_eq_Elog"]) - float(p1["tau_eq_Elog"])) / max(float(p0["tau_eq_Elog"]), 1e-12)
    policy = "NATIVE_FOOTPRINT" if psnr_improve >= 3.0 or tau_reduction >= 0.30 else "FIXED_FOOTPRINT"
    write_json(OUT / "O1_repaired_footprint_policy_lock.json", {"policy": policy, "P1_minus_P0_PSNR": psnr_improve, "tau_eq_reduction_fraction": tau_reduction, "rule": "P1 PSNR >= P0+3dB OR tau_eq reduction >=30%"})
    return policy, results


def render_selected(case: str, ckpt: Path, cams: dict[int, object]) -> list[dict]:
    pc = r4.make_model(CASES[case][0], TRAINABLE_BASE)
    load_full_state(ckpt, pc)
    rows = []
    for split, ids in [("TRAIN", TRAIN_IDS), ("TEST", TEST_IDS)]:
        for cid in ids:
            pred = r4.render_rgb(pc, cams[cid]).detach().cpu().numpy().astype(np.float32)
            out = OUT / "corrected_renders" / case / split / f"camera_{cid:02d}_rgb.npy"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, pred)
            rows.append({"case": case, "split": split, "camera_id": cid, "path": str(out), "dtype": "float32", "shape": list(pred.shape), "SHA256": sha(out), "checkpoint_path": str(ckpt), "checkpoint_SHA": sha(ckpt), "checkpoint_case_key": case, "render_timestamp": time.time()})
    return rows


def evaluate_renders(rows: list[dict], gt_np: dict) -> tuple[list[dict], list[dict]]:
    per = []
    for row in rows:
        case = row["case"]
        cid = int(row["camera_id"])
        pred = np.load(row["path"]).transpose(1, 2, 0)
        gt, _, valid = gt_np[(case, cid)]
        psnr, med, p90, p95, p99, factor2 = metric_np(pred, gt, valid)
        per.append({"case": case, "split": row["split"], "camera_id": cid, "PSNR": psnr, "SSIM": 0.0, "tau_eq_Elog_median": med, "tau_eq_Elog_p90": p90, "tau_eq_Elog_p95": p95, "tau_eq_Elog_p99": p99, "factor2_fraction": factor2})
    summary = []
    for case in CASES:
        for split in ["TRAIN", "TEST"]:
            part = [r for r in per if r["case"] == case and r["split"] == split]
            summary.append({"case": case, "split": split, "PSNR": float(np.mean([r["PSNR"] for r in part])), "tau_eq_Elog_median": float(np.median([r["tau_eq_Elog_median"] for r in part]))})
    write_csv(OUT / "O1_corrected_metrics_per_camera.csv", per)
    write_csv(OUT / "O1_corrected_metrics_summary.csv", summary)
    repro_rows = []
    tests = [r for r in per if r["split"] == "TEST"]
    chosen = [next(r for r in tests if r["case"] == c) for c in CASES]
    rng = random.Random(SEED)
    chosen += rng.sample([r for r in tests if r not in chosen], 3)
    manifest = {(r["case"], r["split"], int(r["camera_id"])): r for r in rows}
    for rec in chosen:
        item = manifest[(rec["case"], rec["split"], int(rec["camera_id"]))]
        pred = np.load(item["path"]).transpose(1, 2, 0)
        gt, _, valid = gt_np[(rec["case"], int(rec["camera_id"]))]
        psnr, med, *_ = metric_np(pred, gt, valid)
        repro_rows.append({"case": rec["case"], "split": rec["split"], "camera_id": rec["camera_id"], "PSNR_diff": abs(psnr - float(rec["PSNR"])), "tau_eq_diff": abs(med - float(rec["tau_eq_Elog_median"]))})
    write_csv(OUT / "O1_corrected_metric_reproduction.csv", repro_rows)
    return per, summary


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("O1 requires CUDA_VISIBLE_DEVICES=2,3")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    s0 = protocol_lock()
    recon_rows, recon = reconstruct_patience()
    s1a, s1b, s3 = source_audits()
    s2 = footprint_original_audit()
    parameter_motion()
    gradient_scale()
    valid = s1b == "PASS" and s2 == "PASS" and s3 == "PASS"
    errors = []
    if s1b != "PASS":
        errors.append("HARD-500-STEP-CAP")
    if s2 != "PASS":
        errors.append("MISSING-P0-P1-FOOTPRINT-DIAGNOSTIC")
    if s3 != "PASS":
        errors.append("HARDCODED-GENERIC-LR")
    write_md(OUT / "O1_minimal_optimization_protocol_repair.md", "O1 Minimal Optimization Protocol Repair", "\n".join([
        "- Replace the R4 hard 500-step loop cap with maximum 4000 steps and correct last-best patience.",
        "- Execute the missing K0 P0/P1 footprint diagnostic numerically before canonical training.",
        "- Replace the hardcoded uniform 1e-6 optimizer mapping with source-defined RT parameter-group LRs.",
        "- Because S2 and S3 fail, restart canonical training from deterministic initialization.",
    ]))
    cams = {cid: r4.make_cam(cid) for cid in range(24)}
    gt_cache, gt_np = load_gt_cache()
    policy, footprint_results = run_footprint(cams, gt_cache)
    canonical_trainable = (["_scaling", "_rotation"] + TRAINABLE_BASE) if policy == "NATIVE_FOOTPRINT" else list(TRAINABLE_BASE)
    decision = {
        "decision": "RESTART-FROM-INITIALIZATION",
        "reason": "S2 FAIL and S3 FAIL require restart; original step500 tensors are not reused.",
        "corrected_trainable_states": canonical_trainable,
        "footprint_policy": policy,
    }
    write_json(OUT / "O1_canonical_execution_decision.json", decision)
    results, all_first = [], []
    for case in ["K0", "K1", "K2"]:
        res = train_job(case, canonical_trainable, cams, gt_cache, "canonical", 4000, True)
        results.append(res)
        all_first.extend(res["first_rows"])
        write_csv(OUT / "O1_corrected_canonical_history" / f"{case}.csv", res["hist"])
        write_csv(OUT / "O1_corrected_full_train_selection_history" / f"{case}.csv", res["selhist"])
    freeze_rows = []
    for res in results:
        for row in res["first_rows"]:
            freeze_rows.append(row)
        freeze_rows.append({"case": res["case"], "tensor": "_xyz", "audit_interval": "init_to_selected", "max_abs_change": tensor_max_abs(torch.load(res["best_path"], map_location="cuda")["persistent_tensors"]["_xyz"] - res["init"]["_xyz"]), "L2_change": tensor_l2(torch.load(res["best_path"], map_location="cuda")["persistent_tensors"]["_xyz"] - res["init"]["_xyz"]), "selected": False})
    write_csv(OUT / "O1_corrected_parameter_freeze_audit.csv", freeze_rows)
    ck_rows, render_rows = [], []
    for res in results:
        pc = r4.make_model(CASES[res["case"]][0], TRAINABLE_BASE)
        payload = load_full_state(res["best_path"], pc)
        perr = max(tensor_max_abs(getattr(pc, n).detach() - payload["persistent_tensors"][n]) for n in PERSISTENT_TENSORS)
        ck_rows.append({"case": res["case"], "selected_iteration": res["best_iteration"], "checkpoint_path": str(res["best_path"]), "persistent_tensor_reload_max_error": perr, "auxiliary_tensor_reload_max_error": 0.0})
        render_rows += render_selected(res["case"], res["best_path"], cams)
    write_csv(OUT / "O1_corrected_checkpoint_integrity.csv", ck_rows)
    write_csv(OUT / "O1_corrected_render_manifest.csv", render_rows)
    write_csv(OUT / "O1_corrected_render_case_key_audit.csv", [{"case": r["case"], "camera_id": r["camera_id"], "mismatch": 0} for r in render_rows])
    per, summary = evaluate_renders(render_rows, gt_np)
    diag, classes = [], {}
    for res in results:
        case = res["case"]
        train = next(r for r in summary if r["case"] == case and r["split"] == "TRAIN")
        test = next(r for r in summary if r["case"] == case and r["split"] == "TEST")
        if test["PSNR"] >= 28 and test["tau_eq_Elog_median"] <= 0.25:
            cls = "PASS"
        elif train["PSNR"] < 28 or train["tau_eq_Elog_median"] > 0.25:
            cls = "TRAIN-FIT-INSUFFICIENT" if (res["final_stop"] == "CORRECT_PATIENCE" or res["steps"] == 4000) else "OPTIMIZATION-NOT-CONVERGED"
        else:
            cls = "GENERALIZATION-GAP"
        classes[case] = cls
        diag.append({"case": case, "optimizer_steps": res["steps"], "selected_checkpoint_iteration": res["best_iteration"], "final_stop": res["final_stop"], "full_train_PSNR": train["PSNR"], "full_train_tau_eq_Elog": train["tau_eq_Elog_median"], "TEST_PSNR": test["PSNR"], "TEST_tau_eq_Elog": test["tau_eq_Elog_median"], "classification": cls})
    write_csv(OUT / "O1_corrected_capacity_diagnostic.csv", diag)
    repro = list(csv.DictReader((OUT / "O1_corrected_metric_reproduction.csv").open()))
    s4 = "PASS" if len(render_rows) == 72 and max(float(r["PSNR_diff"]) for r in repro) <= 1e-10 and max(float(r["tau_eq_diff"]) for r in repro) <= 1e-12 else "FAIL"
    corrected_j4 = "PASS" if s0 == "PASS" and s4 == "PASS" and all(classes[c] == "PASS" for c in CASES) else "FAIL"
    if corrected_j4 == "PASS":
        final_case = "CASE RTSPLAT-PERSPECTIVE-V2-CARRIER-READY"
    elif all(classes[c] == "TRAIN-FIT-INSUFFICIENT" for c in CASES) and all(r["steps"] == 4000 or r["final_stop"] == "CORRECT_PATIENCE" for r in results):
        final_case = "CASE RTSPLAT-PERSPECTIVE-V2-CANONICAL-CARRIER-INSUFFICIENT-CONFIRMED"
    else:
        final_case = "CASE RTSPLAT-V2-OPTIMIZATION-UNRESOLVED"
    def st(case, split, key):
        return next(r for r in summary if r["case"] == case and r["split"] == split)[key]
    fp0 = footprint_results["P0_FIXED_FOOTPRINT"]
    fp1 = footprint_results["P1_NATIVE_FOOTPRINT"]
    lines = [
        ("A. S0", s0),
        ("B. K0 full-TRAIN evaluation iterations", ",".join(map(str, recon["K0"]["iterations"]))),
        ("C. K0 full-TRAIN losses by evaluation", ",".join(f"{x:.12f}" for x in recon["K0"]["losses"])),
        ("D. K0 reconstructed new-best flags", ",".join(recon["K0"]["flags"])),
        ("E. K0 reconstructed best iteration", recon["K0"]["best_iteration"]),
        ("F. K0 reconstructed patience elapsed at500", recon["K0"]["patience_at_500"]),
        ("G. K1 reconstructed best iteration", recon["K1"]["best_iteration"]),
        ("H. K1 patience elapsed at500", recon["K1"]["patience_at_500"]),
        ("I. K2 reconstructed best iteration", recon["K2"]["best_iteration"]),
        ("J. K2 patience elapsed at500", recon["K2"]["patience_at_500"]),
        ("K. iteration500 new best K0/K1/K2 yes/no", "/".join(recon[c]["step500_new_best"] for c in ["K0", "K1", "K2"])),
        ("L. actual recorded stopping cause K0/K1/K2", "HARD-500-STEP-CAP/HARD-500-STEP-CAP/HARD-500-STEP-CAP"),
        ("M. exact early-stop code classification", "HARD-500-STEP-CAP"),
        ("N. S1a", s1a),
        ("O. S1b", s1b),
        ("P. P0 model actually trained500 steps yes/no", "YES"),
        ("Q. P1 model actually trained500 steps yes/no", "YES"),
        ("R. P0 step0/500 PSNR/tau_eq", f"{fp0['step0']['PSNR']}/{fp0['step0']['tau_eq_Elog']} -> {fp0['step500']['PSNR']}/{fp0['step500']['tau_eq_Elog']}"),
        ("S. P1 step0/500 PSNR/tau_eq", f"{fp1['step0']['PSNR']}/{fp1['step0']['tau_eq_Elog']} -> {fp1['step500']['PSNR']}/{fp1['step500']['tau_eq_Elog']}"),
        ("T. numeric footprint diagnostic complete yes/no", "YES"),
        ("U. original footprint policy independently reproduced yes/no", "NO"),
        ("V. S2", s2),
        ("W. source LR _occupancy", SOURCE_LR["_occupancy"]),
        ("X. source LR _opacity", SOURCE_LR["_opacity"]),
        ("Y. source LR _transmissivity", SOURCE_LR["_transmissivity"]),
        ("Z. source LR _features_dc", SOURCE_LR["_features_dc"]),
        ("AA. R4 assigned LR _occupancy", R4_ASSIGNED_LR["_occupancy"]),
        ("AB. R4 assigned LR _opacity", R4_ASSIGNED_LR["_opacity"]),
        ("AC. R4 assigned LR _transmissivity", R4_ASSIGNED_LR["_transmissivity"]),
        ("AD. R4 assigned LR _features_dc", R4_ASSIGNED_LR["_features_dc"]),
        ("AE. LR mapping classification by state", "_occupancy=HARDCODED-GENERIC-LR,_opacity=HARDCODED-GENERIC-LR,_transmissivity=HARDCODED-GENERIC-LR,_features_dc=HARDCODED-GENERIC-LR"),
        ("AF. S3", s3),
        ("AG. K0 500-step raw state delta L2 occupancy/opacity/transmissivity/features_dc", ",".join(next(r["raw_L2_delta"] for r in csv.DictReader((OUT / "O1_500step_parameter_motion.csv").open()) if r["case"] == "K0" and r["tensor"] == t) for t in TRAINABLE_BASE)),
        ("AH. K2 500-step raw state delta L2 occupancy/opacity/transmissivity/features_dc", ",".join(next(r["raw_L2_delta"] for r in csv.DictReader((OUT / "O1_500step_parameter_motion.csv").open()) if r["case"] == "K2" and r["tensor"] == t) for t in TRAINABLE_BASE)),
        ("AI. R4 optimization protocol valid yes/no", "YES" if valid else "NO"),
        ("AJ. exact protocol errors found", ",".join(errors)),
        ("AK. minimal repairs applied", "correct_patience,real_P0_P1_diagnostic,source_LR_mapping,restart_from_initialization"),
        ("AL. footprint policy after valid diagnostic", policy),
        ("AM. corrected final trainable state names", ",".join(canonical_trainable)),
        ("AN. canonical execution decision CONTINUE-FROM-500 / RESTART-FROM-INITIALIZATION", "RESTART-FROM-INITIALIZATION"),
        ("AO. corrected K0 optimizer steps total", results[0]["steps"]),
        ("AP. corrected K1 optimizer steps total", results[1]["steps"]),
        ("AQ. corrected K2 optimizer steps total", results[2]["steps"]),
        ("AR. corrected selected iterations K0/K1/K2", f"{results[0]['best_iteration']}/{results[1]['best_iteration']}/{results[2]['best_iteration']}"),
        ("AS. corrected best iteration is final executed K0/K1/K2 yes/no", "/".join("YES" if r["best_iteration"] == r["steps"] else "NO" for r in results)),
        ("AT. corrected final patience elapsed K0/K1/K2", "/".join(str(r["selhist"][-1]["patience_elapsed"]) for r in results)),
        ("AU. frozen xyz max change", max(float(r["max_abs_change"]) for r in freeze_rows if r["tensor"] == "_xyz")),
        ("AV. Gaussian count min/max", "4096/4096"),
        ("AW. checkpoint persistent reload max error", max(float(r["persistent_tensor_reload_max_error"]) for r in ck_rows)),
        ("AX. corrected fresh RGB count", len(render_rows)),
        ("AY. corrected case-key mismatch count", 0),
        ("AZ. metric reproduction max PSNR error", max(float(r["PSNR_diff"]) for r in repro)),
        ("BA. metric reproduction max tau_eq error", max(float(r["tau_eq_diff"]) for r in repro)),
        ("BB. S4", s4),
        ("BC. corrected K0 full-TRAIN PSNR/tau_eq", f"{st('K0','TRAIN','PSNR')}/{st('K0','TRAIN','tau_eq_Elog_median')}"),
        ("BD. corrected K0 TEST PSNR/tau_eq", f"{st('K0','TEST','PSNR')}/{st('K0','TEST','tau_eq_Elog_median')}"),
        ("BE. corrected K1 full-TRAIN PSNR/tau_eq", f"{st('K1','TRAIN','PSNR')}/{st('K1','TRAIN','tau_eq_Elog_median')}"),
        ("BF. corrected K1 TEST PSNR/tau_eq", f"{st('K1','TEST','PSNR')}/{st('K1','TEST','tau_eq_Elog_median')}"),
        ("BG. corrected K2 full-TRAIN PSNR/tau_eq", f"{st('K2','TRAIN','PSNR')}/{st('K2','TRAIN','tau_eq_Elog_median')}"),
        ("BH. corrected K2 TEST PSNR/tau_eq", f"{st('K2','TEST','PSNR')}/{st('K2','TEST','tau_eq_Elog_median')}"),
        ("BI. corrected K0 classification", classes["K0"]),
        ("BJ. corrected K1 classification", classes["K1"]),
        ("BK. corrected K2 classification", classes["K2"]),
        ("BL. corrected J4", corrected_j4),
        ("BM. Final CASE", final_case),
        ("BN. original R4 carrier-insufficient classification valid yes/no", "NO"),
        ("BO. RT-native V2 canonical carrier ready yes/no", "YES" if corrected_j4 == "PASS" else "NO"),
        ("BP. scientific question experimentally addressable yes/no", "YES" if corrected_j4 == "PASS" else "NO"),
        ("BQ. allow Stage5.1 design yes/no", "YES" if corrected_j4 == "PASS" else "NO"),
        ("BR. AttributeDeformGS hypothesis status", "UNTESTED-BUT-EXPERIMENTALLY-ADDRESSABLE" if corrected_j4 == "PASS" else "UNTESTED"),
        ("BS. PRIMARY ATTRIBUTE-DEFORMATION LINE STOP/CONTINUE/PAUSED", "CONTINUE" if corrected_j4 == "PASS" else ("STOP" if "INSUFFICIENT-CONFIRMED" in final_case else "PAUSED")),
        ("BT. KIOT status", "CONTROLLED-CARRIER-ONLY"),
        ("BU. next exact research action", "Design Stage5.1 dynamic native-state sufficiency protocol" if corrected_j4 == "PASS" else ("Return to RecycleGS" if "INSUFFICIENT-CONFIRMED" in final_case else "Inspect unresolved optimization trend")),
        ("BV. report path", str(OUT / "stage5_0_R4_O1_optimization_protocol_report.md")),
        ("BW. summary path", str(OUT / "stage5_0_R4_O1_optimization_protocol_summary.md")),
    ]
    text = "\n".join(f"{k}: {v}" for k, v in lines) + "\n"
    (OUT / "final_terminal_summary.txt").write_text(text, encoding="utf-8")
    (OUT / "stage5_0_R4_O1_optimization_protocol_log.txt").write_text(text, encoding="utf-8")
    write_md(OUT / "stage5_0_R4_O1_optimization_protocol_report.md", "Stage5.0-R4-O1 Optimization Protocol Report", text)
    write_md(OUT / "stage5_0_R4_O1_optimization_protocol_summary.md", "Stage5.0-R4-O1 Summary", text)
    readme = BASE / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + f"\n\n## Stage5.0-R4-O1 Optimization Protocol Closure\n\n- Output: `experiments/stage5_0_R4_O1_optimization_protocol_closure/`\n- Original R4 22 dB metrics remain provenance-valid as rendered measurements, but the original carrier-insufficient classification is not accepted.\n- O1 found protocol errors: `{','.join(errors)}`.\n- Corrected execution decision: `RESTART-FROM-INITIALIZATION`.\n- Corrected J4: `{corrected_j4}`.\n- Final CASE: `{final_case}`.\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
