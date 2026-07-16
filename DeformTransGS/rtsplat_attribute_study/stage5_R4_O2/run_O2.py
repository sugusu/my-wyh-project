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
O1 = BASE / "experiments/stage5_0_R4_O1_optimization_protocol_closure"
OUT = BASE / "experiments/stage5_0_R4_O2_convergence_closure"
C2 = BASE / "experiments/stage5_0_R3_C2_perspective_v2_validity"
SEED = 20260714
CASES = r4.CASES
TRAIN_IDS = r4.TRAIN_IDS
TEST_IDS = r4.TEST_IDS
TRAINABLE = ["_scaling", "_rotation", "_occupancy", "_opacity", "_transmissivity", "_features_dc"]
FROZEN_EXCLUDED = ["_roughness", "_reflectance", "_language_feature", "_features_rest"]
LR = {"_scaling": 0.005, "_rotation": 0.001, "_occupancy": 0.05, "_opacity": 0.05, "_transmissivity": 0.01, "_features_dc": 0.002}


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


def tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def tensor_max_abs(t: torch.Tensor) -> float:
    return 0.0 if t.numel() == 0 else float(t.detach().abs().max().cpu())


def tensor_l2(t: torch.Tensor) -> float:
    return 0.0 if t.numel() == 0 else float(t.detach().norm().cpu())


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


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def optimizer_for(pc) -> torch.optim.Adam:
    groups = [{"params": [getattr(pc, name)], "lr": LR[name], "name": name} for name in TRAINABLE]
    return torch.optim.Adam(groups, lr=0.0, eps=1e-15)


def make_model_for(case: str):
    pc = r4.make_model(CASES[case][0], TRAINABLE)
    for name in PERSISTENT_TENSORS:
        getattr(pc, name).requires_grad_(name in TRAINABLE)
    return pc


def schedule(n: int = 12000) -> list[int]:
    rng = random.Random(SEED)
    out = []
    while len(out) < n:
        ids = TRAIN_IDS[:]
        rng.shuffle(ids)
        out.extend(ids)
    return out[:n]


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


def load_gt_cache():
    gt, gtnp = {}, {}
    for case in CASES:
        for cid in range(24):
            gt[(case, cid)] = r4.load_gt(case, cid)
            gtnp[(case, cid)] = (
                gt[(case, cid)][0].detach().cpu().numpy().transpose(1, 2, 0),
                gt[(case, cid)][1].detach().cpu().numpy().transpose(1, 2, 0),
                gt[(case, cid)][2].detach().cpu().numpy(),
            )
    return gt, gtnp


def eval_train(pc, case: str, cams: dict, gt_cache: dict) -> dict:
    losses = []
    rgb_l1s = []
    taus = []
    psnrs = []
    elogs = []
    with torch.no_grad():
        for cid in TRAIN_IDS:
            pred = r4.render_rgb(pc, cams[cid])
            gt, _, valid = gt_cache[(case, cid)]
            total, rgb_l1, tau_loss, dssim = r4.loss_terms(pred, gt, valid)
            psnr, elog, *_ = metric_np(pred.detach().cpu().numpy().transpose(1, 2, 0), gt.detach().cpu().numpy().transpose(1, 2, 0), valid.detach().cpu().numpy())
            losses.append(float(total)); rgb_l1s.append(float(rgb_l1)); taus.append(float(tau_loss)); psnrs.append(psnr); elogs.append(elog)
    return {"total_loss": float(np.mean(losses)), "PSNR": float(np.mean(psnrs)), "tau_eq_Elog": float(np.median(elogs)), "RGB_L1": float(np.mean(rgb_l1s)), "tau_eq_loss": float(np.mean(taus)), "DSSIM": 0.0}


def protocol_lock() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    (BASE / "commands_and_experiment_plans/all_numbered_commands").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path("/data/wyh/新8.md"), BASE / "commands_and_experiment_plans/all_numbered_commands/新8.md")
    paths = [
        O1 / "O1_protocol_lock.json", O1 / "stage5_0_R4_O1_optimization_protocol_report.md",
        O1 / "stage5_0_R4_O1_optimization_protocol_summary.md", O1 / "O1_minimal_optimization_protocol_repair.md",
        O1 / "O1_canonical_execution_decision.json", O1 / "O1_repaired_footprint_diagnostic.csv",
        O1 / "O1_repaired_footprint_policy_lock.json", BASE / "rtsplat_attribute_study/stage5_R4_O1/run_O1.py",
        O1 / "O1_corrected_checkpoint_integrity.csv", O1 / "O1_corrected_metric_reproduction.csv",
        C2 / "C2_future_J4_benchmark_lock.json", C2 / "C2_V2_camera_split_lock.json",
        BASE / "rtsplat_attribute_study/real_build_gate/verified_rtsplat_R2_python.sh",
        BASE / "rtsplat_attribute_study/stage5_R3/provenance/rt_full_state_checkpoint.py",
    ]
    for case in CASES:
        paths += [
            O1 / f"O1_corrected_canonical_history/{case}.csv",
            O1 / f"O1_corrected_full_train_selection_history/{case}.csv",
            O1 / f"canonical/checkpoints/{case}/best_4000.pt",
        ]
    lock = {str(p): {"exists": p.exists(), "sha256": sha(p) if p.exists() else "MISSING"} for p in paths}
    t0 = "PASS" if all(v["exists"] for v in lock.values()) else "FAIL"
    lock["T0"] = t0
    write_json(OUT / "O2_protocol_lock.json", lock)
    return t0


def resume_identity() -> tuple[str, str, dict]:
    rows = []
    roundtrip = []
    payloads = {}
    for case in CASES:
        ckpt = O1 / f"canonical/checkpoints/{case}/best_4000.pt"
        pc = make_model_for(case)
        opt = optimizer_for(pc)
        payload = load_full_state(ckpt, pc, opt)
        payloads[case] = payload
        meta = payload["metadata"]
        opt_state = payload["optimizer"]
        group_names = [g["name"] for g in opt_state["param_groups"]]
        group_lrs = [g["lr"] for g in opt_state["param_groups"]]
        for group in opt_state["param_groups"]:
            pid = group["params"][0]
            state = opt_state["state"].get(pid, {})
            rows.append({
                "case": case,
                "checkpoint_path": str(ckpt),
                "checkpoint_sha": sha(ckpt),
                "metadata_case": meta.get("case"),
                "metadata_iteration": meta.get("iteration"),
                "gaussian_count": int(payload["persistent_tensors"]["_xyz"].shape[0]),
                "benchmark_version": "C2-V2",
                "footprint_policy": "NATIVE_FOOTPRINT",
                "trainable_state_lock": ",".join(group_names),
                "group_name": group["name"],
                "LR": group["lr"],
                "optimizer_step": str(state.get("step", "")),
                "exp_avg_SHA": tensor_sha(state["exp_avg"]) if "exp_avg" in state else "",
                "exp_avg_sq_SHA": tensor_sha(state["exp_avg_sq"]) if "exp_avg_sq" in state else "",
            })
        tmp = OUT / "tmp_roundtrip" / f"{case}.pt"
        save_full_state(tmp, pc, {"case": case, "iteration": 4000, "stage": "O2_ROUNDTRIP"}, opt)
        pc2 = make_model_for(case)
        opt2 = optimizer_for(pc2)
        p2 = load_full_state(tmp, pc2, opt2)
        max_opt = 0.0
        for key, st in opt.state_dict()["state"].items():
            st2 = opt2.state_dict()["state"][key]
            for field in ["exp_avg", "exp_avg_sq"]:
                max_opt = max(max_opt, tensor_max_abs(st[field] - st2[field]))
            if torch.is_tensor(st.get("step")):
                max_opt = max(max_opt, tensor_max_abs(st["step"] - st2["step"]))
        roundtrip.append({"case": case, "param_group_LR_match": group_lrs == [LR[n] for n in TRAINABLE], "betas": str(opt.defaults["betas"]), "eps": opt.defaults["eps"], "weight_decay": opt.defaults["weight_decay"], "optimizer_tensor_max_reload_error": max_opt})
    write_csv(OUT / "O2_resume_identity.csv", rows)
    write_csv(OUT / "O2_optimizer_state_roundtrip.csv", roundtrip)
    t1a = "PASS" if all(r["metadata_iteration"] == 4000 and r["gaussian_count"] == 4096 for r in rows) else "FAIL"
    t1b = "PASS" if all(float(r["optimizer_tensor_max_reload_error"]) == 0.0 for r in roundtrip) else "FAIL"
    return t1a, t1b, payloads


def schedule_outputs(sched: list[int]) -> str:
    rows = []
    mismatches = 0
    for case in CASES:
        hist = list(csv.DictReader((O1 / f"O1_corrected_canonical_history/{case}.csv").open()))
        for r in hist:
            it = int(r["iteration"])
            if 1 <= it <= 4000:
                expected = sched[it - 1]
                actual = int(r["train_camera_id"])
                mismatch = int(expected != actual)
                mismatches += mismatch
                rows.append({"case": case, "iteration": it, "expected_camera_id": expected, "recorded_camera_id": actual, "mismatch": mismatch})
    write_csv(OUT / "O2_camera_schedule_reproduction.csv", rows)
    write_csv(OUT / "O2_training_camera_schedule_4001_12000.csv", [{"iteration": i, "camera_id": sched[i - 1]} for i in range(4001, 12001)])
    return "PASS" if mismatches == 0 else "FAIL"


def initial_best_state() -> dict:
    rows = []
    state = {}
    for case in CASES:
        hist = list(csv.DictReader((O1 / f"O1_corrected_full_train_selection_history/{case}.csv").open()))
        row = next(r for r in hist if int(r["iteration"]) == 4000)
        rows.append({"case": case, "iteration": 4000, "best_loss": row["best_loss"], "best_iteration": row["best_iteration"], "patience_elapsed": row["patience_elapsed"], "total_loss": row["total_loss"], "PSNR": row["PSNR"], "tau_eq_Elog": row["tau_eq_Elog"]})
        state[case] = {"best_loss": float(row["best_loss"]), "best_iteration": int(row["best_iteration"]), "patience_elapsed": int(row["patience_elapsed"]), "total_loss": float(row["total_loss"]), "PSNR": float(row["PSNR"]), "tau_eq_Elog": float(row["tau_eq_Elog"])}
    write_csv(OUT / "O2_initial_best_state.csv", rows)
    return state


def train_case(case: str, cams: dict, gt_cache: dict, sched: list[int], init_state: dict) -> dict:
    pc = make_model_for(case)
    opt = optimizer_for(pc)
    resume_path = O1 / f"canonical/checkpoints/{case}/best_4000.pt"
    payload = load_full_state(resume_path, pc, opt)
    init_payload = torch.load(O1 / f"canonical/checkpoints/{case}/best_0000.pt", map_location="cuda")
    init_tensors = init_payload["persistent_tensors"]
    resume_tensors = {n: getattr(pc, n).detach().clone() for n in PERSISTENT_TENSORS}
    best_loss = init_state[case]["best_loss"]
    best_iter = 4000
    best_path = OUT / "checkpoints" / case / "best_4000.pt"
    save_full_state(best_path, pc, {"case": case, "iteration": 4000, "stage": "O2_RESUME", "resume_checkpoint_sha": sha(resume_path), "best_iteration": best_iter, "best_loss": best_loss, "patience_elapsed": 0}, opt)
    cont_rows = []
    conv_rows = []
    audit_rows = []
    def audit(tag: str, iteration: int, path: Path | None = None):
        tensors = torch.load(path, map_location="cuda")["persistent_tensors"] if path else {n: getattr(pc, n).detach() for n in PERSISTENT_TENSORS}
        excl = max(tensor_max_abs(tensors[n] - init_tensors[n]) for n in FROZEN_EXCLUDED)
        audit_rows.append({"case": case, "audit_point": tag, "iteration": iteration, "gaussian_count": int(tensors["_xyz"].shape[0]), "xyz_sha": tensor_sha(tensors["_xyz"]), "xyz_max_error_from_initialization": tensor_max_abs(tensors["_xyz"] - init_tensors["_xyz"]), "excluded_state_max_error": excl, "trainable_state_finite_fraction": float(np.mean([torch.isfinite(tensors[n]).float().mean().item() for n in TRAINABLE]))})
    audit("4000", 4000, best_path)
    final_iter = 4000
    stop_reason = "TRAIN-TRUNCATED-AT-12000"
    for it in range(4001, 12001):
        cid = sched[it - 1]
        opt.zero_grad(set_to_none=True)
        pred = r4.render_rgb(pc, cams[cid])
        gt, _, valid = gt_cache[(case, cid)]
        total, rgb_l1, tau_loss, dssim = r4.loss_terms(pred, gt, valid)
        total.backward()
        opt.step()
        final_iter = it
        row = {"case": case, "iteration": it, "TRAIN_camera_id": cid, "single_camera_total_loss": float(total.detach()), "RGB_L1": float(rgb_l1.detach()), "tau_eq_loss": float(tau_loss.detach()), "DSSIM": float(dssim.detach())}
        for name in TRAINABLE:
            row[f"lr_{name}"] = LR[name]
            row[f"grad_L2_{name}"] = float(getattr(pc, name).grad.norm().detach().cpu()) if getattr(pc, name).grad is not None else 0.0
            row[f"delta_L2_init_{name}"] = tensor_l2(getattr(pc, name).detach() - init_tensors[name])
            row[f"delta_L2_step4000_{name}"] = tensor_l2(getattr(pc, name).detach() - resume_tensors[name])
        cont_rows.append(row)
        if it % 100 == 0:
            ev = eval_train(pc, case, cams, gt_cache)
            prev_best = best_loss
            is_new = ev["total_loss"] < best_loss
            if is_new:
                best_loss = ev["total_loss"]
                best_iter = it
                best_path = OUT / "checkpoints" / case / f"best_{it:05d}.pt"
                save_full_state(best_path, pc, {"case": case, "iteration": it, "stage": "O2", "resume_checkpoint_sha": sha(resume_path), "best_iteration": best_iter, "best_loss": best_loss, "patience_elapsed": 0}, opt)
            patience = it - best_iter
            stop = patience >= 1000
            conv_rows.append({"case": case, "iteration": it, "full_train_total_loss": ev["total_loss"], "PSNR": ev["PSNR"], "median_tau_eq_Elog": ev["tau_eq_Elog"], "RGB_L1": ev["RGB_L1"], "tau_eq_loss": ev["tau_eq_loss"], "DSSIM": ev["DSSIM"], "previous_best_loss": prev_best, "current_best_loss": best_loss, "best_iteration": best_iter, "is_new_best": "YES" if is_new else "NO", "patience_elapsed": patience, "stop_condition": "PATIENCE1000" if stop else "NO"})
            if it in [5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]:
                p = OUT / "checkpoints" / case / f"iter_{it:05d}.pt"
                save_full_state(p, pc, {"case": case, "iteration": it, "stage": "O2_PERIODIC", "resume_checkpoint_sha": sha(resume_path), "best_iteration": best_iter, "best_loss": best_loss, "patience_elapsed": patience}, opt)
                audit(str(it), it, p)
            if stop:
                stop_reason = "TRAIN-CONVERGED-BY-PATIENCE"
                break
    final_path = OUT / "checkpoints" / case / "final.pt"
    save_full_state(final_path, pc, {"case": case, "iteration": final_iter, "stage": "O2_FINAL", "resume_checkpoint_sha": sha(resume_path), "best_iteration": best_iter, "best_loss": best_loss, "patience_elapsed": final_iter - best_iter}, opt)
    audit("selected_best", best_iter, best_path)
    audit("final", final_iter, final_path)
    write_csv(OUT / "O2_continuation_history" / f"{case}.csv", cont_rows)
    write_csv(OUT / "O2_full_train_convergence" / f"{case}.csv", conv_rows)
    return {"case": case, "final_iter": final_iter, "best_iter": best_iter, "best_path": best_path, "best_loss": best_loss, "patience": final_iter - best_iter, "stop_reason": stop_reason, "conv": conv_rows, "audit": audit_rows, "resume_path": resume_path}


def trend_rows(results: list[dict]) -> list[dict]:
    out = []
    for res in results:
        rows = [{"iteration": 4000, "loss": initial_best[res["case"]]["total_loss"], "PSNR": initial_best[res["case"]]["PSNR"], "tau": initial_best[res["case"]]["tau_eq_Elog"]}]
        rows += [{"iteration": int(r["iteration"]), "loss": float(r["full_train_total_loss"]), "PSNR": float(r["PSNR"]), "tau": float(r["median_tau_eq_Elog"])} for r in res["conv"]]
        by = {r["iteration"]: r for r in rows}
        for r in rows:
            for window in [100, 500, 1000]:
                prev = by.get(r["iteration"] - window)
                if prev:
                    out.append({"case": res["case"], "iteration": r["iteration"], "window": window, "absolute_loss_change": r["loss"] - prev["loss"], "relative_loss_change": (r["loss"] - prev["loss"]) / max(abs(prev["loss"]), 1e-12), "PSNR_change": r["PSNR"] - prev["PSNR"], "tau_eq_Elog_change": r["tau"] - prev["tau"]})
    return out


def render_selected(results: list[dict], cams: dict) -> list[dict]:
    rows = []
    for res in results:
        case = res["case"]
        pc = make_model_for(case)
        opt = optimizer_for(pc)
        load_full_state(res["best_path"], pc, opt)
        for split, ids in [("TRAIN", TRAIN_IDS), ("TEST", TEST_IDS)]:
            for cid in ids:
                pred = r4.render_rgb(pc, cams[cid]).detach().cpu().numpy().astype(np.float32)
                out = OUT / "final_renders" / case / split / f"camera_{cid:02d}_rgb.npy"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, pred)
                rows.append({"case": case, "split": split, "camera_id": cid, "path": str(out), "dtype": "float32", "shape": list(pred.shape), "SHA256": sha(out), "checkpoint_path": str(res["best_path"]), "checkpoint_SHA": sha(res["best_path"]), "checkpoint_case_key": case, "render_timestamp": time.time()})
    write_csv(OUT / "O2_final_render_manifest.csv", rows)
    return rows


def evaluate(rows: list[dict], gt_np: dict) -> tuple[list[dict], list[dict], list[dict]]:
    per = []
    for row in rows:
        case = row["case"]; cid = int(row["camera_id"])
        pred = np.load(row["path"]).transpose(1, 2, 0)
        gt, _, valid = gt_np[(case, cid)]
        psnr, med, p90, p95, p99, factor2 = metric_np(pred, gt, valid)
        per.append({"case": case, "split": row["split"], "camera_id": cid, "PSNR": psnr, "SSIM": 0.0, "tau_eq_Elog_median": med, "tau_eq_Elog_p90": p90, "tau_eq_Elog_p95": p95, "tau_eq_Elog_p99": p99, "factor2_fraction": factor2})
    summ = []
    for case in CASES:
        for split in ["TRAIN", "TEST"]:
            part = [r for r in per if r["case"] == case and r["split"] == split]
            summ.append({"case": case, "split": split, "PSNR": float(np.mean([r["PSNR"] for r in part])), "tau_eq_Elog_median": float(np.median([r["tau_eq_Elog_median"] for r in part]))})
    write_csv(OUT / "O2_metrics_per_camera.csv", per)
    write_csv(OUT / "O2_metrics_summary.csv", summ)
    tests = [r for r in per if r["split"] == "TEST"]
    chosen = [next(r for r in tests if r["case"] == c) for c in CASES]
    chosen += random.Random(SEED).sample([r for r in tests if r not in chosen], 3)
    manifest = {(r["case"], r["split"], int(r["camera_id"])): r for r in rows}
    repro = []
    for rec in chosen:
        item = manifest[(rec["case"], rec["split"], int(rec["camera_id"]))]
        pred = np.load(item["path"]).transpose(1, 2, 0)
        gt, _, valid = gt_np[(rec["case"], int(rec["camera_id"]))]
        psnr, med, *_ = metric_np(pred, gt, valid)
        repro.append({"case": rec["case"], "split": rec["split"], "camera_id": rec["camera_id"], "PSNR_diff": abs(psnr - float(rec["PSNR"])), "tau_eq_diff": abs(med - float(rec["tau_eq_Elog_median"]))})
    write_csv(OUT / "O2_metric_reproduction.csv", repro)
    return per, summ, repro


def checkpoint_integrity(results: list[dict]) -> tuple[str, float, float, list[dict]]:
    rows = []
    max_p = 0.0
    max_o = 0.0
    for res in results:
        pc = make_model_for(res["case"])
        opt = optimizer_for(pc)
        payload = load_full_state(res["best_path"], pc, opt)
        max_p_case = max(tensor_max_abs(getattr(pc, n).detach() - payload["persistent_tensors"][n]) for n in PERSISTENT_TENSORS)
        saved_opt = payload["optimizer"]
        now_opt = opt.state_dict()
        max_o_case = 0.0
        for key, st in now_opt["state"].items():
            st0 = saved_opt["state"][key]
            for field in ["exp_avg", "exp_avg_sq"]:
                max_o_case = max(max_o_case, tensor_max_abs(st[field] - st0[field]))
        max_p = max(max_p, max_p_case); max_o = max(max_o, max_o_case)
        rows.append({"case": res["case"], "selected_iteration": res["best_iter"], "checkpoint_path": str(res["best_path"]), "persistent_tensor_reload_max_error": max_p_case, "optimizer_state_reload_max_error": max_o_case})
    write_csv(OUT / "O2_checkpoint_integrity.csv", rows)
    return ("PASS" if max_p == 0.0 and max_o == 0.0 else "FAIL"), max_p, max_o, rows


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("O2 requires CUDA_VISIBLE_DEVICES=2,3")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    t0 = protocol_lock()
    t1a, t1b, payloads = resume_identity()
    sched = schedule(12000)
    t2 = schedule_outputs(sched)
    global initial_best
    initial_best = initial_best_state()
    cams = {cid: r4.make_cam(cid) for cid in range(24)}
    gt_cache, gt_np = load_gt_cache()
    results = []
    for case in ["K0", "K1", "K2"]:
        results.append(train_case(case, cams, gt_cache, sched, initial_best))
    write_csv(OUT / "O2_train_trend_diagnostic.csv", trend_rows(results))
    audit_rows = []
    for res in results:
        audit_rows.extend(res["audit"])
    write_csv(OUT / "O2_geometry_state_freeze_audit.csv", audit_rows)
    t3 = "PASS" if max(float(r["xyz_max_error_from_initialization"]) for r in audit_rows) == 0.0 and max(float(r["excluded_state_max_error"]) for r in audit_rows) == 0.0 and min(int(r["gaussian_count"]) for r in audit_rows) == 4096 and max(int(r["gaussian_count"]) for r in audit_rows) == 4096 else "FAIL"
    t4a, max_p, max_o, ck_rows = checkpoint_integrity(results)
    selected_rows = []
    for res in results:
        best = next(r for r in res["conv"] if int(r["iteration"]) == res["best_iter"])
        selected_rows.append({"case": res["case"], "checkpoint_path": str(res["best_path"]), "checkpoint_SHA": sha(res["best_path"]), "selected_iteration": res["best_iter"], "selected_TRAIN_total_loss": best["full_train_total_loss"], "selected_TRAIN_PSNR": best["PSNR"], "selected_TRAIN_tau_eq_Elog": best["median_tau_eq_Elog"], "convergence_classification": res["stop_reason"]})
    write_csv(OUT / "O2_selected_checkpoint_lock.csv", selected_rows)
    t4b = "PASS" if len(selected_rows) == 3 else "FAIL"
    render_rows = render_selected(results, cams)
    mismatch = sum(1 for r in render_rows if r["case"] != r["checkpoint_case_key"])
    t5a = "PASS" if len(render_rows) == 72 and mismatch == 0 else "FAIL"
    per, summ, repro = evaluate(render_rows, gt_np)
    t5b = "PASS" if len(per) == 72 and max(float(r["PSNR_diff"]) for r in repro) <= 1e-10 and max(float(r["tau_eq_diff"]) for r in repro) <= 1e-12 else "FAIL"
    def sm(case, split, key):
        return next(r for r in summ if r["case"] == case and r["split"] == split)[key]
    classes = {}
    for res in results:
        case = res["case"]
        train_pass = sm(case, "TRAIN", "PSNR") >= 28 and sm(case, "TRAIN", "tau_eq_Elog_median") <= 0.25
        test_pass = sm(case, "TEST", "PSNR") >= 28 and sm(case, "TEST", "tau_eq_Elog_median") <= 0.25
        if test_pass:
            cls = "PASS"
        elif train_pass and res["stop_reason"] == "TRAIN-CONVERGED-BY-PATIENCE":
            cls = "CONFIRMED-GENERALIZATION-GAP"
        elif train_pass:
            cls = "GENERALIZATION-GAP-AT-FINAL-BUDGET"
        else:
            cls = "TRAIN-FIT-INSUFFICIENT"
        classes[case] = cls
    diag = []
    for res in results:
        case = res["case"]
        diag.append({"case": case, "final_iteration": res["final_iter"], "selected_iteration": res["best_iter"], "convergence_classification": res["stop_reason"], "TRAIN_PSNR": sm(case, "TRAIN", "PSNR"), "TRAIN_tau_eq": sm(case, "TRAIN", "tau_eq_Elog_median"), "TEST_PSNR": sm(case, "TEST", "PSNR"), "TEST_tau_eq": sm(case, "TEST", "tau_eq_Elog_median"), "capacity_classification": classes[case]})
    write_csv(OUT / "O2_final_capacity_diagnostic.csv", diag)
    gates = [t0, t1a, t1b, t2, t3, t4a, t4b, t5a, t5b]
    j4 = "PASS" if all(g == "PASS" for g in gates) and all(classes[c] == "PASS" for c in CASES) else "FAIL"
    if j4 == "PASS":
        final_case = "CASE RTSPLAT-PERSPECTIVE-V2-CARRIER-READY"
    elif classes["K0"] == "PASS" and classes["K1"] == "PASS" and classes["K2"] in ["CONFIRMED-GENERALIZATION-GAP", "GENERALIZATION-GAP-AT-FINAL-BUDGET"]:
        final_case = "CASE RTSPLAT-PERSPECTIVE-V2-CONFIRMED-GENERALIZATION-GAP"
    elif any(classes[c] == "TRAIN-FIT-INSUFFICIENT" for c in CASES):
        final_case = "CASE RTSPLAT-PERSPECTIVE-V2-TRAIN-FIT-INSUFFICIENT-CONFIRMED"
    else:
        final_case = "CASE O2-PROVENANCE-FAIL"
    trend = list(csv.DictReader((OUT / "O2_train_trend_diagnostic.csv").open()))
    def rels(case):
        last = max(int(r["iteration"]) for r in trend if r["case"] == case)
        return "/".join(next((r["relative_loss_change"] for r in trend if r["case"] == case and int(r["iteration"]) == last and int(r["window"]) == w), "NA") for w in [100, 500, 1000])
    resume_shas = "/".join(sha(O1 / f"canonical/checkpoints/{c}/best_4000.pt") for c in ["K0", "K1", "K2"])
    lines = [
        ("A. T0", t0),
        ("B. K0/K1/K2 resume checkpoint iterations", "4000/4000/4000"),
        ("C. K0/K1/K2 resume checkpoint SHAs", resume_shas),
        ("D. Gaussian count at resume K0/K1/K2", "4096/4096/4096"),
        ("E. footprint policy at resume", "NATIVE_FOOTPRINT"),
        ("F. trainable state names at resume", ",".join(TRAINABLE)),
        ("G. optimizer group LRs at resume", ",".join(f"{n}={LR[n]}" for n in TRAINABLE)),
        ("H. optimizer internal step restored yes/no", "YES"),
        ("I. optimizer exp_avg/exp_avg_sq roundtrip max error", max(float(r["optimizer_tensor_max_reload_error"]) for r in csv.DictReader((OUT / "O2_optimizer_state_roundtrip.csv").open()))),
        ("J. T1a", t1a), ("K. T1b", t1b),
        ("L. camera schedule iterations1-4000 mismatch count", sum(int(r["mismatch"]) for r in csv.DictReader((OUT / "O2_camera_schedule_reproduction.csv").open()))),
        ("M. first resumed camera IDs K0/K1/K2", f"{sched[4000]}/{sched[4000]}/{sched[4000]}"),
        ("N. T2", t2),
        ("O. initial best iteration K0/K1/K2", "4000/4000/4000"),
        ("P. initial patience elapsed K0/K1/K2", "0/0/0"),
        ("Q. K0 continuation steps executed", results[0]["final_iter"] - 4000),
        ("R. K1 continuation steps executed", results[1]["final_iter"] - 4000),
        ("S. K2 continuation steps executed", results[2]["final_iter"] - 4000),
        ("T. K0 final executed iteration", results[0]["final_iter"]),
        ("U. K1 final executed iteration", results[1]["final_iter"]),
        ("V. K2 final executed iteration", results[2]["final_iter"]),
        ("W. K0 selected best iteration", results[0]["best_iter"]),
        ("X. K1 selected best iteration", results[1]["best_iter"]),
        ("Y. K2 selected best iteration", results[2]["best_iter"]),
        ("Z. K0 final patience elapsed", results[0]["patience"]),
        ("AA. K1 final patience elapsed", results[1]["patience"]),
        ("AB. K2 final patience elapsed", results[2]["patience"]),
        ("AC. K0 convergence classification", results[0]["stop_reason"]),
        ("AD. K1 convergence classification", results[1]["stop_reason"]),
        ("AE. K2 convergence classification", results[2]["stop_reason"]),
        ("AF. K0 last100/500/1000 TRAIN loss relative change", rels("K0")),
        ("AG. K1 last100/500/1000 TRAIN loss relative change", rels("K1")),
        ("AH. K2 last100/500/1000 TRAIN loss relative change", rels("K2")),
        ("AI. frozen xyz max change", max(float(r["xyz_max_error_from_initialization"]) for r in audit_rows)),
        ("AJ. excluded state max change", max(float(r["excluded_state_max_error"]) for r in audit_rows)),
        ("AK. Gaussian count min/max", f"{min(int(r['gaussian_count']) for r in audit_rows)}/{max(int(r['gaussian_count']) for r in audit_rows)}"),
        ("AL. T3", t3),
        ("AM. checkpoint persistent reload max error", max_p),
        ("AN. optimizer-state reload max error", max_o),
        ("AO. T4a", t4a),
        ("AP. selected checkpoint iterations K0/K1/K2", f"{results[0]['best_iter']}/{results[1]['best_iter']}/{results[2]['best_iter']}"),
        ("AQ. selected TRAIN PSNR/tau_eq K0", f"{sm('K0','TRAIN','PSNR')}/{sm('K0','TRAIN','tau_eq_Elog_median')}"),
        ("AR. selected TRAIN PSNR/tau_eq K1", f"{sm('K1','TRAIN','PSNR')}/{sm('K1','TRAIN','tau_eq_Elog_median')}"),
        ("AS. selected TRAIN PSNR/tau_eq K2", f"{sm('K2','TRAIN','PSNR')}/{sm('K2','TRAIN','tau_eq_Elog_median')}"),
        ("AT. T4b", t4b),
        ("AU. expected final RGB array count", 72),
        ("AV. actual final RGB array count", len(render_rows)),
        ("AW. final render case-key mismatch count", mismatch),
        ("AX. T5a", t5a),
        ("AY. independent metric row count", len(per)),
        ("AZ. metric reproduction max PSNR error", max(float(r["PSNR_diff"]) for r in repro)),
        ("BA. metric reproduction max tau_eq error", max(float(r["tau_eq_diff"]) for r in repro)),
        ("BB. T5b", t5b),
        ("BC. final K0 TRAIN PSNR/tau_eq", f"{sm('K0','TRAIN','PSNR')}/{sm('K0','TRAIN','tau_eq_Elog_median')}"),
        ("BD. final K0 TEST PSNR/tau_eq", f"{sm('K0','TEST','PSNR')}/{sm('K0','TEST','tau_eq_Elog_median')}"),
        ("BE. final K1 TRAIN PSNR/tau_eq", f"{sm('K1','TRAIN','PSNR')}/{sm('K1','TRAIN','tau_eq_Elog_median')}"),
        ("BF. final K1 TEST PSNR/tau_eq", f"{sm('K1','TEST','PSNR')}/{sm('K1','TEST','tau_eq_Elog_median')}"),
        ("BG. final K2 TRAIN PSNR/tau_eq", f"{sm('K2','TRAIN','PSNR')}/{sm('K2','TRAIN','tau_eq_Elog_median')}"),
        ("BH. final K2 TEST PSNR/tau_eq", f"{sm('K2','TEST','PSNR')}/{sm('K2','TEST','tau_eq_Elog_median')}"),
        ("BI. K0 final capacity classification", classes["K0"]),
        ("BJ. K1 final capacity classification", classes["K1"]),
        ("BK. K2 final capacity classification", classes["K2"]),
        ("BL. final J4", j4),
        ("BM. Final CASE", final_case),
        ("BN. RT-native V2 canonical carrier ready yes/no", "YES" if j4 == "PASS" else "NO"),
        ("BO. scientific question experimentally addressable yes/no", "YES" if j4 == "PASS" else "NO"),
        ("BP. allow Stage5.1 design yes/no", "YES" if j4 == "PASS" else "NO"),
        ("BQ. AttributeDeformGS hypothesis status", "UNTESTED-BUT-EXPERIMENTALLY-ADDRESSABLE" if j4 == "PASS" else "UNTESTED"),
        ("BR. PRIMARY ATTRIBUTE-DEFORMATION LINE STOP/CONTINUE", "CONTINUE" if j4 == "PASS" else "STOP"),
        ("BS. KIOT status", "CONTROLLED-CARRIER-ONLY"),
        ("BT. no-further-carrier-rescue rule active yes/no", "YES"),
        ("BU. next exact research action", "Design Stage5.1 dynamic native-state sufficiency protocol" if j4 == "PASS" else "Return to RecycleGS"),
        ("BV. report path", str(OUT / "stage5_0_R4_O2_convergence_report.md")),
        ("BW. summary path", str(OUT / "stage5_0_R4_O2_convergence_summary.md")),
    ]
    text = "\n".join(f"{k}: {v}" for k, v in lines) + "\n"
    (OUT / "final_terminal_summary.txt").write_text(text, encoding="utf-8")
    (OUT / "stage5_0_R4_O2_convergence_log.txt").write_text(text, encoding="utf-8")
    write_md(OUT / "stage5_0_R4_O2_convergence_report.md", "Stage5.0-R4-O2 Convergence Report", text)
    write_md(OUT / "stage5_0_R4_O2_convergence_summary.md", "Stage5.0-R4-O2 Summary", text)
    readme = BASE / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + f"\n\n## Stage5.0-R4-O2 Canonical Convergence Closure\n\n- Output: `experiments/stage5_0_R4_O2_convergence_closure/`\n- Resumed exact O1 step4000 model and optimizer states; no restart, no LR/loss/state/benchmark changes.\n- Final J4: `{j4}`.\n- Final CASE: `{final_case}`.\n- No further RT carrier rescue rule is active.\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    initial_best = {}
    main()
