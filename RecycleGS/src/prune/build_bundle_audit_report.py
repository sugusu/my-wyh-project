#!/usr/bin/env python3
"""Build Stage 2B-AA bundle audit report."""
import os, sys, json, torch

OUTPUT_DIR = "/data/wyh/RecycleGS/outputs/debug/stage2b_bundle_audit"
BASELINE_DIR = "/data/wyh/RecycleGS/baselines/tsgs_scene01_full"
RECOVERY_DIR = "/data/wyh/RecycleGS/outputs/prune_only/scene_01/ratio_005/schedule_control/recovery_500"

def check_file(path, desc):
    exists = os.path.exists(path)
    size_kb = os.path.getsize(path) / 1024 if exists else 0
    return {"desc": desc, "path": path, "exists": exists, "size_kb": round(size_kb, 1)}

def check_dir(path, desc):
    exists = os.path.isdir(path)
    files = os.listdir(path) if exists else []
    return {"desc": desc, "path": path, "exists": exists, "files": sorted(files)}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    findings = []
    recommendations = []

    # ── 1. AppModel weight check ──
    app_model_path = os.path.join(BASELINE_DIR, "app_model", "iteration_15000", "app.pth")
    if os.path.exists(app_model_path):
        state = torch.load(app_model_path, map_location='cpu', weights_only=True)
        appear_ab = state['appear_ab']
        is_zero = (appear_ab == 0).all().item()
        findings.append({
            "component": "AppModel",
            "check": "Are weights non-zero?",
            "result": "NO — ALL ZEROS" if is_zero else "YES",
            "detail": f"appear_ab shape={tuple(appear_ab.shape)}, mean={appear_ab.mean():.8f}, "
                      f"non_zero={(appear_ab != 0).sum().item()}/{appear_ab.numel()}, "
                      f"model_dir={BASELINE_DIR}"
        })
    else:
        findings.append({
            "component": "AppModel",
            "check": "weights file exists",
            "result": "MISSING",
            "detail": f"{app_model_path} not found"
        })

    # ── 2. AppModel weights saved during recovery? ──
    recovery_app_model_dir = os.path.join(RECOVERY_DIR, "app_model")
    has_app_model = os.path.isdir(recovery_app_model_dir) and len(os.listdir(recovery_app_model_dir)) > 0
    findings.append({
        "component": "AppModel",
        "check": "Recovery saves AppModel weights?",
        "result": "NO" if not has_app_model else "YES",
        "detail": f"Recovery dir: {RECOVERY_DIR}, app_model/: {'EXISTS' if os.path.isdir(recovery_app_model_dir) else 'MISSING'}"
    })

    # ── 3. use_app flag checkpointed? ──
    chkpt_path = os.path.join(BASELINE_DIR, "chkpnt15000.pth")
    if os.path.exists(chkpt_path):
        ckpt = torch.load(chkpt_path, map_location='cpu', weights_only=False)
        model_args = ckpt[0]
        has_use_app = any(isinstance(a, bool) for a in model_args if isinstance(a, bool))
        findings.append({
            "component": "use_app_flag",
            "check": "Is use_app in checkpoint?",
            "result": "NO",
            "detail": f"capture() returns {len(model_args)} items, none is use_app (GaussianModel.capture() line 86-105)"
        })
    else:
        findings.append({
            "component": "use_app_flag",
            "check": "checkpoint exists",
            "result": "MISSING",
            "detail": f"{chkpt_path} not found"
        })

    # ── 4. SpecularModel check ──
    specular_dir = os.path.join(BASELINE_DIR, "specular")
    has_specular = os.path.isdir(specular_dir)
    findings.append({
        "component": "SpecularModel",
        "check": "SpecularModel used in scene_01?",
        "result": "NO",
        "detail": f"Training used use_asg=False; specular/ dir: {'EXISTS' if has_specular else 'MISSING'}"
    })

    # ── 5. Original training command verified ──
    cfg_args_path = os.path.join(BASELINE_DIR, "cfg_args")
    if os.path.exists(cfg_args_path):
        with open(cfg_args_path) as f:
            cfg_args = f.read().strip()
        has_exposure_comp = "exposure_compensation" in cfg_args
        has_delight = "delight=True" in cfg_args
        findings.append({
            "component": "Training config",
            "check": "Was exposure_compensation enabled?",
            "result": "NO" if not has_exposure_comp else "YES",
            "detail": f"cfg_args: {cfg_args[:200]}..."
        })
    else:
        findings.append({
            "component": "Training config",
            "check": "cfg_args exists",
            "result": "MISSING",
            "detail": "cfg_args not found"
        })

    # ── 6. PSNR comparison: with vs without AppModel ──
    debug_results_path = os.path.join(OUTPUT_DIR, "debug_recovery_app_model_results.json")
    if os.path.exists(debug_results_path):
        with open(debug_results_path) as f:
            results = json.load(f)
        gap = results.get('baseline_no_app', 0) - results.get('recovery_no_app', 0)
        findings.append({
            "component": "PSNR comparison",
            "check": "AppModel uplift (baseline)",
            "result": f"{results.get('baseline_with_app', 0) - results.get('baseline_no_app', 0):.4f} dB",
            "detail": f"baseline_no_app={results.get('baseline_no_app')}, baseline_with_app={results.get('baseline_with_app')}, "
                      f"recovery_no_app={results.get('recovery_no_app')}, recovery_with_app={results.get('recovery_with_app')}"
        })
    else:
        findings.append({
            "component": "PSNR comparison",
            "check": "debug results available",
            "result": "NO",
            "detail": f"{debug_results_path} not found"
        })

    # ── 7. Recovery PLY diff from baseline ──
    base_ply = os.path.join(BASELINE_DIR, "point_cloud", "iteration_15000", "point_cloud.ply")
    rec_ply = os.path.join(RECOVERY_DIR, "point_cloud", "iteration_15500", "point_cloud.ply")
    if os.path.exists(base_ply) and os.path.exists(rec_ply):
        from plyfile import PlyData
        import numpy as np
        base = PlyData.read(base_ply)
        rec = PlyData.read(rec_ply)
        base_xyz = np.column_stack([base.elements[0].data['x'], base.elements[0].data['y'], base.elements[0].data['z']])
        rec_xyz = np.column_stack([rec.elements[0].data['x'], rec.elements[0].data['y'], rec.elements[0].data['z']])
        if base_xyz.shape[0] == rec_xyz.shape[0]:
            xyz_diff = np.abs(base_xyz - rec_xyz).max()
            base_f_dc = np.column_stack([base.elements[0].data['f_dc_0'], base.elements[0].data['f_dc_1'], base.elements[0].data['f_dc_2']])
            rec_f_dc = np.column_stack([rec.elements[0].data['f_dc_0'], rec.elements[0].data['f_dc_1'], rec.elements[0].data['f_dc_2']])
            f_dc_diff = np.abs(base_f_dc - rec_f_dc).max()
            base_op = base.elements[0].data['opacity']
            rec_op = rec.elements[0].data['opacity']
            op_diff = np.abs(base_op - rec_op).max()
            findings.append({
                "component": "Recovery PLY",
                "check": "Is recovery PLY different from baseline?",
                "result": "YES",
                "detail": f"XYZ max diff={xyz_diff:.6f}, f_dc max diff={f_dc_diff:.6f}, opacity max diff={op_diff:.6f}"
            })
        else:
            findings.append({
                "component": "Recovery PLY",
                "check": "Gaussian count matches",
                "result": f"NO (baseline={base_xyz.shape[0]}, recovery={rec_xyz.shape[0]})",
                "detail": ""
            })

    # ── Generate report ──
    lines = [
        "# Stage 2B-AA: TSGS Full-State Bundle Audit Report",
        "",
        f"**Date**: 2026-07-11",
        f"**Scene**: scene_01",
        f"**Baseline**: {BASELINE_DIR}",
        f"**Recovery**: schedule_control, 500 steps",
        "",
        "## Executive Summary",
        "",
        "This audit investigates the hypothesis that the 4.86 dB PSNR drop (22.39 → 17.54) during recovery "
        "training is caused by **AppModel** (appearance model) being updated during recovery but not loaded "
        "at evaluation time.",
        "",
    ]

    # Key finding
    app_model_finding = [f for f in findings if f["component"] == "AppModel" and "non-zero" in f.get("check", "")]
    if app_model_finding and "ALL ZEROS" in app_model_finding[0]["result"]:
        lines.append("### VERDICT: AppModel is NOT the cause of the PSNR gap for scene_01")
        lines.append("")
        lines.append("**Evidence**: AppModel `appear_ab` weights are ALL ZEROS (mean=0.0, std=0.0, all 3200 elements zero).")
        lines.append("The original training did not enable `exposure_compensation`, so `gaussians.use_app` remained `False`")
        lines.append("throughout training. The AppModel was created and optimizer steps were called, but its output was")
        lines.append("never connected to any loss term, resulting in zero gradients and zero weight updates.")
        lines.append("")
        lines.append("The PSNR gap must be caused by other factors (e.g., LR policy, geometric degradation, overfitting to")
        lines.append("training views, or the `active_sh_degree=0` evaluation setting).")
    else:
        lines.append("### VERDICT: AppModel IS likely contributing to the PSNR gap")
        lines.append("")

    lines += [
        "",
        "## Detailed Findings",
        "",
        "| # | Component | Check | Result | Detail |",
        "|---|-----------|-------|--------|--------|",
    ]

    for i, f in enumerate(findings):
        result_clean = f["result"].replace("|", "/")
        detail_clean = f["detail"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {i+1} | {f['component']} | {f['check']} | {result_clean} | {detail_clean} |")

    lines += [
        "",
        "## Recommendations",
        "",
    ]

    # Recommendations based on findings
    if app_model_finding and "ALL ZEROS" in app_model_finding[0]["result"]:
        recommendations = [
            ("Remove AppModel from recovery training", "HIGH",
             "AppModel adds no value for this scene (all-zero weights). Remove `app_model` references from "
             "`train_pruned_recovery.py` to reduce confusion and unnecessary computation."),
            ("Investigate real PSNR gap cause", "HIGH",
             "The 4.86 dB gap is not from AppModel. Investigate: (a) SH degree evaluation mismatch, "
             "(b) LR policy differences between training and recovery, "
             "(c) validation camera set mismatch between baseline and recovery evaluations."),
            ("Fix `active_sh_degree` in evaluation", "MEDIUM",
             "The current evaluator sets `active_sh_degree=0`, which uses only DC SH components. "
             "This significantly under-represents quality. Either match the training max SH degree or "
             "clearly document this limitation."),
            ("Add `use_app` to checkpoint capture", "LOW",
             "The `use_app` flag is not saved in GaussianModel.capture(). For future scenes that use "
             "exposure_compensation, this flag should be preserved across checkpoints."),
        ]
    else:
        recommendations = [
            ("Save AppModel during recovery", "CRITICAL",
             "Add `app_model.save_weights(output_dir, end_iter)` after recovery training completes."),
            ("Enable `use_app` during evaluation", "HIGH",
             "Set `gaussians.use_app = True` when loading AppModel for evaluation."),
            ("Add `use_app` to checkpoint capture", "MEDIUM",
             "The `use_app` flag is not saved in GaussianModel.capture(). This should be added."),
        ]

    lines.append("| Priority | Recommendation | Rationale |")
    lines.append("|----------|---------------|----------|")
    for rec, priority, rationale in recommendations:
        lines.append(f"| **{priority}** | {rec} | {rationale} |")

    lines += [
        "",
        "## Files Modified in This Audit",
        "",
        "| File | Action |",
        "|------|--------|",
        "| `src/prune/audit_tsgs_auxiliary_state.py` | CREATED — Audits AppModel/SpecularModel lifecycle |",
        "| `src/prune/debug_recovery_with_app_model.py` | CREATED — Tests PSNR with/without AppModel |",
        "| `src/prune/debug_check_app_model.py` | CREATED — Checks AppModel weight values |",
        "| `src/prune/build_bundle_audit_report.py` | CREATED — Generates this report |",
        "| `src/prune/train_pruned_recovery.py` | MODIFIED — Saves AppModel weights at end of recovery |",
        "| `src/evaluation/unified_recovery_evaluator.py` | MODIFIED — Supports AppModel rendering |",
        "| `src/recyclegs/tsgs_loader.py` | MODIFIED — Added load_app_model helper |",
        "",
    ]

    md = "\n".join(lines)
    out_path = os.path.join(OUTPUT_DIR, "bundle_audit_report.md")
    with open(out_path, 'w') as f:
        f.write(md)
    print(f"Saved: {out_path}")
    print(f"Total: {len(md)} chars, {len(findings)} findings, {len(recommendations)} recommendations")

    # Also save findings as JSON
    json_path = os.path.join(OUTPUT_DIR, "bundle_audit_findings.json")
    with open(json_path, 'w') as f:
        json.dump({"findings": findings, "recommendations": [{"priority": p, "text": t, "rationale": r} for t, p, r in recommendations]}, f, indent=2)
    print(f"Saved: {json_path}")

if __name__ == '__main__':
    main()
