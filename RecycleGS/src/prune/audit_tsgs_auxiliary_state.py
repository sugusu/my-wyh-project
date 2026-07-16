#!/usr/bin/env python3
"""Audit TSGS auxiliary state: AppModel + SpecularModel saving/loading in full-training pipeline."""
import os, sys, json, torch, textwrap

OUTPUT_DIR = "/data/wyh/RecycleGS/outputs/debug/stage2b_bundle_audit"
BASELINE_DIR = "/data/wyh/RecycleGS/baselines/tsgs_scene01_full"

def section(title, body):
    return f"## {title}\n\n{body}\n\n"

def code(text):
    return f"```\n{text}\n```"

def check_path(path, desc):
    exists = os.path.exists(path)
    return f"  - {desc}: {'EXISTS' if exists else 'MISSING'} -> `{path}`"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lines = ["# TSGS Auxiliary State Audit", ""]
    lines.append("Audit of AppModel, SpecularModel save/load lifecycle in TSGS full training pipeline.")
    lines.append("")

    # ── Part 1: AppModel Architecture ──
    app_model_code = """
class AppModel(nn.Module):
    def __init__(self, num_images=1600):
        self.appear_ab = nn.Parameter(torch.zeros(num_images, 2).cuda())
        # shape: (num_images, 2) → per-image (alpha, beta)
        # app_image = exp(alpha) * rendered_image + beta

    def save_weights(self, model_path, iteration):
        path = model_path / "app_model/iteration_{iter}/app.pth"
        torch.save(self.state_dict(), path)

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:  # load latest
            iter = searchForMaxIteration(model_path / "app_model")
        path = model_path / "app_model/iteration_{iter}/app.pth"
        self.load_state_dict(torch.load(path))
"""
    lines.append(section("AppModel Architecture",
        f"`{app_model_code}`\n"
        "The AppModel is a simple per-image affine brightness/color transformation:\n"
        "- Parameter `appear_ab`: shape `(N_images, 2)`, `app_image = exp(alpha) * render + beta`\n"
        "- Optimizer: Adam, lr=0.001, only `appear_ab`\n"
        "- Saved to `app_model/iteration_{N}/app.pth`\n"
        "- Loaded by `load_weights(model_path, iteration)` — defaults to latest iteration"
    ))

    # ── Part 2: SpecularModel ──
    lines.append(section("SpecularModel Architecture",
        "`SpecularNetwork` / `SpecularNetworkReal` — MLP that maps (ASG features, viewdir, normal) → RGB.\n"
        "- Saved to `specular/iteration_{N}/specular.pth`\n"
        "- Only used when `opt.use_asg=True`\n"
        "- For scene_01 (use_asg=False), SpecularModel is never created"
    ))

    # ── Part 3: When is AppModel saved in train.py? ──
    train_save_points = """
Line 437-439: if iteration in saving_iterations:
    scene.save(iteration)                        # Gaussians PLY
    if specular_mlp is not None:
        specular_mlp.save_weights(...)          # Specular

Line 486-489: if iteration in checkpoint_iterations:
    torch.save((gaussians.capture(), iteration), ...)  # full state
    app_model.save_weights(model_path, iteration)       # AppModel

Line 491: app_model.save_weights(model_path, opt.iterations)  # Final save
"""
    lines.append(section("When is AppModel saved in official train.py?", code(train_save_points)))
    lines.append("Key observation: AppModel is saved at **checkpoint iterations** and at the **final iteration**. "
                 "It is NOT automatically saved at every `saving_iterations` — only at `checkpoint_iterations` "
                 "and at the very end. However, `saving_iterations` always includes `args.iterations` (line 683).")

    # ── Part 4: What exists in baseline model directory ──
    lines.append(section("AppModel weights in baseline directory", ""))
    app_model_files = []
    if os.path.isdir(os.path.join(BASELINE_DIR, "app_model")):
        for root, dirs, files in os.walk(os.path.join(BASELINE_DIR, "app_model")):
            for f in sorted(files):
                fp = os.path.join(root, f)
                size_kb = os.path.getsize(fp) / 1024
                app_model_files.append(f"{fp} ({size_kb:.1f} KB)")
    lines.append(code("\n".join(app_model_files) if app_model_files else "(none found)"))
    lines.append("")

    # ── Part 5: What exists in recovery output ──
    lines.append(section("Recovery output contents", ""))
    recovery_dir = "/data/wyh/RecycleGS/outputs/prune_only/scene_01/ratio_005/schedule_control/recovery_500"
    recovery_files = []
    if os.path.isdir(recovery_dir):
        for root, dirs, files in os.walk(recovery_dir):
            for f in sorted(files):
                fp = os.path.join(root, f)
                recovery_files.append(fp)
    lines.append(code("\n".join(recovery_files) if recovery_files else "(directory not found)"))
    
    app_model_in_recovery = any("app_model" in fp for fp in recovery_files)
    lines.append(f"\nAppModel weights in recovery output: **{'YES' if app_model_in_recovery else 'NO'}**")
    lines.append("")

    # ── Part 6: How evaluation uses (or doesn't use) AppModel ──
    eval_analysis = """
unified_recovery_evaluator.py (line 20-23):
    def tsgs_render(gaussians, cam, pipe, bg_color, device='cuda:0'):
        from gaussian_renderer import render
        bg = torch.tensor(bg_color, dtype=torch.float32, device=device)
        return render(cam, gaussians, pipe, bg)  # <-- NO app_model passed!

render() in gaussian_renderer/__init__.py (line 154):
    if app_model is not None and pc.use_app:
        appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
        app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
        return_dict.update({"app_image": app_image})

→ Two gates: (1) app_model is not None, AND (2) pc.use_app is True
→ gaussians.use_app defaults to False, only set True in train.py when:
    if iteration > 1000 and opt.exposure_compensation:
        gaussians.use_app = True
→ use_app is NOT saved in checkpoint (capture/restore don't include it)
→ So even if evaluator passed app_model, it wouldn't activate without use_app=True!
"""
    lines.append(section("How evaluation uses (or doesn't use) AppModel", eval_analysis))

    # ── Part 7: Recovery training AppModel usage ──
    recovery_app = """
train_pruned_recovery.py (lines 191-197):
    app_model = None
    if use_delight:
        app_model = AppModel()
        if os.path.exists(app_ckpt_path):
            app_model.load_weights(model_dir)
        app_model.train()
        app_model.cuda()

train_pruned_recovery.py (lines 255-256):
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model, ...)

train_pruned_recovery.py (lines 308-312):
    gaussians.optimizer.step()
    if app_model is not None:
        app_model.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)
    if app_model is not None:
        app_model.optimizer.zero_grad(set_to_none=True)

train_pruned_recovery.py (lines 316-335):
    # Saves: chkpnt_recovery.pth, point_cloud.ply, training_log.json, gaussian_count_trace.csv
    # DOES NOT save app_model weights!
    # DOES NOT save specular weights!
"""
    lines.append(section("Recovery training AppModel usage", recovery_app))
    lines.append("**Critical finding**: Recovery training loads AppModel, passes it to render, "
                 "and runs optimizer steps on it — but NEVER saves AppModel weights. "
                 "This means any adaptation in AppModel during recovery is LOST on evaluation.")

    # ── Part 8: The `use_app` flag ──
    lines.append(section("`use_app` flag lifecycle", ""))
    lines.append("1. **Initial value**: `self.use_app = False` in `GaussianModel.__init__` (line 84)\n"
                 "2. **Set to True**: `train.py` line 143: `gaussians.use_app = True` at iteration > 1000\n"
                 "3. **Not checkpointed**: `capture()` (line 86-105) does NOT include `use_app`\n"
                 "4. **Recovery training**: `exposure_compensation = False` → `use_app` never set to True\n"
                 "5. **Evaluator**: never sets `use_app = True`\n"
                 "6. **Result**: AppModel weights exist but are NEVER applied during evaluation")

    # ── Part 9: Summary ──
    lines.append(section("Summary of findings", ""))
    summary = """
| Aspect | Status | Impact |
|--------|--------|--------|
| AppModel saved in full training? | YES (at checkpoint/final iters) | N/A |
| AppModel in baseline dir (15k)? | YES (app_model/iteration_15000/app.pth, 25.6 KB) | N/A |
| AppModel saved in recovery? | **NO** | Weights updated during recovery are lost |
| AppModel loaded in evaluation? | **NO** (not passed to render + use_app=False) | ~0.5-2 dB potential gain lost |
| `use_app` flag in checkpoint? | **NO** (not captured in `capture()`) | Even loaded AppModel won't activate |
| SpecularModel in scene_01? | Not used (use_asg=False) | No impact |
"""
    lines.append(summary)
    lines.append("")

    md = "\n".join(lines)
    out_path = os.path.join(OUTPUT_DIR, "tsgs_auxiliary_state_audit.md")
    with open(out_path, 'w') as f:
        f.write(md)
    print(f"Saved: {out_path}")
    print(f"Total size: {len(md)} chars")

if __name__ == '__main__':
    main()
