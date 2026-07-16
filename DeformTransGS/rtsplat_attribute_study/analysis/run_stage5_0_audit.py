from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
RT = ROOT / "repos" / "RT-Splatting"
OUT = PROJECT / "experiments" / "stage5_0_rtsplat_native_carrier_gate"
GT_ROOT = PROJECT / "experiments" / "stage4_0_R2A_GT1_gt_optical_semantics_closure" / "clean_gt"
MIGRATION = PROJECT / "runtime" / "rtsplat_migration_bundle"
USER_SITE = Path("/home/wyh/.local/lib/python3.10/site-packages")
NVDIFRAST = ROOT / "repos" / "nvdiffrast"


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None, timeout: int = 30) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 999, "", repr(e)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> dict:
    exists = path.exists()
    st = path.stat() if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "size": st.st_size if st else "",
        "mtime": st.st_mtime if st else "",
        "sha256": sha256_file(path) if exists and path.is_file() else "",
    }


def import_test(module: str, extra_path: str = "") -> dict:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "2,3"
    pp = [str(RT)]
    if extra_path:
        pp.append(extra_path)
    pp.append(str(USER_SITE))
    env["PYTHONPATH"] = ":".join(pp)
    code = f"import importlib, traceback\ntry:\n m=importlib.import_module('{module}')\n print('FILE='+str(getattr(m,'__file__','BUILTIN')))\nexcept Exception:\n traceback.print_exc()\n raise\n"
    rc, out, err = run(["/usr/bin/python3", "-c", code], env=env, timeout=20)
    trace = out + err
    if rc == 0:
        status = "PASS"
        cls = "PASS"
        path = out.strip().replace("FILE=", "")
    else:
        status = "FAIL"
        path = ""
        if "No module named" in trace:
            if "_C" in trace or "_nvdiffrast_c" in trace or "diff_surfel_anych" in trace:
                cls = "C_EXTENSION_NOT_FOUND"
            else:
                cls = "MODULE_NOT_FOUND"
        elif "undefined symbol" in trace:
            cls = "UNDEFINED_SYMBOL"
        elif "GLIBCXX" in trace:
            cls = "GLIBCXX_ERROR"
        else:
            cls = "OTHER"
    return {"module": module, "status": status, "classification": cls, "path": path, "returncode": rc, "trace": trace}


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("Stage5.0 must run with CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)

    git_cmds = {
        "status": ["git", "status", "--short"],
        "head": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "remote": ["git", "remote", "-v"],
        "submodule": ["git", "submodule", "status"],
    }
    git = {}
    for name, cmd in git_cmds.items():
        rc, out, err = run(cmd, RT)
        git[name] = {"returncode": rc, "stdout": out, "stderr": err}
    write_text(
        OUT / "rtsplat_repository_state.txt",
        "\n".join([f"## git {k}\nreturncode={v['returncode']}\n{v['stdout']}{v['stderr']}" for k, v in git.items()]) + "\n",
    )

    lock_paths = [
        RT / "README.md",
        RT / "requirements.txt",
        RT / "scene" / "gaussian_model.py",
        RT / "gaussian_renderer" / "__init__.py",
        RT / "train.py",
        RT / "submodules" / "diff-surfel-anych" / "setup.py",
        RT / "submodules" / "simple-knn" / "setup.py",
        GT_ROOT,
    ]
    protocol = {
        "repository_root": str(RT),
        "git": git,
        "files": {str(p): file_record(p) for p in lock_paths},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_text(OUT / "stage5_0_protocol_lock.json", json.dumps(protocol, indent=2) + "\n")
    J0 = "PASS" if RT.exists() and (RT / "README.md").exists() and (RT / ".git").exists() and GT_ROOT.exists() else "FAIL"

    readme = (RT / "README.md").read_text(errors="replace") if (RT / "README.md").exists() else ""
    repo_class = "LOCAL-FORK-TRACEABLE" if "github.com/sjj118/RT-Splatting" in git["remote"]["stdout"] else "LOCAL-SOURCE-PROVENANCE-UNKNOWN"
    repo_complete = all((RT / p).exists() for p in ["scene/gaussian_model.py", "gaussian_renderer/__init__.py", "submodules/diff-surfel-anych/setup.py", "submodules/simple-knn/setup.py"])
    write_text(
        OUT / "rtsplat_repository_identity.md",
        f"""# RT-Splatting Repository Identity

Repository root: `{RT}`

Classification: `{repo_class}`

Complete runnable source tree: `{"YES" if repo_complete else "NO"}`

Commit: `{git['head']['stdout'].strip()}`

Branch: `{git['branch']['stdout'].strip()}`

Remote:

```text
{git['remote']['stdout'].strip()}
```

README title/source signal:

```text
{readme[:1200]}
```

Submodule status:

```text
{git['submodule']['stdout'] or 'NO_SUBMODULE_STATUS_OUTPUT'}
```
""",
    )

    states = [
        ("_xyz", "GEOMETRY_POSITION", "get_xyz", "Gaussian position", "gaussian_model.py:101,264,460"),
        ("_features_dc", "SH_APPEARANCE", "get_features", "SH DC color", "gaussian_model.py:102,265,461"),
        ("_features_rest", "SH_APPEARANCE", "get_features", "SH residual color", "gaussian_model.py:103,266,462"),
        ("_scaling", "GEOMETRY_SCALE", "get_scaling=exp(_scaling)", "native 2D surfel footprint scale", "gaussian_model.py:104,210,267,466"),
        ("_rotation", "GEOMETRY_ROTATION", "get_rotation=normalize(_rotation)", "native surfel rotation", "gaussian_model.py:105,214,268,467"),
        ("_occupancy", "GEOMETRIC_OCCUPANCY", "get_occupancy=sigmoid(_occupancy)", "surface/volume occupancy and raster opacities input", "gaussian_model.py:106,228,269,463; renderer.py:56,71,108"),
        ("_opacity", "OPTICAL_OPACITY", "get_opacity=sigmoid(_opacity)", "volume pass optical opacity and surface extra", "gaussian_model.py:117,232,270,464; renderer.py:57,64,71,101"),
        ("_transmissivity", "TRANSMISSIVITY", "get_transmissivity=sigmoid(_transmissivity)", "mixes transmitted/scattered final image", "gaussian_model.py:118,236,271,465; renderer.py:101,114,186-189"),
        ("_roughness", "ROUGHNESS", "get_roughness=sigmoid(_roughness)", "specular MLP roughness level", "gaussian_model.py:116,202,274,469; renderer.py:101,153,168"),
        ("_reflectance", "REFLECTION_COLOR", "get_reflectance=sigmoid(_reflectance+inv_sigmoid(0.5))", "scales final_spec", "gaussian_model.py:194,275,470; renderer.py:101,114,188"),
        ("_language_feature", "MATERIAL_FEATURE", "get_language_feature=tanh(_language_feature)", "feature input to specular MLP", "gaussian_model.py:206,276,472; renderer.py:101,155-177"),
    ]
    inv_rows = [
        {
            "source_file": "scene/gaussian_model.py",
            "class": "GaussianModel",
            "attribute_name": name,
            "semantic_label": sem,
            "activation": act,
            "initialization": "create_from_pcd/load_ply",
            "optimizer_group": name.strip("_") if name not in ["_features_dc", "_features_rest", "_language_feature"] else ("f_dc/f_rest" if "features" in name else "feature"),
            "checkpoint_key": "PLY attribute or torch module sidecar" if name != "_xyz" else "PLY x/y/z",
            "render_consumer": desc,
            "source_lines": lines,
        }
        for name, sem, act, desc, lines in states
    ]
    write_csv(OUT / "rtsplat_state_tensor_inventory.csv", inv_rows)

    write_text(
        OUT / "rtsplat_opacity_decoupling_trace.md",
        """# RT-Splatting Opacity Decoupling Trace

Geometric occupancy equation:

`occupancy = sigmoid(_occupancy)` from `scene/gaussian_model.py:89-91,227-229`.

Optical opacity equation:

`opacity = sigmoid(_opacity)` from `scene/gaussian_model.py:92-93,231-233`.

Transmissivity equation:

`transmissivity = sigmoid(_transmissivity)` from `scene/gaussian_model.py:94-95,235-237`.

Volume pass:

`opacities = occupancy * opacity` and `extras = [opacity]` in `gaussian_renderer/__init__.py:64-72`.

Surface/deferred pass:

`opacities = occupancy`; extras include `roughness`, `feature`, `inside_mask`, `inside_mask * reflectance`, `opacity`, and `inside_mask * transmissivity` in `gaussian_renderer/__init__.py:101-114`.

Final image:

`final_tran = render_tran * render_transmissivity`

`final_scat = render_scat * (1 - render_transmissivity)`

`final_spec = render_spec * render_reflectance`

`final_rendering = final_tran + final_scat (+ final_spec after init stage)` in `gaussian_renderer/__init__.py:186-194`.

Decision:

`NATIVE-TRANSPARENT-DECOUPLING = PASS`. `_occupancy`, `_opacity`, and `_transmissivity` are persistent semantically distinct state tensors and are not duplicate aliases of one scalar opacity.
""",
    )
    J1 = "PASS"

    dep_rows = [
        {"state_tensor": "_occupancy", "activation": "sigmoid", "intermediate": "occupancy", "renderer_input": "opacities=occupancy*opacity in volume; opacities=occupancy in surface", "image_component": "render_tran/render_scat/surface_alpha/volume_alpha", "affects": "effective alpha, foreground, depth/normal support"},
        {"state_tensor": "_opacity", "activation": "sigmoid", "intermediate": "opacity", "renderer_input": "extras=[opacity], volume opacities=occupancy*opacity, surface extra surface_opacity", "image_component": "render_tran volume opacity and surface_opacity diagnostics", "affects": "direct transparent volume contribution"},
        {"state_tensor": "_transmissivity", "activation": "sigmoid", "intermediate": "render_transmissivity", "renderer_input": "surface extra inside_mask*transmissivity", "image_component": "final_tran/final_scat/final_rendering", "affects": "transmission/scatter mixing"},
        {"state_tensor": "_roughness", "activation": "sigmoid", "intermediate": "render_roughness", "renderer_input": "surface extra roughness -> SphMip level", "image_component": "final_spec after init stage", "affects": "specular material response"},
        {"state_tensor": "_reflectance", "activation": "sigmoid + bias", "intermediate": "render_reflectance", "renderer_input": "surface extra inside_mask*reflectance", "image_component": "final_spec", "affects": "reflection contribution"},
        {"state_tensor": "_language_feature", "activation": "tanh then normalized", "intermediate": "render_feature", "renderer_input": "surface extra feature -> light_mlp", "image_component": "render_spec/final_spec", "affects": "material feature/specular color"},
    ]
    write_csv(OUT / "rtsplat_material_state_dependency.csv", dep_rows)
    write_text(OUT / "rtsplat_material_state_graph.md", "\n".join(f"- `{r['state_tensor']}` -> {r['activation']} -> {r['intermediate']} -> {r['renderer_input']} -> {r['image_component']}" for r in dep_rows) + "\n")

    serializable = ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_occupancy", "_opacity", "_transmissivity", "_roughness", "_reflectance", "_language_feature", "light_mlp", "dir_encoding"]
    capture_missing = ["_opacity", "_roughness", "_reflectance", "_language_feature", "light_mlp", "dir_encoding"]
    write_text(
        OUT / "rtsplat_checkpoint_state_audit.md",
        f"""# Checkpoint Serialization Audit

PLY save/load path serializes the persistent point attributes: `{', '.join(serializable[:11])}`.

Sidecar module files are written for `light_mlp` and `dir_encoding` in `save_ply`.

However, the in-memory `capture()` / `restore()` training checkpoint tuple omits: `{', '.join(capture_missing)}`.

J2a classification: `FAIL`.

Reason: not all transparent/material persistent states identified for J1/J5 can be serialized and restored by the training checkpoint tuple without relying on separate PLY/module save paths or defaults.
""",
    )
    J2a = "FAIL"

    dep_inventory = []
    req = (RT / "requirements.txt").read_text().splitlines()
    for line in req:
        line = line.strip()
        if line and not line.startswith("#"):
            dep_inventory.append({"requirement": line, "source": "requirements.txt"})
    write_csv(OUT / "rtsplat_dependency_inventory.csv", dep_inventory)

    so_paths = []
    for root in [RT, USER_SITE, NVDIFRAST]:
        if root.exists():
            so_paths.extend(root.rglob("*.so"))
    ext_rows = []
    ldd_rows = []
    for p in sorted(set(so_paths)):
        if not any(s in str(p) for s in ["simple_knn", "diff_surfel", "nvdiffrast"]):
            continue
        ext_rows.append({"path": str(p), "size": p.stat().st_size, "sha256": sha256_file(p), "abi": p.name})
        rc, out, err = run(["ldd", str(p)], timeout=20)
        unresolved = sum(1 for line in (out + err).splitlines() if "not found" in line)
        ldd_rows.append({"path": str(p), "returncode": rc, "unresolved_dependency_count": unresolved, "ldd": out + err})
    for setup in [RT / "submodules/simple-knn/setup.py", RT / "submodules/diff-surfel-anych/setup.py"]:
        ext_rows.append({"path": str(setup), "size": setup.stat().st_size if setup.exists() else "", "sha256": sha256_file(setup) if setup.exists() else "", "abi": "SOURCE_SETUP"})
    write_csv(OUT / "rtsplat_extension_inventory.csv", ext_rows)
    write_csv(OUT / "rtsplat_extension_dependency_audit.csv", ldd_rows)

    modules = ["torch", "numpy", "nvdiffrast", "nvdiffrast.torch", "simple_knn._C", "diff_surfel_anych", "scene.gaussian_model", "gaussian_renderer"]
    imps = [import_test(m, str(NVDIFRAST) if m.startswith("nvdiffrast") else "") for m in modules]
    write_csv(OUT / "rtsplat_runtime_import_matrix.csv", [{k: v for k, v in r.items() if k != "trace"} for r in imps])
    write_text(OUT / "rtsplat_import_full_traces.txt", "\n\n".join(f"## {r['module']} {r['status']}\n{r['trace']}" for r in imps) + "\n")
    import_pass = sum(1 for r in imps if r["status"] == "PASS")
    import_total = len(imps)
    critical_ext_pass = sum(1 for r in imps if r["module"] in ["simple_knn._C", "diff_surfel_anych", "nvdiffrast.torch"] and r["status"] == "PASS")
    critical_ext_total = 3
    unresolved_ldd = sum(int(r["unresolved_dependency_count"]) for r in ldd_rows)

    hist = """# Historical RT-Splatting Migration Failure Audit

The migration bundle required a CUDA 11.8 devel container, torch==2.0.1+cu118, torchvision==0.15.2, nvdiffrast build, and local builds of `diff-surfel-anych` and `simple-knn`.

Current local evidence:

- Current usable torch is in user site as torch 2.6.0+cu124, not the requested torch 2.0.1+cu118.
- Clean `/usr/bin/python3` with `PYTHONNOUSERSITE=1` cannot import torch/numpy.
- Adding user site exposes torch/numpy and an existing `simple_knn._C`.
- `nvdiffrast.torch` fails because `_nvdiffrast_c` is missing.
- `diff_surfel_anych` is not installed.
- `/usr/include/python3.10/Python.h` is missing, so local C++/CUDA extension rebuild is not compatible without changing system packages.

Classifications:

- torch version conflict: `STILL-APPLIES-TO-CURRENT-LOCAL-REPO`
- CUDA mismatch: `UNRESOLVED`
- cudnn9 requirement: `NO-LONGER-APPLIES` to current source audit; not the observed blocker
- extension build failure risk: `STILL-APPLIES-TO-CURRENT-LOCAL-REPO`
- missing Python development headers: `STILL-APPLIES-TO-CURRENT-LOCAL-REPO`
"""
    write_text(OUT / "historical_rtsplat_failure_audit.md", hist)

    local_candidates = [
        {"package": "torch", "candidate": str(USER_SITE / "torch"), "status": "AVAILABLE_USER_SITE", "version": "2.6.0+cu124 observed in Stage4"},
        {"package": "simple_knn._C", "candidate": "/home/wyh/.local/lib/python3.10/site-packages/simple_knn/_C.cpython-310-x86_64-linux-gnu.so", "status": "AVAILABLE_BUT_LDD_NEEDS_TORCH_LIB_PATH", "version": "cpython-310"},
        {"package": "nvdiffrast", "candidate": str(NVDIFRAST), "status": "SOURCE_ONLY_NO__nvdiffrast_c", "version": "local source"},
        {"package": "diff_surfel_anych", "candidate": str(RT / "submodules/diff-surfel-anych"), "status": "SOURCE_ONLY_NO_INSTALLED_EXTENSION", "version": "local source"},
    ]
    write_csv(OUT / "rtsplat_local_package_candidates.csv", local_candidates)

    strategy = {
        "decision": "RUNTIME-C",
        "label": "LOCAL-COMPATIBLE-RUNTIME-NOT-AVAILABLE",
        "reason": "critical nvdiffrast and diff_surfel_anych extensions are not importable; project-local rebuild is blocked by missing Python.h and no compatible prebuilt local runtime; downloading a new PyTorch/CUDA stack is forbidden",
        "selected_interpreter": "/usr/bin/python3",
        "selected_torch": "torch 2.6.0+cu124 only via explicit user-site PYTHONPATH; not sufficient for RT critical extensions",
    }
    write_text(OUT / "rtsplat_runtime_strategy.json", json.dumps(strategy, indent=2) + "\n")

    J2b = "FAIL"
    J2 = "FAIL"
    write_text(OUT / "verified_rtsplat_runtime_lock.json", json.dumps({"accepted": False, "J2b": J2b, "reason": strategy["reason"], "import_pass": import_pass, "import_total": import_total}, indent=2) + "\n")
    launcher = PROJECT / "rtsplat_attribute_study" / "runtime" / "verified_rtsplat_python.sh"
    write_text(launcher, "#!/usr/bin/env bash\nset -euo pipefail\necho 'RT-Splatting runtime is not verified: LOCAL-COMPATIBLE-RUNTIME-NOT-AVAILABLE' >&2\nexit 2\n")
    launcher.chmod(0o755)

    # Downstream gated outputs are intentionally not fabricated.
    not_exec_rows = [{"status": "NOT_EXECUTED_J2_FAIL", "reason": "RT runtime/checkpoint-state gate failed before native forward, J3, and J4"}]
    for fname in [
        "rtsplat_native_forward_smoke.csv",
        "rtsplat_native_state_perturbation.csv",
        "rtsplat_native_state_gradient.csv",
        "stage5_canonical_case_lock.csv",
        "stage5_canonical_gt_identity.csv",
        "stage5_native_carrier_audit.csv",
        "stage5_footprint_diagnostic.csv",
        "stage5_canonical_first_step_audit.csv",
        "stage5_canonical_checkpoint_integrity.csv",
        "stage5_canonical_render_manifest.csv",
        "stage5_canonical_metrics.csv",
        "stage5_metric_reproduction.csv",
        "stage5_canonical_capacity_diagnostic.csv",
        "stage5_vs_stage4_canonical_comparison.csv",
    ]:
        write_csv(OUT / fname, not_exec_rows)
    write_text(OUT / "stage5_native_carrier_initialization.md", "NOT_EXECUTED_J2_FAIL: native RT runtime was not verified.\n")
    write_text(OUT / "stage5_canonical_trainable_state_lock.json", json.dumps({"status": "NOT_EXECUTED_J2_FAIL"}, indent=2) + "\n")
    write_text(OUT / "stage5_footprint_policy_lock.json", json.dumps({"status": "NOT_EXECUTED_J2_FAIL"}, indent=2) + "\n")
    for case in ["K0", "K1", "K2"]:
        write_csv(OUT / "stage5_canonical_history" / f"{case}.csv", not_exec_rows)

    final_case = "CASE LOCAL-COMPATIBLE-RUNTIME-NOT-AVAILABLE"
    items = [
        ("A", "J0", J0),
        ("B", "local RT-Splatting repository classification", repo_class),
        ("C", "git commit / branch", f"{git['head']['stdout'].strip()} / {git['branch']['stdout'].strip()}"),
        ("D", "repository complete yes/no", "YES" if repo_complete else "NO"),
        ("E", "identified persistent state tensor count", str(len(states))),
        ("F", "transparent/material state inventory", "GEOMETRIC_OCCUPANCY->_occupancy; OPTICAL_OPACITY->_opacity; TRANSMISSIVITY->_transmissivity; ROUGHNESS->_roughness; REFLECTION_COLOR->_reflectance; MATERIAL_FEATURE->_language_feature; SH_APPEARANCE->_features_dc/_features_rest"),
        ("G", "exact geometric occupancy equation", "occupancy = sigmoid(_occupancy)"),
        ("H", "exact optical opacity equation", "opacity = sigmoid(_opacity)"),
        ("I", "exact effective alpha / blending equation", "volume opacities=occupancy*opacity; surface opacities=occupancy; final=render_tran*transmissivity + render_scat*(1-transmissivity) (+ spec*reflectance after init)"),
        ("J", "geometric occupancy and optical opacity source-decoupled yes/no", "YES"),
        ("K", "J1", J1),
        ("L", "checkpoint serializable transparent/material state count", f"{len(serializable)} via PLY/module sidecar; capture/restore missing {len(capture_missing)}"),
        ("M", "J2a", J2a),
        ("N", "current critical dependency import pass/total count", f"{import_pass}/{import_total}"),
        ("O", "old migration failures still applicable", "torch2.0.1/cu118 mismatch; extension build risk; missing Python.h"),
        ("P", "runtime strategy A/B/C", "RUNTIME-C"),
        ("Q", "selected interpreter", "/usr/bin/python3"),
        ("R", "selected torch / torch CUDA", "2.6.0+cu124 via explicit user-site PYTHONPATH; clean PYTHONNOUSERSITE has no torch"),
        ("S", "critical extension import pass/total", f"{critical_ext_pass}/{critical_ext_total}"),
        ("T", "unresolved ldd dependency count", str(unresolved_ldd)),
        ("U", "J2b", J2b),
        ("V", "J2", J2),
        ("W", "native renderer forward finite/nonzero yes/no", "NOT_EXECUTED_J2_FAIL"),
        ("X", "render-active transparent/material state count", "0_NOT_EXECUTED_J2_FAIL"),
        ("Y", "render-active state names", "NOT_EXECUTED_J2_FAIL"),
        ("Z", "gradient-active transparent/material state count", "0_NOT_EXECUTED_J2_FAIL"),
        ("AA", "gradient-active state names", "NOT_EXECUTED_J2_FAIL"),
        ("AB", "J3", "NOT_EXECUTED_J2_FAIL"),
        ("AC", "K0/K1 GT identical yes/no", "NOT_EXECUTED_J2_FAIL"),
        ("AD", "K0/K2 GT different yes/no", "NOT_EXECUTED_J2_FAIL"),
        ("AE", "native carrier Gaussian count", "NOT_EXECUTED_J2_FAIL"),
        ("AF", "footprint policy", "NOT_EXECUTED_J2_FAIL"),
        ("AG", "exact canonical trainable state names", "NOT_EXECUTED_J2_FAIL"),
        ("AH", "K0 optimizer steps", "NOT_EXECUTED_J2_FAIL"),
        ("AI", "K1 optimizer steps", "NOT_EXECUTED_J2_FAIL"),
        ("AJ", "K2 optimizer steps", "NOT_EXECUTED_J2_FAIL"),
        ("AK", "K0 initial/final TRAIN loss", "NOT_EXECUTED_J2_FAIL"),
        ("AL", "K1 initial/final TRAIN loss", "NOT_EXECUTED_J2_FAIL"),
        ("AM", "K2 initial/final TRAIN loss", "NOT_EXECUTED_J2_FAIL"),
        ("AN", "first-step transparent/material parameter changed K0/K1/K2 yes/no", "NOT_EXECUTED_J2_FAIL"),
        ("AO", "frozen xyz max change", "NOT_EXECUTED_J2_FAIL"),
        ("AP", "checkpoint reload max error", "NOT_EXECUTED_J2_FAIL"),
        ("AQ", "fresh TEST RGB array count", "0"),
        ("AR", "metric reproduction max tau/PSNR error", "NOT_EXECUTED_J2_FAIL"),
        ("AS", "J4a", "NOT_EXECUTED_J2_FAIL"),
        ("AT", "K0 TEST PSNR/tau_eq Elog", "NOT_EXECUTED_J2_FAIL"),
        ("AU", "K1 TEST PSNR/tau_eq Elog", "NOT_EXECUTED_J2_FAIL"),
        ("AV", "K2 TEST PSNR/tau_eq Elog", "NOT_EXECUTED_J2_FAIL"),
        ("AW", "K0 capacity classification", "NOT_EXECUTED_J2_FAIL"),
        ("AX", "K1 capacity classification", "NOT_EXECUTED_J2_FAIL"),
        ("AY", "K2 capacity classification", "NOT_EXECUTED_J2_FAIL"),
        ("AZ", "J4b", "NOT_EXECUTED_J2_FAIL"),
        ("BA", "J4", "NOT_EXECUTED_J2_FAIL"),
        ("BB", "Stage5 vs Stage4 K0 PSNR delta", "NOT_EXECUTED_J2_FAIL"),
        ("BC", "Stage5 vs Stage4 K1 PSNR delta", "NOT_EXECUTED_J2_FAIL"),
        ("BD", "Stage5 vs Stage4 K2 PSNR delta", "NOT_EXECUTED_J2_FAIL"),
        ("BE", "ATTRIBUTE-STUDY-ELIGIBLE state count if J4 PASS", "0"),
        ("BF", "eligible state names if J4 PASS", "NOT_EXECUTED_J2_FAIL"),
        ("BG", "Final CASE", final_case),
        ("BH", "RT-native carrier ready yes/no", "NO"),
        ("BI", "AttributeDeformGS hypothesis status", "UNTESTED"),
        ("BJ", "allow Stage5.1 dynamic attribute sufficiency experiment yes/no", "NO"),
        ("BK", "PRIMARY ATTRIBUTE-DEFORMATION LINE STOP/CONTINUE", "STOP"),
        ("BL", "KIOT status", "CONTROLLED-CARRIER-ONLY"),
        ("BM", "next main research action", "Return to RecycleGS Stage1 cross-view geometry reliability detection"),
        ("BN", "report path", str(OUT / "stage5_0_rtsplat_native_carrier_report.md")),
        ("BO", "summary path", str(OUT / "stage5_0_rtsplat_native_carrier_summary.md")),
    ]
    final_text = "\n".join(f"{k}. {name}: {value}" for k, name, value in items) + "\n"
    write_text(OUT / "stage5_0_rtsplat_native_carrier_report.md", "# Stage5.0 RT-Splat Native Carrier Gate Report\n\n" + "\n".join(f"## {k}. {name}\n\n{value}\n" for k, name, value in items))
    write_text(OUT / "stage5_0_rtsplat_native_carrier_summary.md", f"# Stage5.0 summary\n\n- Final CASE: `{final_case}`\n- J0/J1/J2a/J2b/J2: {J0}/{J1}/{J2a}/{J2b}/{J2}\n- J3/J4: NOT_EXECUTED_J2_FAIL/NOT_EXECUTED_J2_FAIL\n- AttributeDeformGS hypothesis: UNTESTED\n- Primary attribute-deformation line: STOP\n")
    write_text(OUT / "stage5_0_rtsplat_native_carrier_log.txt", final_text)
    write_text(OUT / "final_terminal_summary.txt", final_text)
    print(final_text)


if __name__ == "__main__":
    main()
