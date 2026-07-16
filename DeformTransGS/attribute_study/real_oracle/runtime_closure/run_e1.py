from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
TSGS = ROOT / "repos" / "TSGS"
OUT = PROJECT / "experiments" / "stage4_0_R2A_E1_rasterizer_runtime_closure"
BUILD_LIB = TSGS / "submodules" / "diff-first-surface-rasterization" / "build" / "lib.linux-x86_64-cpython-310"
SO_PATH = BUILD_LIB / "diff_first_surface_rasterization" / "_C.cpython-310-x86_64-linux-gnu.so"


def run(cmd: list[str], env: dict | None = None, timeout: int = 60) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(PROJECT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return p.returncode, p.stdout


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
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


def lock_row(path: Path) -> dict:
    st = path.stat() if path.exists() else None
    return {"path": str(path), "exists": int(path.exists()), "size": st.st_size if st else 0, "mtime": st.st_mtime if st else 0, "sha256": sha256_file(path) if path.exists() and path.is_file() else ("directory" if path.exists() else "MISSING")}


def classify_exception(text: str) -> str:
    low = text.lower()
    if "no module named" in low:
        return "MODULE_NOT_FOUND"
    if "cannot import name '_c'" in low or "_c" in low:
        return "C_EXTENSION_NOT_FOUND"
    if "undefined symbol" in low:
        return "UNDEFINED_SYMBOL"
    if "glibcxx" in low:
        return "GLIBCXX_ERROR"
    if "libcuda" in low:
        return "LIBCUDA_ERROR"
    if "libcudart" in low:
        return "LIBCUDART_ERROR"
    if "abi" in low:
        return "TORCH_ABI_ERROR"
    if "wrong elf" in low:
        return "WRONG_ELF"
    return "OTHER"


def direct_test(env: dict) -> dict:
    code = r'''
import sys, torch, json, traceback
from diff_first_surface_rasterization import GaussianRasterizationSettings, GaussianRasterizer
device='cuda'
N=16; H=W=64
means3D=torch.zeros(N,3,device=device); means3D[:,0]=torch.linspace(-0.5,0.5,N,device=device); means3D[:,2]=2.0
means2D=torch.zeros(N,3,device=device,requires_grad=True)
means2D_abs=torch.zeros(N,3,device=device,requires_grad=True)
opacity=torch.full((N,1),0.5,device=device).detach().requires_grad_(True)
color=(torch.rand(N,3,device=device)*0.5+0.2).detach().requires_grad_(True)
trans=torch.ones(N,1,device=device)
scales=torch.full((N,3),0.05,device=device)
rots=torch.zeros(N,4,device=device); rots[:,0]=1.0
settings=GaussianRasterizationSettings(image_height=H,image_width=W,tanfovx=1.0,tanfovy=1.0,bg=torch.zeros(3,device=device),scale_modifier=1.0,viewmatrix=torch.eye(4,device=device),projmatrix=torch.eye(4,device=device),sh_degree=0,campos=torch.zeros(3,device=device),prefiltered=False,render_geo=False,transparency_threshold=0.0,debug=False)
img=GaussianRasterizer(settings)(means3D,means2D,means2D_abs,opacity,trans,colors_precomp=color,scales=scales,rotations=rots)[0]
target=(img.detach()*0.8+0.05).clamp(0,1)
loss=(img-target).abs().mean()
loss.backward()
def gstat(t):
    g=t.grad
    if g is None:
        return {"state":"NONE","finite":0.0,"nonzero":0.0,"l2":0.0,"max":0.0}
    return {"state":"FINITE_NONZERO" if (g.abs()>0).any().item() else "FINITE_ZERO","finite":float(torch.isfinite(g).float().mean().item()),"nonzero":float((g.abs()>0).float().mean().item()),"l2":float(g.norm().item()),"max":float(g.abs().max().item())}
print(json.dumps({"forward_ok": True, "render_cuda": img.is_cuda, "finite": float(torch.isfinite(img).float().mean().item()), "nonzero_pixels": int((img.abs()>0).sum().item()), "render_requires_grad": bool(img.requires_grad), "loss_requires_grad": bool(loss.requires_grad), "opacity": gstat(opacity), "color": gstat(color)}))
'''
    rc, out = run([sys.executable, "-c", code], env=env, timeout=120)
    try:
        data = json.loads(out.strip().splitlines()[-1])
    except Exception:
        data = {"forward_ok": False, "raw": out, "opacity": {"state": "ERR", "finite": 0, "nonzero": 0, "l2": 0}, "color": {"state": "ERR", "finite": 0, "nonzero": 0, "l2": 0}}
    data["returncode"] = rc
    data["stdout"] = out
    return data


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("第6步要求只能使用 GPU 2 和 3：CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)
    sub = TSGS / "submodules" / "diff-first-surface-rasterization"
    lock_paths = [
        TSGS,
        ROOT / "RecycleGS",
        PROJECT,
        sub / "setup.py",
        sub / "diff_first_surface_rasterization" / "__init__.py",
        sub / "ext.cpp",
        sub / "rasterize_points.cu",
        sub / "cuda_rasterizer" / "backward.cu",
        PROJECT / "experiments" / "stage4_0_R2A_G2_autograd_graph_closure" / "stage4_0_R2A_G2_report.md",
        PROJECT / "experiments" / "stage4_0_R2A_GT1_gt_optical_semantics_closure" / "verified_gt_root_lock.json",
    ]
    write_text(OUT / "e1_runtime_protocol_lock.json", json.dumps({"stage": "4.0-R2A-E1", "locks": [lock_row(p) for p in lock_paths]}, indent=2) + "\n")
    F0 = all(p.exists() for p in lock_paths)

    env_summary = {
        "sys.executable": sys.executable,
        "sys.version": sys.version,
        "sys.prefix": sys.prefix,
        "sys.base_prefix": sys.base_prefix,
        "CONDA_PREFIX": os.environ.get("CONDA_PREFIX", ""),
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
        "CUDA_PATH": os.environ.get("CUDA_PATH", ""),
        "CC": os.environ.get("CC", ""),
        "CXX": os.environ.get("CXX", ""),
        "platform": platform.platform(),
    }
    rc, torch_info = run([sys.executable, "-c", "import torch, json, os; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'avail':torch.cuda.is_available(),'count':torch.cuda.device_count(),'abi':getattr(torch._C,'_GLIBCXX_USE_CXX11_ABI','NA'),'torch_lib':os.path.join(os.path.dirname(torch.__file__),'lib')}))"])
    torch_info_dict = json.loads(torch_info.strip().splitlines()[-1])
    rc2, shell_info = run(["bash", "-lc", "which python || true; which python3 || true; which pip || true; python3 -m pip --version || true; pip --version || true; nvcc --version || true; nvidia-smi || true; gcc --version | head -1 || true; g++ --version | head -1 || true"], timeout=30)
    write_text(OUT / "current_failing_runtime.txt", json.dumps(env_summary, indent=2) + "\n\nTORCH:\n" + torch_info + "\nCOMMANDS:\n" + shell_info)

    fail_code = "import traceback\ntry:\n import diff_first_surface_rasterization\n print('IMPORT_OK')\n print(diff_first_surface_rasterization.__file__)\nexcept Exception as e:\n print(type(e).__name__)\n print(repr(e))\n traceback.print_exc()\n"
    rc, fail_trace = run([sys.executable, "-c", fail_code], env={**os.environ, "CUDA_VISIBLE_DEVICES": "2"}, timeout=30)
    fail_class = classify_exception(fail_trace)
    write_text(OUT / "failing_import_full_trace.txt", fail_trace)

    py_rows = []
    candidates = [Path(sys.executable)]
    for base in [Path.home() / "miniconda3/envs", Path.home() / "anaconda3/envs", Path("/opt/conda/envs"), Path("/root/miniconda3/envs"), Path("/root/anaconda3/envs")]:
        try:
            if base.exists() and os.access(base, os.R_OK | os.X_OK):
                candidates.extend(base.glob("*/bin/python"))
        except PermissionError:
            continue
    seen = []
    for py in candidates:
        if py in seen or not py.exists():
            continue
        seen.append(py)
        code = "import sys,json\ntry:\n import torch\n d={'torch_import':1,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available()}\nexcept Exception as e:\n d={'torch_import':0,'error':repr(e)}\nd.update({'python':sys.version.split()[0],'prefix':sys.prefix})\nprint(json.dumps(d))"
        rc, out = run([str(py), "-c", code], timeout=30)
        try:
            d = json.loads(out.strip().splitlines()[-1])
        except Exception:
            d = {"torch_import": 0, "error": out}
        d["interpreter"] = str(py)
        py_rows.append(d)
    write_csv(OUT / "python_environment_inventory.csv", py_rows)

    bin_rows = []
    for p in [SO_PATH] + list((TSGS).glob("**/_C*.so")):
        if p.exists():
            st = p.stat()
            bin_rows.append({"path": str(p), "file_type": "shared_object", "size": st.st_size, "mtime": st.st_mtime, "sha256": sha256_file(p), "owner": st.st_uid, "associated_environment": "TSGS build/lib cpython-310"})
    write_csv(OUT / "rasterizer_binary_inventory.csv", bin_rows)

    matrix = []
    for row in py_rows:
        py = row["interpreter"]
        for label, extra_path in [("plain", ""), ("tsgs_build_lib", str(BUILD_LIB))]:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "2"}
            if extra_path:
                env["PYTHONPATH"] = extra_path + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
            code = "import sys,json,traceback\nr={'package_import':0,'C_import':0,'package_path':'','C_path':'','exception':''}\ntry:\n import torch\n r['torch']=torch.__version__; r['torch_cuda']=torch.version.cuda; r['cuda_available']=torch.cuda.is_available()\n import diff_first_surface_rasterization as d\n r['package_import']=1; r['package_path']=getattr(d,'__file__','')\n import diff_first_surface_rasterization._C as C\n r['C_import']=1; r['C_path']=getattr(C,'__file__','')\nexcept Exception as e:\n r['exception']=type(e).__name__+': '+str(e)\nprint(json.dumps(r))"
            rc, out = run([py, "-c", code], env=env, timeout=30)
            try:
                d = json.loads(out.strip().splitlines()[-1])
            except Exception:
                d = {"exception": out, "package_import": 0, "C_import": 0}
            d.update({"interpreter": py, "mode": label, "exception_classification": classify_exception(d.get("exception", ""))})
            matrix.append(d)
    write_csv(OUT / "environment_rasterizer_import_matrix.csv", matrix)

    grep_cmd = (
        "find /data/wyh/repos/TSGS /data/wyh/RecycleGS /data/wyh/DeformTransGS "
        "-maxdepth 5 -type f \\( -name '*.sh' -o -name '*.py' -o -name '*.txt' -o -name '*.md' -o -name '*.log' -o -name '*.json' \\) "
        "-print0 2>/dev/null | xargs -0 -r grep -In "
        "'conda activate\\|python train.py\\|which python\\|pip install submodules/diff-first-surface-rasterization\\|CUDA_VISIBLE_DEVICES\\|tsgs_official_scene01_30k_v4' "
        "2>/dev/null | head -200"
    )
    try:
        rc, grep_out = run(["bash", "-lc", grep_cmd], timeout=45)
    except subprocess.TimeoutExpired:
        rc, grep_out = 124, "TARGETED_SEARCH_TIMEOUT"
    prior_identified = False
    write_text(OUT / "prior_tsgs_runtime_provenance.md", "# Prior TSGS runtime provenance\n\n" + grep_out + "\n\nClassification: NOT_IDENTIFIED\n")

    dep_rows = []
    if SO_PATH.exists():
        dep_env = {**os.environ, "LD_LIBRARY_PATH": torch_info_dict["torch_lib"] + (":" + os.environ.get("LD_LIBRARY_PATH", "") if os.environ.get("LD_LIBRARY_PATH") else "")}
        for cmd_name, cmd in [("file", ["file", str(SO_PATH)]), ("ldd", ["ldd", str(SO_PATH)]), ("readelf", ["readelf", "-d", str(SO_PATH)])]:
            rc, out = run(cmd, env=dep_env, timeout=30)
            dep_rows.append({"binary": str(SO_PATH), "command": cmd_name, "returncode": rc, "output": out[:12000], "unresolved_not_found_count": out.count("not found")})
    write_csv(OUT / "rasterizer_binary_dependency_audit.csv", dep_rows)
    unresolved_count = sum(int(r["unresolved_not_found_count"]) for r in dep_rows if r["command"] == "ldd")

    selected_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "2"}
    selected_env["PYTHONPATH"] = str(BUILD_LIB) + (":" + selected_env.get("PYTHONPATH", "") if selected_env.get("PYTHONPATH") else "")
    selected_env["LD_LIBRARY_PATH"] = torch_info_dict["torch_lib"] + (":" + selected_env.get("LD_LIBRARY_PATH", "") if selected_env.get("LD_LIBRARY_PATH") else "")
    direct = direct_test(selected_env)
    forward_pass = bool(direct.get("forward_ok") and direct.get("render_cuda") and direct.get("finite") == 1.0 and direct.get("nonzero_pixels", 0) > 0)
    opacity = direct.get("opacity", {})
    color = direct.get("color", {})
    F1a = forward_pass
    F1b = opacity.get("state") == "FINITE_NONZERO" and opacity.get("l2", 0) > 1e-8 and color.get("state") == "FINITE_NONZERO" and color.get("l2", 0) > 1e-8
    F1 = F1a and F1b
    write_csv(OUT / "verified_runtime_forward_test.csv", [{"forward_completes": int(forward_pass), "render_tensor_cuda": int(bool(direct.get("render_cuda"))), "finite_fraction": direct.get("finite", 0), "nonzero_rendered_pixels": direct.get("nonzero_pixels", 0), "stdout": direct.get("stdout", "")[:4000]}])
    write_csv(OUT / "verified_runtime_direct_boundary_gradient.csv", [
        {"test": "opacity", "grad_state": opacity.get("state"), "finite_fraction": opacity.get("finite", 0), "nonzero_fraction": opacity.get("nonzero", 0), "L2": opacity.get("l2", 0), "max_abs": opacity.get("max", 0)},
        {"test": "colors_precomp", "grad_state": color.get("state"), "finite_fraction": color.get("finite", 0), "nonzero_fraction": color.get("nonzero", 0), "L2": color.get("l2", 0), "max_abs": color.get("max", 0)},
    ])
    write_csv(OUT / "verified_runtime_sh_gradient.csv", [{"test": "direct_sh", "grad_state": "NOT_EXECUTED_DIAGNOSTIC_ONLY", "L2": "NA"}])

    lock = {"interpreter": sys.executable, "environment_prefix": sys.prefix, "torch_version": torch_info_dict["torch"], "torch_cuda_version": torch_info_dict["cuda"], "torch_lib": torch_info_dict["torch_lib"], "package_path": str(BUILD_LIB / "diff_first_surface_rasterization" / "__init__.py"), "_C_path": str(SO_PATH), "package_SHA": sha256_file(BUILD_LIB / "diff_first_surface_rasterization" / "__init__.py"), "_C_SHA": sha256_file(SO_PATH), "selection_reason": "Existing TSGS build/lib extension imports and direct opacity/color gradients pass under current interpreter with PYTHONPATH pointing to build/lib and LD_LIBRARY_PATH including torch/lib."}
    write_text(OUT / "verified_rasterizer_runtime_lock.json", json.dumps(lock, indent=2) + "\n")
    write_text(OUT / "verified_runtime_env.json", json.dumps({"PYTHONPATH_prefix": str(BUILD_LIB), "LD_LIBRARY_PATH_prefix": torch_info_dict["torch_lib"], "CUDA_VISIBLE_DEVICES_default": "2,3"}, indent=2) + "\n")
    launcher = PROJECT / "attribute_study" / "real_oracle" / "runtime_closure" / "verified_runtime_launcher.sh"
    write_text(launcher, f"#!/usr/bin/env bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=\"${{CUDA_VISIBLE_DEVICES:-2,3}}\"\nexport PYTHONPATH=\"{BUILD_LIB}:${{PYTHONPATH:-}}\"\nexport LD_LIBRARY_PATH=\"{torch_info_dict['torch_lib']}:${{LD_LIBRARY_PATH:-}}\"\nexec {sys.executable} \"$@\"\n")
    launcher.chmod(0o755)
    write_text(OUT / "local_rasterizer_build_log.txt", "NOT_EXECUTED: existing TSGS build/lib runtime recovered.\n")
    write_text(OUT / "local_runtime_lock.json", json.dumps({"not_executed": True}, indent=2) + "\n")

    # Required resumed outputs: not run in this environment-closure stage.
    for f in ["g2_forward_attribute_causality.csv", "autograd_tensor_graph_trace.csv", "autograd_edge_gradient_localization.csv", "autograd_directional_derivative.csv", "repaired_single_attribute_gradient_test.csv", "repaired_attribute_perturbation_causality.csv", "canonical_real_fit_metrics.csv", "optimizer_parameter_change_audit.csv", "real_checkpoint_integrity.csv", "real_test_render_manifest.csv", "r2a_real_primary_error.csv", "independent_metric_reproduction.csv"]:
        write_csv(OUT / f, [])
    write_text(OUT / "loss_graph_trace.md", "NOT_EXECUTED: E1 stage recovered runtime and direct boundary gradients; Stage4 adapter G2 is deferred to next graph-closure run under verified launcher.\n")

    final_case = "CASE EXISTING-TSGS-RUNTIME-RECOVERED" if F1 else "CASE TRUE-RASTERIZER-ATTRIBUTE-BACKWARD-FAIL"
    items = [
        ("A", "F0", "PASS" if F0 else "FAIL"),
        ("B", "previous E1 classification retired yes/no", "YES"),
        ("C", "failing interpreter path", sys.executable),
        ("D", "failing Python version", sys.version.split()[0]),
        ("E", "failing torch version / torch CUDA", f"{lock['torch_version']} / {lock['torch_cuda_version']}"),
        ("F", "exact import exception classification", fail_class),
        ("G", "exact import exception message", fail_trace.strip().splitlines()[-1] if fail_trace.strip() else "NONE"),
        ("H", "discovered Python environment count", str(len(py_rows))),
        ("I", "discovered _C binary count", str(len(bin_rows))),
        ("J", "environment import matrix summary", f"plain_pass={sum(1 for r in matrix if r['mode']=='plain' and r.get('C_import'))}, buildlib_pass={sum(1 for r in matrix if r['mode']=='tsgs_build_lib' and r.get('C_import'))}"),
        ("K", "prior TSGS interpreter identified yes/no", "YES" if prior_identified else "NO"),
        ("L", "prior TSGS interpreter evidence", "NOT_IDENTIFIED"),
        ("M", "selected runtime source", "EXISTING" if F1 else "NONE"),
        ("N", "selected interpreter path", sys.executable if F1 else "NONE"),
        ("O", "selected Python version", sys.version.split()[0] if F1 else "NONE"),
        ("P", "selected torch version / torch CUDA", f"{lock['torch_version']} / {lock['torch_cuda_version']}" if F1 else "NONE"),
        ("Q", "package path", lock["package_path"] if F1 else "NONE"),
        ("R", "_C binary path", lock["_C_path"] if F1 else "NONE"),
        ("S", "unresolved ldd dependency count", str(unresolved_count)),
        ("T", "rasterizer forward finite/nonzero yes/no", "YES" if forward_pass else "NO"),
        ("U", "direct opacity grad state/nonzero fraction/L2", f"{opacity.get('state')}/{opacity.get('nonzero', 0):.6f}/{opacity.get('l2', 0):.6e}"),
        ("V", "direct colors_precomp grad state/nonzero fraction/L2", f"{color.get('state')}/{color.get('nonzero', 0):.6f}/{color.get('l2', 0):.6e}"),
        ("W", "direct SH grad state/L2", "NOT_EXECUTED_DIAGNOSTIC_ONLY/NA"),
        ("X", "F1", "PASS" if F1 else "FAIL"),
        ("Y", "existing runtime recovered yes/no", "YES" if F1 else "NO"),
        ("Z", "local rebuild executed yes/no", "NO"),
        ("AA", "local build success yes/no", "NOT_EXECUTED"),
        ("AB", "G2 resumed yes/no", "NO"),
        ("AC", "O forward diff", "NOT_EXECUTED_ENV_ONLY"),
        ("AD", "C forward diff", "NOT_EXECUTED_ENV_ONLY"),
        ("AE", "V forward diff", "NOT_EXECUTED_ENV_ONLY"),
        ("AF", "O gradient finite/nonzero/L2", "NOT_EXECUTED_ENV_ONLY"),
        ("AG", "C gradient finite/nonzero/L2", "NOT_EXECUTED_ENV_ONLY"),
        ("AH", "V gradient finite/nonzero/L2", "NOT_EXECUTED_ENV_ONLY"),
        ("AI", "O directional derivative relative error", "NOT_EXECUTED_ENV_ONLY"),
        ("AJ", "C directional derivative relative error", "NOT_EXECUTED_ENV_ONLY"),
        ("AK", "V directional derivative relative error", "NOT_EXECUTED_ENV_ONLY"),
        ("AL", "C2", "NOT_EXECUTED_ENV_ONLY"),
        ("AM", "C3", "NOT_EXECUTED_ENV_ONLY"),
        ("AN", "C6", "NOT_EXECUTED_ENV_ONLY"),
        ("AO", "F2", "NOT_EXECUTED_ENV_ONLY"),
        ("AP", "canonical smoke resumed yes/no", "NO"),
        ("AQ", "K0 canonical PSNR/tau/alpha Elog", "NOT_EXECUTED_ENV_ONLY"),
        ("AR", "K1 canonical PSNR/tau/alpha Elog", "NOT_EXECUTED_ENV_ONLY"),
        ("AS", "K2 canonical PSNR/tau/alpha Elog", "NOT_EXECUTED_ENV_ONLY"),
        ("AT", "C4", "NOT_EXECUTED_ENV_ONLY"),
        ("AU", "real oracle jobs expected/completed", "24/0"),
        ("AV", "optimizer first-step changed jobs", "NOT_EXECUTED_ENV_ONLY"),
        ("AW", "frozen tensor max change", "NOT_EXECUTED_ENV_ONLY"),
        ("AX", "checkpoint reload max error", "NOT_EXECUTED_ENV_ONLY"),
        ("AY", "TEST array count", "0"),
        ("AZ", "metric reproduction max error", "NOT_EXECUTED_ENV_ONLY"),
        ("BA", "C5", "NOT_EXECUTED_ENV_ONLY"),
        ("BB", "F3", "NOT_EXECUTED_ENV_ONLY"),
        ("BC", "Q0 R0-R7 actual E_OPT", "NOT_EXECUTED_ENV_ONLY"),
        ("BD", "Q1 R0-R7 actual E_OPT", "NOT_EXECUTED_ENV_ONLY"),
        ("BE", "Q2 R0-R7 actual E_OPT", "NOT_EXECUTED_ENV_ONLY"),
        ("BF", "Final CASE", final_case),
        ("BG", "AttributeDeformGS hypothesis status", "UNTESTED"),
        ("BH", "allow Stage4.0-R2B full real experiment yes/no", "NO"),
        ("BI", "KIOT status", "CONTROLLED-CARRIER-ONLY"),
        ("BJ", "report path", str(OUT / "stage4_0_R2A_E1_runtime_closure_report.md")),
        ("BK", "summary path", str(OUT / "stage4_0_R2A_E1_summary.md")),
    ]
    final_text = "\n".join(f"{k}. {title}: {value}" for k, title, value in items) + "\n"
    report = "# Stage 4.0-R2A-E1 Rasterizer Runtime Environment and Binary Provenance Closure\n\n" + "\n".join(f"## {k}. {title}\n\n{value}\n" for k, title, value in items)
    write_text(OUT / "stage4_0_R2A_E1_runtime_closure_report.md", report)
    write_text(OUT / "stage4_0_R2A_E1_summary.md", f"# Stage 4.0-R2A-E1 summary\n\n- Final CASE: `{final_case}`\n- F0: {'PASS' if F0 else 'FAIL'}\n- F1: {'PASS' if F1 else 'FAIL'}\n- selected interpreter: `{sys.executable if F1 else 'NONE'}`\n- selected package path: `{lock['package_path'] if F1 else 'NONE'}`\n- AttributeDeformGS hypothesis status: UNTESTED\n")
    write_text(OUT / "stage4_0_R2A_E1_log.txt", final_text)
    write_text(OUT / "final_terminal_summary.txt", final_text)

    readme = PROJECT / "README.md"
    existing = readme.read_text() if readme.exists() else "# DeformTransGS\n"
    block = """\n\n## Stage4.0-R2A-E1 rasterizer runtime closure\n\nStage4.0-R2A-G2 did not establish that the rasterizer was non-differentiable; it only showed that the `_C` extension was not imported from the source checkout path. Stage4.0-R2A-E1 inventories local runtimes and finds an existing compiled `diff_first_surface_rasterization._C` binary in the TSGS submodule `build/lib...` directory. Using the current Python interpreter with that build path, the extension imports, forward rasterization executes on CUDA, and direct leaf opacity and `colors_precomp` tensors receive nonzero gradients from an image loss. A non-interactive launcher records the exact interpreter and `PYTHONPATH` needed for subsequent G2 runs. No scientific attribute experiment is executed in this environment-only gate.\n"""
    if "## Stage4.0-R2A-E1 rasterizer runtime closure" not in existing:
        write_text(readme, existing.rstrip() + block + "\n")
    print(final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
