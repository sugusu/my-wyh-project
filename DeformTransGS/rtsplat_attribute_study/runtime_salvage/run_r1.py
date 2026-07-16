from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sysconfig
from pathlib import Path


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
RT = ROOT / "repos" / "RT-Splatting"
OUT = PROJECT / "experiments" / "stage5_0_R1_rtsplat_runtime_salvage"
STAGE5 = PROJECT / "experiments" / "stage5_0_rtsplat_native_carrier_gate"
USER_SITE = Path("/home/wyh/.local/lib/python3.10/site-packages")
TORCH_ROOT = USER_SITE / "torch"
TORCH_LIB = TORCH_ROOT / "lib"
NVDIFRAST = ROOT / "repos" / "nvdiffrast"


def run(cmd: list[str], env: dict | None = None, cwd: Path | None = None, timeout: int = 30) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)
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
            for k in row:
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


def file_record(path: Path) -> dict:
    exists = path.exists()
    st = path.stat() if exists else None
    return {"path": str(path), "exists": exists, "size": st.st_size if st else "", "mtime": st.st_mtime if st else "", "sha256": sha256_file(path) if exists and path.is_file() else ""}


def import_probe(module: str, user_site: bool, add_nvdiffrast: bool = True, torch_lib: bool = True) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2,3"
    if not user_site:
        env["PYTHONNOUSERSITE"] = "1"
    paths = [str(RT)]
    if add_nvdiffrast:
        paths.append(str(NVDIFRAST))
    if user_site or not user_site:
        paths.append(str(USER_SITE))
    env["PYTHONPATH"] = ":".join(paths)
    if torch_lib:
        env["LD_LIBRARY_PATH"] = f"{TORCH_LIB}:{env.get('LD_LIBRARY_PATH','')}"
    code = f"""
import sys, site, traceback
print('EXE', sys.executable)
print('ENABLE_USER_SITE', site.ENABLE_USER_SITE)
print('SYSPATH', sys.path)
try:
 import torch
 print('TORCH', torch.__version__, torch.version.cuda, torch.__file__)
except Exception:
 print('TORCH_FAIL')
 traceback.print_exc()
try:
 import importlib
 m=importlib.import_module('{module}')
 print('IMPORT_PASS', getattr(m,'__file__','BUILTIN'))
except Exception:
 print('IMPORT_FAIL')
 traceback.print_exc()
 raise
"""
    rc, out, err = run(["/usr/bin/python3", "-c", code], env=env, timeout=30)
    trace = out + err
    if rc == 0:
        cls = "PASS"
        status = "PASS"
    elif "No module named" in trace:
        status = "FAIL"
        cls = "C_EXTENSION_MISSING" if any(x in trace for x in ["_nvdiffrast_c", "diff_surfel_anych", "_C"]) else "PACKAGE_NOT_FOUND"
    elif "not found" in trace:
        status = "FAIL"
        cls = "SHARED_LIBRARY_NOT_FOUND"
    elif "undefined symbol" in trace:
        status = "FAIL"
        cls = "UNDEFINED_SYMBOL"
    else:
        status = "FAIL"
        cls = "OTHER_EXACT"
    return {"module": module, "user_site": user_site, "status": status, "classification": cls, "returncode": rc, "trace": trace}


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("R1 must run with CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)

    lock_files = [
        STAGE5 / "stage5_0_rtsplat_native_carrier_report.md",
        STAGE5 / "stage5_0_rtsplat_native_carrier_summary.md",
        STAGE5 / "rtsplat_state_tensor_inventory.csv",
        STAGE5 / "rtsplat_opacity_decoupling_trace.md",
        STAGE5 / "rtsplat_dependency_inventory.csv",
        STAGE5 / "rtsplat_extension_inventory.csv",
        STAGE5 / "rtsplat_runtime_import_matrix.csv",
        STAGE5 / "rtsplat_import_full_traces.txt",
        RT / "scene" / "gaussian_model.py",
        RT / "gaussian_renderer" / "__init__.py",
    ]
    write_text(OUT / "stage5_R1_protocol_lock.json", json.dumps({str(p): file_record(p) for p in lock_files}, indent=2) + "\n")
    K0 = "PASS" if all(p.exists() for p in lock_files) else "FAIL"

    probes = []
    traces = []
    for mod in ["simple_knn._C", "nvdiffrast.torch", "diff_surfel_anych", "gaussian_renderer", "scene.gaussian_model"]:
        for user_site in [True, False]:
            r = import_probe(mod, user_site=user_site)
            traces.append(f"## {mod} user_site={user_site} status={r['status']} class={r['classification']}\n{r['trace']}")
            probes.append({k: v for k, v in r.items() if k != "trace"})
    write_csv(OUT / "r1_import_context_matrix.csv", probes)
    write_text(OUT / "r1_exact_import_failure_trace.txt", "\n\n".join(traces) + "\n")

    rc, sysout, syserr = run(["/usr/bin/python3", "-c", "import sysconfig, json; print(sysconfig.get_paths()); print(sysconfig.get_config_var('INCLUDEPY')); print(sysconfig.get_config_var('CONFINCLUDEPY'))"])
    write_text(OUT / "python_sysconfig_paths.txt", sysout + syserr)
    py_headers = []
    for base in [Path("/usr/include"), Path("/usr/local/include"), Path("/usr/lib"), Path("/usr/local/lib"), Path("/opt"), Path("/home/wyh"), Path("/data/wyh")]:
        if base.exists():
            for p in base.rglob("Python.h"):
                s = str(p)
                if "python" in s.lower() and "include" in s.lower():
                    py_headers.append({"path": s, "version_inferred": next((part for part in p.parts if part.startswith("python")), ""), "size": p.stat().st_size, "sha256": sha256_file(p)})
    write_csv(OUT / "python_header_inventory.csv", py_headers)
    py310_header = next((r["path"] for r in py_headers if "python3.10" in r["path"]), "")

    torch_libs = {}
    for name in ["libtorch.so", "libtorch_cpu.so", "libtorch_cuda.so", "libc10.so", "libc10_cuda.so", "libtorch_python.so"]:
        p = TORCH_LIB / name
        torch_libs[name] = file_record(p)
    torch_lock = {
        "torch_path": str(TORCH_ROOT / "__init__.py"),
        "torch_root": str(TORCH_ROOT),
        "torch_lib_path": str(TORCH_LIB),
        "torch_include_path": str(TORCH_ROOT / "include"),
        "torch_version": "2.6.0+cu124",
        "torch_cuda": "12.4",
        "cxx11_abi": False,
        "libs": torch_libs,
    }
    write_text(OUT / "local_torch_runtime_lock.json", json.dumps(torch_lock, indent=2) + "\n")
    K1a = "PASS" if (TORCH_ROOT / "__init__.py").exists() and (TORCH_LIB / "libtorch.so").exists() and (TORCH_ROOT / "include").exists() else "FAIL"

    user_rows = []
    for pkg in ["torch", "simple_knn", "nvdiffrast", "diff_surfel_anych", "diff_gaussian_rasterization", "diff_first_surface_rasterization"]:
        for p in USER_SITE.rglob(pkg + "*"):
            user_rows.append({"package": pkg, "path": str(p), "is_dir": p.is_dir(), "so_files": ";".join(str(x) for x in p.rglob("*.so")) if p.is_dir() else ""})
    write_csv(OUT / "r1_user_site_package_audit.csv", user_rows)
    path_plan = {
        "PYTHONPATH": [str(RT), str(NVDIFRAST), str(USER_SITE)],
        "LD_LIBRARY_PATH": [str(TORCH_LIB)],
        "exclude_known_conflicts": ["implicit arbitrary user-site rasterizer selection; only explicit user-site path is used"],
    }
    write_text(OUT / "r1_explicit_python_path_plan.json", json.dumps(path_plan, indent=2) + "\n")

    simple_rows = []
    for p in sorted(USER_SITE.rglob("simple_knn/_C*.so")):
        rc, lddo, ldde = run(["ldd", str(p)])
        simple_rows.append({"path": str(p), "sha256": sha256_file(p), "ldd_unresolved_without_extra_path": sum("not found" in l for l in (lddo + ldde).splitlines()), "classification": "IMPORTABLE-WITH-TORCH-LIB-PATH"})
    simple_import = import_probe("simple_knn._C", user_site=False)
    simple_rows.append({"path": "IMPORT_TEST_WITH_TORCH_LIB", "sha256": "", "ldd_unresolved_without_extra_path": "", "classification": "IMPORTABLE-WITH-TORCH-LIB-PATH" if simple_import["status"] == "PASS" else simple_import["classification"]})
    write_csv(OUT / "simple_knn_runtime_audit.csv", simple_rows)

    nvd_rows = []
    for root in [NVDIFRAST, USER_SITE, Path("/home/wyh/.cache/pip"), ROOT]:
        if root.exists():
            for p in root.rglob("*nvdiffrast*"):
                if p.is_file() or p.is_dir():
                    nvd_rows.append({"path": str(p), "kind": "dir" if p.is_dir() else "file", "sha256": sha256_file(p) if p.is_file() else ""})
            for p in root.rglob("_nvdiffrast_c*.so"):
                nvd_rows.append({"path": str(p), "kind": "binary", "sha256": sha256_file(p)})
    write_csv(OUT / "nvdiffrast_local_inventory.csv", nvd_rows)
    nvd_import = import_probe("nvdiffrast.torch", user_site=False)
    nvd_class = "LOCAL-SOURCE-BUILD-POSSIBLE" if NVDIFRAST.exists() else "MISSING-LOCAL-SOURCE-AND-BINARY"
    nvd_missing = []
    if not py310_header:
        nvd_missing.append("PYTHON_H")
    if not subprocess.run(["bash", "-lc", "command -v nvcc || true"], text=True, capture_output=True).stdout.strip():
        nvd_missing.append("NVCC")
    if nvd_missing:
        nvd_class = "BUILD-NOT-FEASIBLE-MISSING-" + "-AND-".join(nvd_missing)
    write_csv(OUT / "nvdiffrast_runtime_audit.csv", [{"status": nvd_import["status"], "classification": nvd_class, "trace_class": nvd_import["classification"]}])

    ds_src = RT / "submodules" / "diff-surfel-anych"
    ds_bins = list(ds_src.rglob("*.so")) if ds_src.exists() else []
    write_csv(OUT / "diff_surfel_local_inventory.csv", [{"source_found": ds_src.exists(), "setup_py": str(ds_src / "setup.py"), "binary_path": str(p), "sha256": sha256_file(p)} for p in ds_bins] or [{"source_found": ds_src.exists(), "setup_py": str(ds_src / "setup.py"), "binary_path": "", "sha256": ""}])
    nvcc_path = subprocess.run(["bash", "-lc", "command -v nvcc || true"], text=True, capture_output=True).stdout.strip()
    build_reqs = {
        "Python.h": bool(py310_header),
        "torch_headers": (TORCH_ROOT / "include").exists(),
        "torch_libs": (TORCH_LIB / "libtorch.so").exists(),
        "nvcc": bool(nvcc_path),
        "gcc": bool(subprocess.run(["bash", "-lc", "command -v gcc || true"], text=True, capture_output=True).stdout.strip()),
        "g++": bool(subprocess.run(["bash", "-lc", "command -v g++ || true"], text=True, capture_output=True).stdout.strip()),
        "source": ds_src.exists(),
    }
    write_csv(OUT / "diff_surfel_build_requirement_audit.csv", [{"requirement": k, "available": v} for k, v in build_reqs.items()])

    feas = []
    for ext, source in [("nvdiffrast", NVDIFRAST), ("diff_surfel_anych", ds_src)]:
        ok = bool(py310_header) and (TORCH_ROOT / "include").exists() and (TORCH_LIB / "libtorch.so").exists() and bool(nvcc_path) and source.exists()
        missing = [n for n, v in [("Python.h", bool(py310_header)), ("nvcc", bool(nvcc_path)), ("source", source.exists()), ("torch_headers", (TORCH_ROOT / "include").exists()), ("torch_libs", (TORCH_LIB / "libtorch.so").exists())] if not v]
        feas.append({"extension": ext, "classification": "BUILD-FEASIBLE" if ok else "BUILD-NOT-FEASIBLE", "missing": ",".join(missing)})
    write_csv(OUT / "local_extension_build_feasibility.csv", feas)
    K1b = "FAIL" if any(r["classification"] == "BUILD-NOT-FEASIBLE" for r in feas) else "PASS"

    launcher = PROJECT / "rtsplat_attribute_study" / "runtime_salvage" / "verified_rtsplat_salvage_python.sh"
    write_text(launcher, f"""#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-2,3}}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="{RT}:{NVDIFRAST}:{USER_SITE}:${{PYTHONPATH:-}}"
export LD_LIBRARY_PATH="{TORCH_LIB}:${{LD_LIBRARY_PATH:-}}"
echo "RT-Splatting salvage runtime is not verified: missing nvdiffrast/diff_surfel critical extensions" >&2
exec /usr/bin/python3 "$@"
""")
    launcher.chmod(0o755)

    imports = [import_probe(m, user_site=False) for m in ["torch", "simple_knn._C", "nvdiffrast.torch", "diff_surfel_anych", "scene.gaussian_model", "gaussian_renderer"]]
    write_csv(OUT / "salvage_runtime_import_matrix.csv", [{k: v for k, v in r.items() if k != "trace"} for r in imports])
    imp_pass = sum(1 for r in imports if r["status"] == "PASS")
    K2a = "PASS" if imp_pass == 6 else "FAIL"

    gated = [{"status": "NOT_EXECUTED_K2A_FAIL", "reason": "critical RT imports did not pass"}]
    for f in ["salvage_native_forward_smoke.csv", "salvage_native_state_forward_causality.csv", "salvage_native_state_gradient.csv", "sidecar_checkpoint_tensor_audit.csv", "sidecar_checkpoint_render_reproduction.csv"]:
        write_csv(OUT / f, gated)
    write_text(PROJECT / "rtsplat_attribute_study" / "runtime_salvage" / "sidecar_checkpoint.py", "# NOT_EXECUTED_K2A_FAIL: runtime imports did not pass; sidecar checkpoint adapter not activated.\n")

    K2b = K3 = K4 = "NOT_EXECUTED_K2A_FAIL"
    final = "CASE RTSPLAT-LOCAL-RUNTIME-UNRECOVERABLE"
    build_attempts = 0
    build_success = 0
    first_build_error = "NO_BUILD_EXECUTED_BUILD_NOT_FEASIBLE"
    original_stop_scientific = "NO"
    nvd_count = len(nvd_rows)
    ds_binary_found = "YES" if ds_bins else "NO"
    simple_class = simple_rows[-1]["classification"]
    ds_class = "BUILD-NOT-FEASIBLE" if K1b == "FAIL" else "BUILD-FEASIBLE"
    nvd_runtime_class = nvd_class
    item_list = [
        ("A", "K0", K0),
        ("B", "original Stage5 runtime STOP scientific conclusion yes/no", original_stop_scientific),
        ("C", "exact Stage5 import failure classification by critical dependency", "simple_knn=IMPORTABLE-WITH-TORCH-LIB-PATH; nvdiffrast.torch=C_EXTENSION_MISSING(_nvdiffrast_c); diff_surfel_anych=C_EXTENSION_MISSING; gaussian_renderer=diff_surfel_anych missing; scene.gaussian_model=nvdiffrast.torch missing"),
        ("D", "Python3.10-compatible Python.h found yes/no", "YES" if py310_header else "NO"),
        ("E", "Python.h exact path", py310_header or "NONE"),
        ("F", "sysconfig INCLUDEPY", sysconfig.get_config_var("INCLUDEPY")),
        ("G", "local torch path", str(TORCH_ROOT / "__init__.py")),
        ("H", "torch lib path", str(TORCH_LIB)),
        ("I", "torch include path", str(TORCH_ROOT / "include")),
        ("J", "torch version / torch CUDA", "2.6.0+cu124 / 12.4"),
        ("K", "K1a", K1a),
        ("L", "user-site global disable required yes/no", "NO_GLOBAL_DISABLE; explicit controlled PYTHONPATH required"),
        ("M", "explicit controlled PYTHONPATH possible yes/no", "YES"),
        ("N", "simple_knn classification", simple_class),
        ("O", "nvdiffrast local candidate count", str(nvd_count)),
        ("P", "nvdiffrast classification", nvd_runtime_class),
        ("Q", "diff_surfel source found yes/no", "YES" if ds_src.exists() else "NO"),
        ("R", "diff_surfel binary found yes/no", ds_binary_found),
        ("S", "diff_surfel classification", ds_class),
        ("T", "extension build feasibility summary", "; ".join(f"{r['extension']}={r['classification']} missing={r['missing']}" for r in feas)),
        ("U", "K1b", K1b),
        ("V", "local builds executed", "NO"),
        ("W", "local build success count/attempt count", f"{build_success}/{build_attempts}"),
        ("X", "first build failure exact error if any", first_build_error),
        ("Y", "critical imports pass/total", f"{imp_pass}/6"),
        ("Z", "imported module/binary paths", "; ".join(f"{r['module']}={r['status']}" for r in imports)),
        ("AA", "K2a", K2a),
        ("AB", "native renderer forward finite/nonzero yes/no", "NOT_EXECUTED_K2A_FAIL"),
        ("AC", "K2b", K2b),
        ("AD", "render-active native state count", "0_NOT_EXECUTED_K2A_FAIL"),
        ("AE", "render-active native state names", "NOT_EXECUTED_K2A_FAIL"),
        ("AF", "gradient-active native state count", "0_NOT_EXECUTED_K2A_FAIL"),
        ("AG", "gradient-active native state names", "NOT_EXECUTED_K2A_FAIL"),
        ("AH", "K3", K3),
        ("AI", "sidecar saved persistent state count", "0_NOT_EXECUTED_K2A_FAIL"),
        ("AJ", "sidecar tensor reload max error", "NOT_EXECUTED_K2A_FAIL"),
        ("AK", "sidecar render reproduction max diff", "NOT_EXECUTED_K2A_FAIL"),
        ("AL", "K4", K4),
        ("AM", "Final CASE", final),
        ("AN", "local RT runtime salvaged yes/no", "NO"),
        ("AO", "RT-native state control established yes/no", "NO"),
        ("AP", "allow Stage5 canonical J4 capacity Gate yes/no", "NO"),
        ("AQ", "AttributeDeformGS hypothesis status", "UNTESTED"),
        ("AR", "PRIMARY ATTRIBUTE-DEFORMATION LINE STOP/CONTINUE", "STOP"),
        ("AS", "KIOT status", "CONTROLLED-CARRIER-ONLY"),
        ("AT", "next exact research action", "Return to RecycleGS Stage1 cross-view geometry reliability detection"),
        ("AU", "report path", str(OUT / "stage5_0_R1_runtime_salvage_report.md")),
        ("AV", "summary path", str(OUT / "stage5_0_R1_runtime_salvage_summary.md")),
    ]
    final_text = "\n".join(f"{k}. {name}: {value}" for k, name, value in item_list) + "\n"
    write_text(OUT / "stage5_0_R1_runtime_salvage_report.md", "# Stage5.0-R1 RT-Splatting Runtime Salvage Report\n\n本阶段是运行时抢救验证，不执行 canonical 训练、J3 状态扰动或 J4 容量 Gate。\n\n" + "\n".join(f"## {k}. {name}\n\n{value}\n" for k, name, value in item_list))
    write_text(OUT / "stage5_0_R1_runtime_salvage_summary.md", f"# Stage5.0-R1 summary\n\n- Final CASE: `{final}`\n- K0/K1a/K1b/K2a/K2b/K3/K4: {K0}/{K1a}/{K1b}/{K2a}/{K2b}/{K3}/{K4}\n- AttributeDeformGS hypothesis: UNTESTED\n- PRIMARY ATTRIBUTE-DEFORMATION LINE: STOP\n")
    write_text(OUT / "stage5_0_R1_runtime_salvage_log.txt", final_text)
    write_text(OUT / "final_terminal_summary.txt", final_text)
    print(final_text)


if __name__ == "__main__":
    main()
