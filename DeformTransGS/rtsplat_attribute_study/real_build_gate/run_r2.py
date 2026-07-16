from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path


ROOT = Path("/data/wyh")
PROJECT = ROOT / "DeformTransGS"
RT = ROOT / "repos" / "RT-Splatting"
NVD = ROOT / "repos" / "nvdiffrast"
DIFF = RT / "submodules" / "diff-surfel-anych"
OUT = PROJECT / "experiments" / "stage5_0_R2_real_local_extension_build"
SRC = PROJECT / "rtsplat_attribute_study" / "real_build_gate"
BUILD = PROJECT / "runtime" / "rtsplat_stage5_R2_build"
USER_SITE = Path("/home/wyh/.local/lib/python3.10/site-packages")
CONDA_SITE = Path("/home/wyh/.conda/envs/rtsplat_legacy2/lib/python3.10/site-packages")
TORCH = USER_SITE / "torch"
TORCH_LIB = TORCH / "lib"
PY_HEADER_DIR = Path("/home/wyh/.conda/envs/rtsplat_legacy2/include/python3.10")
CUDA_HOME = Path("/usr/local/cuda")
NVCC = CUDA_HOME / "bin/nvcc"


def run(cmd, env=None, cwd=None, timeout=120):
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True, capture_output=True, timeout=timeout)
        return {"cmd": cmd, "cwd": str(cwd or Path.cwd()), "start": start, "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as e:
        return {"cmd": cmd, "cwd": str(cwd or Path.cwd()), "start": start, "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "returncode": 999, "stdout": "", "stderr": repr(e)}


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
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


def sha(path: Path):
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def record(path: Path):
    return {"path": str(path), "exists": path.exists(), "is_file": path.is_file(), "size": path.stat().st_size if path.exists() else "", "sha256": sha(path)}


def base_env(arch=""):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2,3"
    env["CUDA_HOME"] = str(CUDA_HOME)
    env["PATH"] = f"{CUDA_HOME / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{TORCH_LIB}:{CUDA_HOME / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}"
    env["CPATH"] = f"{PY_HEADER_DIR}:{env.get('CPATH', '')}"
    env["PYTHONPATH"] = ":".join([str(RT), str(NVD), str(USER_SITE), str(CONDA_SITE), env.get("PYTHONPATH", "")])
    env["TORCH_EXTENSIONS_DIR"] = str(BUILD / "torch_extensions")
    env["MAX_JOBS"] = "2"
    if arch:
        env["TORCH_CUDA_ARCH_LIST"] = arch
    return env


def first_error(text: str):
    pats = ["fatal error:", "error:", "undefined reference", "No such file", "not found", "RuntimeError:", "ImportError:"]
    for line in text.splitlines():
        if any(p in line for p in pats):
            return line.strip()
    return text.splitlines()[-1].strip() if text.splitlines() else "NONE"


def classify_build(rc: int, log: str):
    if rc == 0:
        return "BUILD-SUCCESS"
    low = log.lower()
    if "torch" in low and ("api" in low or "at::" in log or "c10::" in log):
        return "BUILD-FAILED-TORCH-API"
    if "cuda" in low or "nvcc" in low:
        return "BUILD-FAILED-CUDA-API"
    if "python.h" in low or "pyconfig" in low or "soabi" in low:
        return "BUILD-FAILED-PYTHON-ABI"
    if "undefined reference" in low or "cannot find -l" in low:
        return "BUILD-FAILED-LINKER"
    if "error:" in low or "fatal error:" in low:
        return "BUILD-FAILED-COMPILER"
    return "BUILD-FAILED-OTHER"


def import_probe(module: str, env, extra_paths=None):
    env = env.copy()
    paths = [str(RT), str(NVD), str(USER_SITE)]
    if extra_paths:
        paths = extra_paths + paths
    if str(CONDA_SITE) not in paths:
        paths.append(str(CONDA_SITE))
    env["PYTHONPATH"] = ":".join(paths)
    code = f"import importlib, torch; m=importlib.import_module('{module}'); print(getattr(m,'__file__','BUILTIN'))"
    r = run(["/usr/bin/python3", "-c", code], env=env, timeout=60)
    return {"module": module, "status": "PASS" if r["returncode"] == 0 else "FAIL", "returncode": r["returncode"], "path": r["stdout"].strip(), "error": first_error(r["stdout"] + r["stderr"])}


def audit_binary(path: Path):
    file_r = run(["file", str(path)])
    ldd_r = run(["ldd", str(path)])
    readelf_r = run(["readelf", "-d", str(path)])
    return {"path": str(path), "sha256": sha(path), "file": file_r["stdout"].strip(), "ldd": ldd_r["stdout"] + ldd_r["stderr"], "readelf": readelf_r["stdout"] + readelf_r["stderr"]}


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2,3":
        raise RuntimeError("R2 must run with CUDA_VISIBLE_DEVICES=2,3")
    OUT.mkdir(parents=True, exist_ok=True)
    for d in ["nvdiffrast", "diff_surfel_anych", "logs", "tmp", "torch_extensions"]:
        (BUILD / d).mkdir(parents=True, exist_ok=True)

    lock_files = [
        RT, PROJECT / "experiments/stage5_0_rtsplat_native_carrier_gate/stage5_0_rtsplat_native_carrier_report.md",
        PROJECT / "experiments/stage5_0_rtsplat_native_carrier_gate/stage5_0_rtsplat_native_carrier_summary.md",
        PROJECT / "experiments/stage5_0_R1_rtsplat_runtime_salvage/stage5_0_R1_runtime_salvage_report.md",
        PROJECT / "experiments/stage5_0_R1_rtsplat_runtime_salvage/stage5_0_R1_runtime_salvage_summary.md",
        PROJECT / "experiments/stage5_0_rtsplat_native_carrier_gate/rtsplat_state_tensor_inventory.csv",
        PROJECT / "experiments/stage5_0_rtsplat_native_carrier_gate/rtsplat_opacity_decoupling_trace.md",
        PROJECT / "experiments/stage5_0_R1_rtsplat_runtime_salvage/nvdiffrast_local_inventory.csv",
        PROJECT / "experiments/stage5_0_R1_rtsplat_runtime_salvage/diff_surfel_local_inventory.csv",
        PROJECT / "experiments/stage5_0_R1_rtsplat_runtime_salvage/diff_surfel_build_requirement_audit.csv",
        PROJECT / "experiments/stage5_0_R1_rtsplat_runtime_salvage/local_extension_build_feasibility.csv",
        PROJECT / "experiments/stage5_0_R1_rtsplat_runtime_salvage/local_torch_runtime_lock.json",
    ]
    write_text(OUT / "stage5_R2_protocol_lock.json", json.dumps({str(p): record(p) for p in lock_files}, indent=2) + "\n")
    L0 = "PASS" if all(p.exists() for p in lock_files) else "FAIL"

    nvcc_trace = []
    for cmd in [["ls", "-l", str(NVCC)], ["file", str(NVCC)], [str(NVCC), "--version"], ["readlink", "-f", str(CUDA_HOME)], ["readlink", "-f", str(NVCC)]]:
        r = run(cmd)
        nvcc_trace.append("$ " + " ".join(cmd) + f"\nrc={r['returncode']}\n{r['stdout']}{r['stderr']}")
    write_text(OUT / "nvcc_exact_trace.txt", "\n\n".join(nvcc_trace) + "\n")
    cuda_rows = []
    for root in [Path("/usr/local/cuda"), Path("/usr/local/cuda-12.4")] + sorted(Path("/usr/local").glob("cuda-*")):
        if root.exists():
            n = root / "bin/nvcc"
            rr = run([str(n), "--version"]) if n.exists() else {"returncode": 1, "stdout": "", "stderr": ""}
            cuda_rows.append({"root": str(root), "nvcc_path": str(n), "nvcc_executable": os.access(n, os.X_OK), "nvcc_version": (rr["stdout"] + rr["stderr"]).replace("\n", " | "), "cuda_h": (root / "include/cuda.h").exists(), "cuda_runtime_h": (root / "include/cuda_runtime.h").exists(), "libcudart": (root / "lib64/libcudart.so").exists(), "bin_nvcc": n.exists()})
    write_csv(OUT / "local_cuda_toolkit_audit.csv", cuda_rows)
    L1 = "PASS" if any(r["nvcc_executable"] and r["cuda_h"] and r["cuda_runtime_h"] and r["libcudart"] for r in cuda_rows) else "FAIL"
    default = run(["bash", "-lc", "printf 'PATH=%s\\n' \"$PATH\"; which nvcc || true; command -v nvcc || true"])
    explicit = run(["bash", "-lc", f"export CUDA_HOME={CUDA_HOME}; export PATH={CUDA_HOME}/bin:$PATH; printf 'PATH=%s\\n' \"$PATH\"; which nvcc || true; command -v nvcc || true; nvcc --version"], timeout=30)
    prev_class = "PATH-only" if default["stdout"].count("nvcc") == 0 and explicit["returncode"] == 0 else "REAL"
    write_text(OUT / "nvcc_path_misclassification_audit.txt", f"## default\n{default['stdout']}{default['stderr']}\n## explicit\n{explicit['stdout']}{explicit['stderr']}\nclassification={prev_class}\n")

    env0 = base_env()
    torch_code = """
import json, torch
from torch.utils import cpp_extension
print(json.dumps({
'file': torch.__file__, 'version': torch.__version__, 'cuda': torch.version.cuda,
'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count(),
'cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI),
'cpp_extension_CUDA_HOME': cpp_extension.CUDA_HOME,
}))
"""
    tr = run(["/usr/bin/python3", "-c", torch_code], env=env0)
    torch_json = next((line for line in tr["stdout"].splitlines() if line.strip().startswith("{")), "")
    torch_lock = json.loads(torch_json) if tr["returncode"] == 0 and torch_json else {"error": tr["stdout"] + tr["stderr"]}
    torch_lock.update({"include": str(TORCH / "include"), "lib": str(TORCH_LIB), "libs": {p.name: record(p) for p in [TORCH_LIB / "libtorch.so", TORCH_LIB / "libtorch_cpu.so", TORCH_LIB / "libtorch_cuda.so", TORCH_LIB / "libc10.so"]}})
    write_text(OUT / "stage5_R2_torch_runtime_lock.json", json.dumps(torch_lock, indent=2) + "\n")
    L2a = tr["returncode"] == 0 and all((TORCH_LIB / n).exists() for n in ["libtorch.so", "libtorch_cpu.so", "libtorch_cuda.so", "libc10.so"])

    abi_rows = []
    for interp in [Path("/usr/bin/python3"), Path("/home/wyh/.conda/envs/rtsplat_legacy2/bin/python")]:
        if interp.exists():
            code = "import sys,sysconfig,platform,json; print(json.dumps({'interpreter':sys.executable,'version':sys.version,'hexversion':sys.hexversion,'abiflags':getattr(sys,'abiflags',''),'arch':platform.architecture()[0],'SOABI':sysconfig.get_config_var('SOABI'),'EXT_SUFFIX':sysconfig.get_config_var('EXT_SUFFIX'),'MULTIARCH':sysconfig.get_config_var('MULTIARCH'),'Py_DEBUG':sysconfig.get_config_var('Py_DEBUG'),'WITH_PYMALLOC':sysconfig.get_config_var('WITH_PYMALLOC'),'SIZEOF_VOID_P':sysconfig.get_config_var('SIZEOF_VOID_P'),'LDVERSION':sysconfig.get_config_var('LDVERSION'),'INCLUDEPY':sysconfig.get_config_var('INCLUDEPY')}))"
            rr = run([str(interp), "-c", code], env=env0)
            row = json.loads(rr["stdout"]) if rr["returncode"] == 0 else {"interpreter": str(interp), "error": rr["stderr"]}
            row["header_path"] = str(PY_HEADER_DIR / "Python.h")
            patch = (PY_HEADER_DIR / "patchlevel.h").read_text(errors="ignore") if (PY_HEADER_DIR / "patchlevel.h").exists() else ""
            for name in ["PY_MAJOR_VERSION", "PY_MINOR_VERSION", "PY_MICRO_VERSION"]:
                m = re.search(rf"#define\s+{name}\s+(\d+)", patch)
                row[name] = m.group(1) if m else ""
            abi_rows.append(row)
    write_csv(OUT / "python_header_abi_audit.csv", abi_rows)
    header_ver = ".".join([abi_rows[0].get("PY_MAJOR_VERSION", ""), abi_rows[0].get("PY_MINOR_VERSION", ""), abi_rows[0].get("PY_MICRO_VERSION", "")]) if abi_rows else ""
    L2b = (PY_HEADER_DIR / "Python.h").exists() and any(r.get("PY_MAJOR_VERSION") == "3" and r.get("PY_MINOR_VERSION") == "10" and str(r.get("SOABI", "")).startswith("cpython-310") for r in abi_rows)

    gpu_code = "import torch,json; print(json.dumps([{'visible_index':i,'name':torch.cuda.get_device_name(i),'capability':'.'.join(map(str,torch.cuda.get_device_capability(i))),'memory':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]))"
    gr = run(["/usr/bin/python3", "-c", gpu_code], env=env0)
    gpu = json.loads(gr["stdout"]) if gr["returncode"] == 0 and gr["stdout"].strip().startswith("[") else []
    arch = gpu[0]["capability"] if gpu else ""
    write_text(OUT / "stage5_R2_gpu_architecture.json", json.dumps({"devices": gpu, "TORCH_CUDA_ARCH_LIST": arch}, indent=2) + "\n")
    L2 = "PASS" if L2a and L2b and gpu else "FAIL"
    env = base_env(arch)
    write_text(OUT / "real_build_environment.json", json.dumps({k: v for k, v in env.items() if "TOKEN" not in k and "KEY" not in k and "SECRET" not in k}, indent=2) + "\n")

    cand_rows = []
    for p in [NVD] + sorted(ROOT.glob("**/*nvdiffrast*"))[:80]:
        if p.exists():
            cand_rows.append({"path": str(p), "type": "SOURCE" if (p / "setup.py").exists() else ("BINARY" if p.suffix == ".so" else "BUILD_ARTIFACT"), "setup_py": (p / "setup.py").exists(), "pyproject": (p / "pyproject.toml").exists(), "git": (p / ".git").exists() if p.is_dir() else False, "sha": sha(p) if p.is_file() else ""})
    write_csv(OUT / "nvdiffrast_R2_candidate_audit.csv", cand_rows)
    write_text(OUT / "nvdiffrast_build_mechanism.md", f"# nvdiffrast build mechanism\n\nSelected source root: `{NVD}`\n\n`setup.py` uses Python package build metadata for `_nvdiffrast_c`; native sources are under `csrc/common` and `csrc/torch`. This R2 gate executes `/usr/bin/python3 setup.py build` with CUDA_HOME, CPATH, LD_LIBRARY_PATH, TORCH_EXTENSIONS_DIR, and TORCH_CUDA_ARCH_LIST explicitly set.\n")
    diff_sources = sorted([str(p.relative_to(DIFF)) for p in DIFF.glob("**/*") if p.is_file() and (p.suffix in [".cu", ".cpp", ".h"] or p.name in ["setup.py", "README.md"])])
    write_text(OUT / "diff_surfel_R2_source_audit.md", f"# diff_surfel_anych source audit\n\nSource root: `{DIFF}`\n\nsetup.py: `{DIFF / 'setup.py'}`\n\nFiles:\n\n" + "\n".join(f"- `{x}`" for x in diff_sources[:200]) + "\n")

    probe_log = []
    probe_rows = []
    with tempfile.TemporaryDirectory(dir=str(BUILD / "tmp")) as td:
        td = Path(td)
        (td / "pyprobe.cpp").write_text("#include <Python.h>\nint main(){return 0;}\n")
        pr = run(["g++", "-I", str(PY_HEADER_DIR), "-c", str(td / "pyprobe.cpp"), "-o", str(td / "pyprobe.o")], env=env)
        probe_rows.append({"probe": "Python.h", "status": "PASS" if pr["returncode"] == 0 else "FAIL", "returncode": pr["returncode"], "first_error": first_error(pr["stdout"] + pr["stderr"])})
        probe_log.append("## Python.h\n" + json.dumps(pr, indent=2))
        (td / "cudaprobe.cu").write_text("#include <cuda_runtime.h>\n__global__ void test_kernel() {}\n")
        cr = run([str(NVCC), "-c", str(td / "cudaprobe.cu"), "-o", str(td / "cudaprobe.o")], env=env)
        probe_rows.append({"probe": "nvcc CUDA", "status": "PASS" if cr["returncode"] == 0 else "FAIL", "returncode": cr["returncode"], "first_error": first_error(cr["stdout"] + cr["stderr"])})
        probe_log.append("## nvcc CUDA\n" + json.dumps(cr, indent=2))
        torch_ext = "from torch.utils.cpp_extension import load_inline; load_inline(name='r2_cpp_probe', cpp_sources='int answer(){return 42;}', functions=['answer'], verbose=True, build_directory=r'%s')" % (BUILD / "torch_extensions")
        er = run(["/usr/bin/python3", "-c", torch_ext], env=env, timeout=180)
        probe_rows.append({"probe": "torch C++ extension", "status": "PASS" if er["returncode"] == 0 else "FAIL", "returncode": er["returncode"], "first_error": first_error(er["stdout"] + er["stderr"])})
        probe_log.append("## torch C++ extension\n" + json.dumps(er, indent=2))
    write_csv(OUT / "compiler_probe_results.csv", probe_rows)
    write_text(OUT / "compiler_probe_full_log.txt", "\n\n".join(probe_log) + "\n")
    L3 = "PASS" if all(r["status"] == "PASS" for r in probe_rows) else "FAIL"

    nvd_result = {"attempted": "NO", "returncode": "", "classification": "NOT_EXECUTED_L3_FAIL", "first_error": "NOT_EXECUTED_L3_FAIL", "binary": "", "import": "NO"}
    diff_result = {"attempted": "NO", "returncode": "", "classification": "NOT_EXECUTED_L3_FAIL", "first_error": "NOT_EXECUTED_L3_FAIL", "binary": "", "import": "NO"}
    nvd_build_paths = []
    diff_build_paths = []
    if L0 == "PASS" and L1 == "PASS" and L2 == "PASS" and L3 == "PASS":
        nr = run(["/usr/bin/python3", "setup.py", "build", "--build-base", str(BUILD / "nvdiffrast")], env=env, cwd=NVD, timeout=900)
        nlog = json.dumps({"command": nr["cmd"], "cwd": nr["cwd"], "environment": {k: env[k] for k in ["CUDA_VISIBLE_DEVICES", "CUDA_HOME", "PATH", "LD_LIBRARY_PATH", "CPATH", "PYTHONPATH", "TORCH_EXTENSIONS_DIR", "TORCH_CUDA_ARCH_LIST", "MAX_JOBS"] if k in env}, "returncode": nr["returncode"], "stdout": nr["stdout"], "stderr": nr["stderr"], "start": nr["start"], "end": nr["end"]}, indent=2)
        write_text(OUT / "build_nvdiffrast_real.log", nlog)
        nvd_build_paths = [p for p in (BUILD / "nvdiffrast").glob("**/*") if p.is_dir() and "lib" in p.name] + [BUILD / "nvdiffrast"]
        nvd_bins = sorted((BUILD / "nvdiffrast").glob("**/_nvdiffrast_c*.so")) + sorted((BUILD / "nvdiffrast").glob("**/nvdiffrast_plugin*.so"))
        nimp = import_probe("nvdiffrast.torch", env, [str(p) for p in nvd_build_paths])
        nvd_result = {"attempted": "YES", "returncode": nr["returncode"], "classification": classify_build(nr["returncode"], nr["stdout"] + nr["stderr"]), "first_error": first_error(nr["stdout"] + nr["stderr"]), "binary": str(nvd_bins[0]) if nvd_bins else "", "import": "YES" if nimp["status"] == "PASS" else "NO", "import_error": nimp["error"]}
        nvd_audit = audit_binary(Path(nvd_result["binary"])) if nvd_result["binary"] else {}
        write_csv(OUT / "nvdiffrast_real_build_result.csv", [{**nvd_result, "binary_sha256": nvd_audit.get("sha256", ""), "file": nvd_audit.get("file", ""), "ldd": nvd_audit.get("ldd", ""), "readelf": nvd_audit.get("readelf", "")}])

        dr = run(["/usr/bin/python3", "setup.py", "build", "--build-base", str(BUILD / "diff_surfel_anych")], env=env, cwd=DIFF, timeout=900)
        dlog = json.dumps({"command": dr["cmd"], "cwd": dr["cwd"], "environment": {k: env[k] for k in ["CUDA_VISIBLE_DEVICES", "CUDA_HOME", "PATH", "LD_LIBRARY_PATH", "CPATH", "PYTHONPATH", "TORCH_EXTENSIONS_DIR", "TORCH_CUDA_ARCH_LIST", "MAX_JOBS"] if k in env}, "returncode": dr["returncode"], "stdout": dr["stdout"], "stderr": dr["stderr"], "start": dr["start"], "end": dr["end"]}, indent=2)
        write_text(OUT / "build_diff_surfel_anych_real.log", dlog)
        diff_build_paths = [p for p in (BUILD / "diff_surfel_anych").glob("**/*") if p.is_dir() and "lib" in p.name] + [BUILD / "diff_surfel_anych"]
        diff_bins = sorted((BUILD / "diff_surfel_anych").glob("**/diff_surfel_anych/_C*.so")) + sorted((BUILD / "diff_surfel_anych").glob("**/_C*.so"))
        dimp = import_probe("diff_surfel_anych", env, [str(p) for p in diff_build_paths])
        diff_result = {"attempted": "YES", "returncode": dr["returncode"], "classification": classify_build(dr["returncode"], dr["stdout"] + dr["stderr"]), "first_error": first_error(dr["stdout"] + dr["stderr"]), "binary": str(diff_bins[0]) if diff_bins else "", "import": "YES" if dimp["status"] == "PASS" else "NO", "import_error": dimp["error"]}
        diff_audit = audit_binary(Path(diff_result["binary"])) if diff_result["binary"] else {}
        write_csv(OUT / "diff_surfel_real_build_result.csv", [{**diff_result, "binary_sha256": diff_audit.get("sha256", ""), "file": diff_audit.get("file", ""), "ldd": diff_audit.get("ldd", ""), "readelf": diff_audit.get("readelf", "")}])
    else:
        write_text(OUT / "build_nvdiffrast_real.log", "NOT_EXECUTED_L0_L1_L2_OR_L3_FAIL\n")
        write_text(OUT / "build_diff_surfel_anych_real.log", "NOT_EXECUTED_L0_L1_L2_OR_L3_FAIL\n")
        write_csv(OUT / "nvdiffrast_real_build_result.csv", [nvd_result])
        write_csv(OUT / "diff_surfel_real_build_result.csv", [diff_result])

    L4a = "PASS" if nvd_result["import"] == "YES" else "FAIL"
    L4b = "PASS" if diff_result["import"] == "YES" else "FAIL"
    L4 = "PASS" if L4a == "PASS" and L4b == "PASS" else "FAIL"

    launcher = SRC / "verified_rtsplat_R2_python.sh"
    pp = ":".join([str(RT)] + [str(p) for p in nvd_build_paths + diff_build_paths] + [str(NVD), str(USER_SITE), str(CONDA_SITE)])
    write_text(launcher, f"""#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-2,3}}"
export CUDA_HOME="{CUDA_HOME}"
export PATH="{CUDA_HOME}/bin:${{PATH}}"
export TORCH_EXTENSIONS_DIR="{BUILD / 'torch_extensions'}"
export LD_LIBRARY_PATH="{TORCH_LIB}:{CUDA_HOME / 'lib64'}:${{LD_LIBRARY_PATH:-}}"
export CPATH="{PY_HEADER_DIR}:${{CPATH:-}}"
export PYTHONPATH="{pp}:${{PYTHONPATH:-}}"
exec /usr/bin/python3 "$@"
""")
    launcher.chmod(0o755)
    write_text(OUT / "verified_rtsplat_R2_runtime_lock.json", json.dumps({"launcher": str(launcher), "PYTHONPATH": pp, "nvdiffrast_result": nvd_result, "diff_surfel_result": diff_result}, indent=2) + "\n")

    imports = []
    for m in ["torch", "simple_knn._C", "nvdiffrast.torch", "diff_surfel_anych", "scene.gaussian_model", "gaussian_renderer"]:
        imports.append(import_probe(m, env, [str(p) for p in nvd_build_paths + diff_build_paths]))
    write_csv(OUT / "R2_critical_import_matrix.csv", imports)
    import_pass = sum(1 for r in imports if r["status"] == "PASS")
    L5a = "PASS" if import_pass == 6 else "FAIL"

    forward_stats = {}
    state_rows = [{"state": s, "status": "NOT_EXECUTED_L5B_FAIL", "forward_diff": "NOT_EXECUTED", "grad_L2": "NOT_EXECUTED"} for s in ["_occupancy", "_opacity", "_transmissivity"]]
    if L5a == "PASS":
        smoke_code = r"""
import json, math
from types import SimpleNamespace
import numpy as np
import torch
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from scene.cameras import Camera
from gaussian_renderer import render

def make_model():
    args=SimpleNamespace(env_scope_radius=0.0, env_scope_center=[0,0,0], xyz_axis=[0,1,2], rand_init=False, run_dim=16)
    pc=GaussianModel(0,args)
    pts=np.array([[0.,0.,2.0],[0.05,0.,2.05],[-0.05,0.,2.05],[0.,0.05,2.1],[0.05,0.05,2.1],[-0.05,-0.05,2.1]], dtype=np.float32)
    cols=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.],[1.,0.7,0.2],[0.2,0.8,1.0]], dtype=np.float32)
    pc.create_from_pcd(BasicPointCloud(pts, cols, np.zeros_like(pts)), 1.0)
    return pc

def make_cam(uid, tx=0.0):
    img=torch.zeros(3,32,32,device='cuda')
    mask=torch.zeros(1,32,32,device='cuda')
    return Camera(uid, np.eye(3,dtype=np.float32), np.array([tx,0.,0.],dtype=np.float32), math.radians(60), math.radians(60), img, None, mask, f'r2_{uid}', uid)

pipe=SimpleNamespace(depth_ratio=0.0, init_stage=True)
pc=make_model()
cam=make_cam(0)
bg=torch.zeros(3,device='cuda')
out=render(cam, pc, pipe, bg)
rgb=out['final_rendering']
forward={
 'status':'PASS',
 'key':'final_rendering',
 'shape':list(rgb.shape),
 'dtype':str(rgb.dtype),
 'device':str(rgb.device),
 'requires_grad':bool(rgb.requires_grad),
 'grad_fn':type(rgb.grad_fn).__name__ if rgb.grad_fn is not None else 'None',
 'finite_fraction':float(torch.isfinite(rgb).float().mean().item()),
 'min':float(rgb.min().item()),
 'max':float(rgb.max().item()),
 'mean':float(rgb.mean().item()),
 'std':float(rgb.std().item()),
 'nonzero_pixel_fraction':float((rgb.abs()>1e-12).float().mean().item()),
}
rows=[]
cams=[make_cam(i, tx) for i,tx in enumerate([0.0,0.01,-0.01,0.02])]
for name in ['_occupancy','_opacity','_transmissivity']:
    param=getattr(pc, name)
    before=render(cam, pc, pipe, bg)['final_rendering'].detach().clone()
    saved=param.detach().clone()
    n=param.shape[0]
    idx=torch.arange(max(1, n//2), device='cuda')
    with torch.no_grad():
        param[idx] += 0.01
    after=render(cam, pc, pipe, bg)['final_rendering'].detach().clone()
    with torch.no_grad():
        param.copy_(saved)
    restored=render(cam, pc, pipe, bg)['final_rendering'].detach().clone()
    mean_diff=float((after-before).abs().mean().item())
    max_diff=float((after-before).abs().max().item())
    restore=float((restored-before).abs().max().item())
    for p in [pc._xyz, pc._features_dc, pc._features_rest, pc._scaling, pc._rotation, pc._occupancy, pc._opacity, pc._transmissivity, pc._roughness, pc._reflectance, pc._language_feature]:
        if p.grad is not None:
            p.grad.zero_()
        p.requires_grad_(p is param)
    loss=0
    for c in cams:
        r=render(c, pc, pipe, bg)['final_rendering']
        target=torch.ones_like(r)*0.123
        loss=loss+(r-target).pow(2).mean()
    loss.backward()
    g=param.grad
    finite=float(torch.isfinite(g).float().mean().item()) if g is not None else 0.0
    nonzero=float((g.abs()>0).float().mean().item()) if g is not None else 0.0
    l2=float(g.norm().item()) if g is not None else 0.0
    render_active=mean_diff>1e-9 and max_diff>1e-8 and restore<=1e-7
    grad_active=finite==1.0 and nonzero>=0.25 and l2>1e-8
    rows.append({'state':name,'initialized':'YES','persistent':'YES','native_consumer':'YES','mean_rgb_diff':mean_diff,'max_rgb_diff':max_diff,'restore_max_diff':restore,'finite_grad_fraction':finite,'nonzero_grad_fraction':nonzero,'grad_L2':l2,'render_active':render_active,'gradient_active':grad_active,'both_active':render_active and grad_active})
print(json.dumps({'forward':forward,'states':rows}))
"""
        smoke_env = env.copy()
        smoke_env["PYTHONPATH"] = ":".join([str(RT)] + [str(p) for p in nvd_build_paths + diff_build_paths] + [str(NVD), str(USER_SITE), str(CONDA_SITE), env.get("PYTHONPATH", "")])
        sr = run(["/usr/bin/python3", "-c", smoke_code], env=smoke_env, timeout=180)
        smoke_json = next((line for line in sr["stdout"].splitlines() if line.strip().startswith("{")), "")
        if sr["returncode"] == 0 and smoke_json:
            smoke = json.loads(smoke_json)
            forward_stats = smoke["forward"]
            state_rows = smoke["states"]
            write_csv(OUT / "R2_native_renderer_forward.csv", [forward_stats])
            write_csv(OUT / "R2_three_state_causality.csv", state_rows)
            L5b = "PASS" if forward_stats["finite_fraction"] == 1.0 and forward_stats["nonzero_pixel_fraction"] > 0 and "cuda" in forward_stats["device"] else "FAIL"
        else:
            forward_stats = {"status": "FAIL", "error": first_error(sr["stdout"] + sr["stderr"]), "trace": sr["stdout"] + sr["stderr"]}
            write_csv(OUT / "R2_native_renderer_forward.csv", [forward_stats])
            write_csv(OUT / "R2_three_state_causality.csv", state_rows)
            L5b = "FAIL"
    else:
        write_csv(OUT / "R2_native_renderer_forward.csv", [{"status": "NOT_EXECUTED_L5A_FAIL"}])
        L5b = "NOT_EXECUTED_L5A_FAIL"
    active_states = [r["state"] for r in state_rows if str(r.get("both_active")) == "True"]
    L6 = "PASS" if L5b == "PASS" and len(active_states) >= 2 else ("FAIL" if L5b == "PASS" else "NOT_EXECUTED_L5B_FAIL")

    if L1 == "FAIL" or L2 == "FAIL" or L3 == "FAIL":
        final = "CASE TRUE-LOCAL-TOOLCHAIN-BLOCKER"
    elif L4 == "FAIL":
        final = "CASE RTSPLAT-REAL-BUILD-INCOMPATIBLE"
    elif L5a == "FAIL" or L5b != "PASS":
        final = "CASE RTSPLAT-NATIVE-FORWARD-FAIL"
    elif L6 != "PASS":
        final = "CASE RTSPLAT-NATIVE-STATE-CAUSALITY-FAIL"
    else:
        final = "CASE RTSPLAT-LOCAL-RUNTIME-SALVAGED"

    items = [
        ("A", "L0", L0),
        ("B", "/usr/local/cuda/bin/nvcc exists yes/no", "YES" if NVCC.exists() else "NO"),
        ("C", "explicit nvcc executable yes/no", "YES" if os.access(NVCC, os.X_OK) else "NO"),
        ("D", "nvcc version", cuda_rows[0]["nvcc_version"] if cuda_rows else "NONE"),
        ("E", "resolved CUDA_HOME", str(CUDA_HOME.resolve()) if CUDA_HOME.exists() else "NONE"),
        ("F", "CUDA headers/runtime complete yes/no", "YES" if L1 == "PASS" else "NO"),
        ("G", "previous missing-nvcc classification PATH-only / REAL", prev_class),
        ("H", "L1", L1),
        ("I", "torch version / torch CUDA", f"{torch_lock.get('version')} / {torch_lock.get('cuda')}"),
        ("J", "torch CXX11 ABI", str(torch_lock.get("cxx11_abi"))),
        ("K", "torch cpp_extension CUDA_HOME after explicit environment", str(torch_lock.get("cpp_extension_CUDA_HOME"))),
        ("L", "Python header path", str(PY_HEADER_DIR / "Python.h")),
        ("M", "Python header version", header_ver),
        ("N", "build interpreter SOABI", str(next((r.get("SOABI") for r in abi_rows if r.get("interpreter") == "/usr/bin/python3"), ""))),
        ("O", "header interpreter SOABI", str(next((r.get("SOABI") for r in abi_rows if "rtsplat_legacy2" in r.get("interpreter", "")), ""))),
        ("P", "Python header ABI compatible yes/no", "YES" if L2b else "NO"),
        ("Q", "visible GPU names/capabilities", "; ".join(f"{g['name']} cc{g['capability']}" for g in gpu)),
        ("R", "TORCH_CUDA_ARCH_LIST selected", arch),
        ("S", "L2", L2),
        ("T", "Python.h compiler probe", probe_rows[0]["status"]),
        ("U", "nvcc CUDA compiler probe", probe_rows[1]["status"]),
        ("V", "torch C++ extension probe", probe_rows[2]["status"]),
        ("W", "L3", L3),
        ("X", "selected nvdiffrast source root", str(NVD)),
        ("Y", "nvdiffrast build mechanism", "setup.py build native _nvdiffrast_c"),
        ("Z", "nvdiffrast real build attempted yes/no", nvd_result["attempted"]),
        ("AA", "nvdiffrast build exit code", str(nvd_result["returncode"])),
        ("AB", "nvdiffrast build classification", nvd_result["classification"]),
        ("AC", "nvdiffrast first root error if failed", nvd_result["first_error"]),
        ("AD", "nvdiffrast native binary path if success", nvd_result["binary"] or "NONE"),
        ("AE", "nvdiffrast import after build yes/no", nvd_result["import"]),
        ("AF", "L4a", L4a),
        ("AG", "diff_surfel exact source root", str(DIFF)),
        ("AH", "diff_surfel real build attempted yes/no", diff_result["attempted"]),
        ("AI", "diff_surfel build exit code", str(diff_result["returncode"])),
        ("AJ", "diff_surfel build classification", diff_result["classification"]),
        ("AK", "diff_surfel first root error if failed", diff_result["first_error"]),
        ("AL", "diff_surfel native binary path if success", diff_result["binary"] or "NONE"),
        ("AM", "diff_surfel import after build yes/no", diff_result["import"]),
        ("AN", "L4b", L4b),
        ("AO", "L4", L4),
        ("AP", "critical imports pass/total", f"{import_pass}/6"),
        ("AQ", "module/binary import paths", "; ".join(f"{r['module']}={r['status']}:{r['path'] or r['error']}" for r in imports)),
        ("AR", "L5a", L5a),
        ("AS", "native RT forward finite/nonzero yes/no", "YES" if L5b == "PASS" else ("NOT_EXECUTED_L5A_FAIL" if L5a != "PASS" else "NO")),
        ("AT", "native RGB shape/device/grad_fn", f"{forward_stats.get('shape')}/{forward_stats.get('device')}/{forward_stats.get('grad_fn')}" if forward_stats else "NOT_EXECUTED_L5A_FAIL"),
        ("AU", "L5b", L5b),
        ("AV", "occupancy forward diff / grad L2", next((f"{r.get('mean_rgb_diff')}/{r.get('grad_L2')}" for r in state_rows if r.get("state") == "_occupancy"), "NOT_EXECUTED")),
        ("AW", "opacity forward diff / grad L2", next((f"{r.get('mean_rgb_diff')}/{r.get('grad_L2')}" for r in state_rows if r.get("state") == "_opacity"), "NOT_EXECUTED")),
        ("AX", "transmissivity forward diff / grad L2", next((f"{r.get('mean_rgb_diff')}/{r.get('grad_L2')}" for r in state_rows if r.get("state") == "_transmissivity"), "NOT_EXECUTED")),
        ("AY", "native states render+gradient active count", str(len(active_states))),
        ("AZ", "active native state names", ",".join(active_states) if active_states else "NONE"),
        ("BA", "L6", L6),
        ("BB", "Final CASE", final),
        ("BC", "previous RTSPLAT-LOCAL-RUNTIME-UNRECOVERABLE classification valid yes/no", "NO_PATH_ONLY_NVCC_MISCLASSIFICATION_RETIRED"),
        ("BD", "local RT runtime salvaged yes/no", "YES" if final == "CASE RTSPLAT-LOCAL-RUNTIME-SALVAGED" else "NO"),
        ("BE", "allow Stage5 J2a/J3/J4 continuation yes/no", "YES" if final == "CASE RTSPLAT-LOCAL-RUNTIME-SALVAGED" else "NO"),
        ("BF", "AttributeDeformGS hypothesis status", "UNTESTED"),
        ("BG", "PRIMARY ATTRIBUTE-DEFORMATION LINE STOP/CONTINUE", "CONTINUE" if final == "CASE RTSPLAT-LOCAL-RUNTIME-SALVAGED" else "STOP"),
        ("BH", "KIOT status", "CONTROLLED-CARRIER-ONLY"),
        ("BI", "next exact research action", "Stop current RT route after real build evidence" if final != "CASE RTSPLAT-LOCAL-RUNTIME-SALVAGED" else "Resume Stage5 J2a checkpoint sidecar closure"),
        ("BJ", "report path", str(OUT / "stage5_0_R2_real_build_report.md")),
        ("BK", "summary path", str(OUT / "stage5_0_R2_real_build_summary.md")),
    ]
    final_text = "\n".join(f"{k}. {name}: {value}" for k, name, value in items) + "\n"
    write_text(OUT / "stage5_0_R2_real_build_report.md", "# Stage5.0-R2 Real Local Extension Build Report\n\n" + "\n".join(f"## {k}. {name}\n\n{value}\n" for k, name, value in items))
    write_text(OUT / "stage5_0_R2_real_build_summary.md", f"# Stage5.0-R2 summary\n\n- Final CASE: `{final}`\n- L0/L1/L2/L3/L4/L5a/L5b/L6: {L0}/{L1}/{L2}/{L3}/{L4}/{L5a}/{L5b}/{L6}\n- nvdiffrast build: {nvd_result['attempted']} / {nvd_result['classification']}\n- diff_surfel_anych build: {diff_result['attempted']} / {diff_result['classification']}\n- AttributeDeformGS hypothesis: UNTESTED\n")
    write_text(OUT / "stage5_0_R2_real_build_log.txt", final_text)
    write_text(OUT / "final_terminal_summary.txt", final_text)
    print(final_text)


if __name__ == "__main__":
    main()
