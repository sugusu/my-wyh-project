import ast
import hashlib
import json
import os
import re
from pathlib import Path


PROJECT_ROOT = Path("/data/wyh/RecycleGS")
OUT_DEBUG = PROJECT_ROOT / "outputs/debug/stage0r"
OUT_STAGE0RC = PROJECT_ROOT / "outputs/debug/stage0rc"
OUT_BASELINE = PROJECT_ROOT / "outputs/baseline"
TSGS_ROOT = Path("/data/wyh/repos/TSGS")


def ensure_dirs():
    OUT_DEBUG.mkdir(parents=True, exist_ok=True)
    OUT_STAGE0RC.mkdir(parents=True, exist_ok=True)
    OUT_BASELINE.mkdir(parents=True, exist_ok=True)


def read_text(path):
    return Path(path).read_text(errors="replace")


def sha256_file(path):
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_cfg_args(path):
    if not Path(path).exists():
        return {}
    raw = Path(path).read_text().strip()
    if not raw:
        raise ValueError(f"empty cfg_args: {path}")
    tree = ast.parse(raw, mode="eval")
    if not isinstance(tree.body, ast.Call):
        raise ValueError(f"cfg_args is not a Namespace call: {path}")
    values = {}
    for kw in tree.body.keywords:
        values[kw.arg] = ast.literal_eval(kw.value)
    return values


def parse_command_line_from_logs(*paths):
    commands = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            if "Command:" in line:
                commands.append(line.split("Command:", 1)[1].strip())
    return commands[-1] if commands else None


def command_has_flag(command, flag):
    if not command:
        return False
    import shlex
    return flag in shlex.split(command)


def command_value(command, flag):
    if not command:
        return None
    import shlex
    parts = shlex.split(command)
    if flag not in parts:
        return None
    idx = parts.index(flag)
    if idx + 1 >= len(parts):
        return True
    return parts[idx + 1]


def parse_saved_defaults(arguments_path):
    """Return simple literal defaults from TSGS arguments/__init__.py."""
    if not Path(arguments_path).exists():
        return {}
    text = Path(arguments_path).read_text(errors="replace")
    defaults = {}
    import ast
    tree = ast.parse(text)
    class_names = {"ModelParams", "OptimizationParams", "PipelineParams"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            for item in ast.walk(node):
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                            name = target.attr.lstrip("_")
                            try:
                                defaults[name] = ast.literal_eval(item.value)
                            except Exception:
                                pass
    return defaults


def parse_typed_value(raw, default=None):
    if raw is True or raw is False or raw is None:
        return raw
    if isinstance(default, bool):
        return str(raw).lower() in {"1", "true", "yes"}
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_md(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n")


def ply_vertex_count(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                return None
    return None


def find_checkpoint(model_dir, iteration):
    model_dir = Path(model_dir)
    candidates = [
        model_dir / f"chkpnt{iteration}.pth",
        model_dir / f"checkpoint_{iteration}.pth",
        model_dir / f"checkpoint{iteration}.pth",
        model_dir / f"chkpnt_{iteration}.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(model_dir.glob(f"**/*{iteration}*.pth"))
    for path in matches:
        name = path.name.lower()
        if "chkpnt" in name or "checkpoint" in name:
            return path
    return None


def extract_last_iteration_and_loss(log_text):
    last_iter = None
    last_loss = None
    for match in re.finditer(r"Iteration\s+(\d+):\s+Loss=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", log_text, re.I):
        last_iter = int(match.group(1))
        last_loss = float(match.group(2))
    for match in re.finditer(r"iteration:\s*(\d+)", log_text, re.I):
        last_iter = max(last_iter or 0, int(match.group(1)))
    return last_iter, last_loss


def log_error_counts(log_text):
    lower = log_text.lower()
    return {
        "traceback_count": lower.count("traceback"),
        "cuda_oom_count": len(re.findall(r"cuda out of memory|torch\.outofmemoryerror", lower)),
        "error_count": len(re.findall(r"traceback|error|exception|cuda out of memory|killed", lower)),
        "nan_count": len(re.findall(r"\bnan\b", lower)),
        "inf_count": len(re.findall(r"\binf\b", lower)),
    }
