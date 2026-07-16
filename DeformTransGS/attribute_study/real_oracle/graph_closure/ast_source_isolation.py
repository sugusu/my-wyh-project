from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN = {
    "synthetic_release_error",
    "preset_metric",
    "target_metric",
    "expected_metric",
    "hardcoded_psnr",
    "hardcoded_elog",
}


def executable_forbidden_references(root: Path) -> list[dict]:
    rows = []
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            rows.append({"path": str(path), "line": exc.lineno or 0, "kind": "SyntaxError", "name": str(exc)})
            continue
        for node in ast.walk(tree):
            name = None
            kind = type(node).__name__
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[-1] in FORBIDDEN:
                        rows.append({"path": str(path), "line": node.lineno, "kind": kind, "name": alias.name})
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in FORBIDDEN:
                        rows.append({"path": str(path), "line": node.lineno, "kind": kind, "name": alias.name})
            if name in FORBIDDEN:
                rows.append({"path": str(path), "line": getattr(node, "lineno", 0), "kind": kind, "name": name})
    return rows
