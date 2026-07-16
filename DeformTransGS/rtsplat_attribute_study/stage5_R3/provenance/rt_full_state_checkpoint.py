from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


PERSISTENT_TENSORS = [
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_occupancy",
    "_opacity",
    "_transmissivity",
    "_roughness",
    "_reflectance",
    "_language_feature",
]


def save_full_state(path: str | Path, model: Any, metadata: dict | None = None, optimizer: Any | None = None) -> None:
    payload = {
        "metadata": metadata or {},
        "active_sh_degree": int(getattr(model, "active_sh_degree", 0)),
        "max_sh_degree": int(getattr(model, "max_sh_degree", 0)),
        "persistent_tensors": {name: getattr(model, name).detach().clone() for name in PERSISTENT_TENSORS if hasattr(model, name)},
        "auxiliary_modules": {
            "dir_encoding": model.dir_encoding.state_dict() if hasattr(model, "dir_encoding") else {},
            "light_mlp": model.light_mlp.state_dict() if hasattr(model, "light_mlp") else {},
        },
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_full_state(path: str | Path, model: Any, optimizer: Any | None = None) -> dict:
    payload = torch.load(path, map_location="cuda")
    for name, tensor in payload["persistent_tensors"].items():
        current = getattr(model, name)
        if isinstance(current, torch.nn.Parameter):
            current.data = tensor.detach().clone().to(current.device)
        else:
            setattr(model, name, torch.nn.Parameter(tensor.detach().clone().cuda().requires_grad_(True)))
    model.active_sh_degree = payload.get("active_sh_degree", model.active_sh_degree)
    if hasattr(model, "dir_encoding") and payload["auxiliary_modules"].get("dir_encoding"):
        model.dir_encoding.load_state_dict(payload["auxiliary_modules"]["dir_encoding"])
    if hasattr(model, "light_mlp") and payload["auxiliary_modules"].get("light_mlp"):
        model.light_mlp.load_state_dict(payload["auxiliary_modules"]["light_mlp"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
