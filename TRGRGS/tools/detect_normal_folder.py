#!/usr/bin/env python3
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("scene", type=Path)
a = p.parse_args()
images = {x.stem for x in (a.scene / "images").glob("*.png")}
for candidate in ("transnormals", "normals"):
    folder = a.scene / candidate
    normals = {x.name.removesuffix("_normal.png") for x in folder.glob("*_normal.png")}
    if images and images <= normals:
        print(candidate)
        raise SystemExit(0)
raise SystemExit("Neither transnormals nor normals completely covers scene images")

