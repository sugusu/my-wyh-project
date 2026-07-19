#!/usr/bin/env python3
import hashlib, json, platform, subprocess, sys
from pathlib import Path
import torch, yaml

ROOT=Path(__file__).resolve().parents[1]
cfgp=ROOT/'configs/scene01_dev.yaml'; cfg=yaml.safe_load(cfgp.read_text())
def cmd(x, cwd=None):
    try: return subprocess.check_output(x, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e: return f"unavailable: {e}"
tsgs=Path(cfg['paths']['tsgs_root']).resolve()
data=Path(cfg['paths']['data_root']).resolve()
env={"python":sys.version,"platform":platform.platform(),"pytorch":torch.__version__,
     "torch_cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),
     "gpu_inventory":cmd(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv,noheader']),
     "allowed_gpus":[2,3],"tsgs_git_commit":cmd(['git','rev-parse','HEAD'],tsgs),
     "tsgs_path":str(tsgs),"data_path":str(data),"scene_path":str(Path(cfg['paths']['scene_path']).resolve()),
     "config_path":str(cfgp),"config_sha256":hashlib.sha256(cfgp.read_bytes()).hexdigest(),
     "normal_folder":cfg['baseline']['normal_folder']}
(ROOT/'reports/environment.json').write_text(json.dumps(env,indent=2)+'\n')
print(json.dumps(env,indent=2))

