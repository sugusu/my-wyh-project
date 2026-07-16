#!/usr/bin/env python3
"""Stage 3.2.6: Canonical-equivalent representation divergence gate"""
import sys, os, csv, math, hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import spearmanr
from pathlib import Path

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage3_2_6_canonical_equivalent_divergence"
os.makedirs(f"{OUTPUT}/reference_alpha", exist_ok=True)

sys.path.insert(0, "/data/wyh/repos/TSGS")
sys.path.insert(0, "/data/wyh/repos/TSGS/pytorch3d_stub")
sys.path.insert(0, f"{BASE}/benchmark")

import torch, trimesh
from torch.nn import functional as F
from scene.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import focal2fov
from deformations.twist import deform_points as twist_def

device="cuda"
log_lines=[]
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)

bg_color=torch.zeros(3,device=device)
pipe=type('obj',(object,),{"debug":False,"convert_SHs_python":False,"compute_cov3D_python":False})()

# ═══════════════════════════════════════════════════════════
# 0. 12-camera rig
# ═══════════════════════════════════════════════════════════
# Use 12 cameras from Stage 2
import json
cam_data = json.load(open(f"{BASE}/experiments/stage1_minimal_gt/cameras.json"))
camera_cfgs = [{"pos":c["origin"],"target":c["target"],"up":c["up"],"id":c["id"]} for c in cam_data]
fit_ids = {0,4,8}
holdout_ids = {i for i in range(12)} - fit_ids

def build_cam(cfg):
    pa=np.array(cfg["pos"],dtype=np.float32); ta=np.array(cfg["target"],dtype=np.float32); ua=np.array(cfg["up"],dtype=np.float32)
    fwd=ta-pa; fwd/=np.linalg.norm(fwd)
    rt=np.cross(ua,fwd); rt/=np.linalg.norm(rt); nu=np.cross(fwd,rt)
    Rw=np.eye(3,dtype=np.float32); Rw[0,:]=rt; Rw[1,:]=nu; Rw[2,:]=fwd
    T=-Rw@pa; R=Rw.T
    fx=256/(2*math.tan(math.radians(45/2)))
    cam=Camera(colmap_id=cfg["id"],R=R,T=T,FoVx=focal2fov(fx,256),FoVy=focal2fov(fx,256),image_width=256,image_height=256,image_path="",image_PIL=None,image_name=f"cam_{cfg['id']:03d}",uid=cfg["id"],preload_img=False,data_device="cpu")
    cam.original_image=torch.zeros(3,256,256); return cam

film_cams=[build_cam(c) for c in camera_cfgs]
cam_ids_list=[c["id"] for c in camera_cfgs]

# ═══════════════════════════════════════════════════════════
# 1. Sheet mesh and reference representation
# ═══════════════════════════════════════════════════════════
hdr("1. Building representations")
W,H=1.5,1.5

def make_sheet(divs):
    verts=[[-W/2+W*i/divs,-H/2+H*j/divs,0.0] for i in range(divs+1) for j in range(divs+1)]
    faces=[]
    for i in range(divs):
        for j in range(divs):
            idx=i*(divs+1)+j
            faces.extend([[idx,idx+1,idx+divs+2],[idx,idx+divs+2,idx+divs+1]])
    return torch.tensor(verts,dtype=torch.float32,device=device), faces

# REF_41: 41x41 with frozen tau from Stage 3.2.5
ckpt_path=f"{BASE}/experiments/stage3_2_5_representation_drift_confirmation/canonical_checkpoint.pt"
ckpt=torch.load(ckpt_path, map_location=device, weights_only=True)
ref_tau_raw=ckpt["tau_raw"]  # (1681, 1)
ref_color_raw=ckpt["color_raw"]

ref_verts, ref_faces = make_sheet(40)
N_ref = ref_verts.shape[0]
spacing = W/40
ref_scale = torch.full((N_ref,3),spacing,device=device); ref_scale[:,2]=spacing*0.1
ref_rot = torch.zeros(N_ref,4,device=device); ref_rot[:,0]=1.0

log(f"REF_41: {N_ref} Gaussians")

# ═══════════════════════════════════════════════════════════
# 2. Alternative representations
# ═══════════════════════════════════════════════════════════
class Repr:
    def __init__(self, name, verts, faces, scale, rot, tau_raw, color_raw, divs):
        self.name=name; self.verts=verts; self.faces=faces; self.scale=scale
        self.rot=rot; self.tau_raw=tau_raw; self.color_raw=color_raw
        self.divs=divs; self.N=verts.shape[0]
    def make_adapter(self):
        class A:
            def __init__(self):
                self._xyz=self_xyz; self._scaling=torch.log(self_scale.clamp(min=1e-8))
                self._rotation=self_rot; self._tau_raw=self_tau; self._color_raw=self_col
                self.active_sh_degree=0; self.max_sh_degree=0; self.use_app=False
            @property
            def get_xyz(self): return self._xyz
            @property
            def get_scaling(self): return torch.exp(self._scaling)
            @property
            def get_rotation(self): return self._rotation/self._rotation.norm(dim=1,keepdim=True).clamp(min=1e-8)
            @property
            def get_opacity(self): return 1-torch.exp(-F.softplus(self._tau_raw))
            @property
            def get_transparency(self): return torch.full((self._xyz.shape[0],1),0.5,device=self._xyz.device)
            @property
            def get_features(self): return torch.sigmoid(self._color_raw).unsqueeze(1)
        self_xyz=self.verts; self_scale=self.scale; self_rot=self.rot
        self_tau=self.tau_raw; self_col=self.color_raw
        return A()
    def clone_with_tau(self, tau):
        return Repr(self.name, self.verts, self.faces, self.scale, self.rot, tau, self.color_raw, self.divs)

# Build scale family
reprs=[]
# REF_41
reprs.append(Repr("REF_41", ref_verts, ref_faces, ref_scale, ref_rot, ref_tau_raw.clone(), ref_color_raw.clone(), 40))

for mul,name in [(0.5,"S05"),(0.75,"S075"),(1.5,"S15"),(2.0,"S20")]:
    sc=ref_scale.clone()*mul
    tau0=torch.full((N_ref,1), 0.0, device=device)
    col0=torch.zeros(N_ref,3,device=device)
    reprs.append(Repr(name, ref_verts, ref_faces, sc, ref_rot, tau0, col0, 40))

# Resolution family
for divs,name in [(20,"R21"),(40,"R41"),(80,"R81")]:
    N_r=(divs+1)**2
    sp=W/divs
    v_r=torch.tensor([[-W/2+sp*i,-H/2+sp*j,0.0] for i in range(divs+1) for j in range(divs+1)],dtype=torch.float32,device=device)
    sc_r=torch.full((N_r,3),sp,device=device); sc_r[:,2]=sp*0.1
    ro_r=torch.zeros(N_r,4,device=device); ro_r[:,0]=1.0
    tau0=torch.full((N_r,1),0.0,device=device)
    col0=torch.zeros(N_r,3,device=device)
    reprs.append(Repr(name, v_r, None, sc_r, ro_r, tau0, col0, divs))

log(f"Total representations: {len(reprs)}")

# ═══════════════════════════════════════════════════════════
# 3. White-pass + fitting functions
# ═══════════════════════════════════════════════════════════
def white_pass_adapter(adapter, cam):
    r2=render(cam,adapter,pipe,bg_color,app_model=None,override_color=torch.ones_like(torch.sigmoid(adapter._color_raw)),return_plane=False,return_depth_normal=False)
    return r2["render"].mean(dim=0,keepdim=True).clamp(0,1), r2["radii"]

def render_repr(rep, cam_idx):
    adpt = rep.make_adapter()
    A,_ = white_pass_adapter(adpt, film_cams[cam_idx])
    return A

def get_interior_mask(rep, cam_idx):
    A = render_repr(rep, cam_idx)
    return (A > 0.01).bool()

# ═══════════════════════════════════════════════════════════
# 4. Canonical equivalence fitting
# ═══════════════════════════════════════════════════════════
hdr("2. Canonical equivalence fitting")
# First render REF_41 as target
ref_rep = reprs[0]
target_A = {}
for ci, cid in enumerate(cam_ids_list):
    A = render_repr(ref_rep, ci)
    target_A[cid] = A

# Fit each alternative
for i, rep in enumerate(reprs):
    if rep.name == "REF_41" or rep.name == "R41":  # skip reference (REF_41 is at index 0, R41 is already equivalent)
        continue
    log(f"  Fitting {rep.name} ({rep.N} Gaussians)...")
    tau_fit = rep.tau_raw.clone().requires_grad_(True)
    opt = torch.optim.Adam([tau_fit], lr=1e-2)
    
    for it in range(3000):
        opt.zero_grad(); loss=0
        rep_fit = rep.clone_with_tau(tau_fit)
        for ci, cid in enumerate(cam_ids_list):
            if cid not in fit_ids: continue
            A_alt = render_repr(rep_fit, ci)
            # Only compare in valid region
            mask = get_interior_mask(ref_rep, ci)
            if mask.sum() < 10: continue
            te_alt = -torch.log(1 - A_alt.clamp(max=0.9999))
            te_ref = -torch.log(1 - target_A[cid].clamp(max=0.9999))
            loss += (A_alt - target_A[cid]).abs().mean() + 0.2*(te_alt - te_ref).abs().mean()
        loss.backward(retain_graph=True); opt.step()
        if it%1000==0: log(f"    iter {it}: loss={loss.item():.6f}")
    
    rep.tau_raw = tau_fit.detach().clone()
    log(f"    Done. tau mean={F.softplus(rep.tau_raw).mean().item():.4f}")

# ═══════════════════════════════════════════════════════════
# 5. Canonical equivalence evaluation
# ═══════════════════════════════════════════════════════════
hdr("3. Equivalence evaluation")
eq_rows=[]
for i, rep in enumerate(reprs):
    if rep.name == "R41": continue
    fit_errs=[]; hold_errs=[]
    for ci, cid in enumerate(cam_ids_list):
        A_alt = render_repr(rep, ci)
        mask = get_interior_mask(ref_rep, ci)
        if mask.sum()<10: continue
        mae = (A_alt - target_A[cid]).abs()[mask].mean().item()
        if cid in fit_ids: fit_errs.append(mae)
        else: hold_errs.append(mae)
    
    fit_ok = np.mean(fit_errs) <= 0.002 if fit_errs else False
    hold_ok = np.mean(hold_errs) <= 0.005 if hold_errs else False
    eq_status = "CANONICAL_EQUIVALENT" if (fit_ok and hold_ok) else "FAIL"
    log(f"  {rep.name:8s}: fit_MAE={np.mean(fit_errs):.4f} hold_MAE={np.mean(hold_errs):.4f} -> {eq_status}")
    eq_rows.append({"name":rep.name,"N":rep.N,"fit_MAE":np.mean(fit_errs) if fit_errs else 99,"hold_MAE":np.mean(hold_errs) if hold_errs else 99,"status":eq_status})

# ═══════════════════════════════════════════════════════════
# 6. Deformation and response spread
# ═══════════════════════════════════════════════════════════
hdr("4. Deformation response spread")
verts_t=torch.tensor(np.array([[0.0,0.0,0.0]]),device=device)  # dummy
# Actually get the real verts from the canonical mesh
mesh=trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
verts_canon=torch.tensor(np.array(mesh.vertices),dtype=torch.float32,device=device)
z_range=(verts_canon[:,2].min().item(),verts_canon[:,2].max().item())

# For scale family, use the same 41x41 grid
# For resolution family, create deformed verts
def deform_verts(verts, name):
    if name=="canonical": return verts
    s=float(name.split("_")[1])
    dv=verts.clone()
    if name.startswith("stretch"): dv[:,0]*=s
    elif name.startswith("biaxial"): dv[:,0]*=s; dv[:,1]*=s
    elif name.startswith("twist"): dv=twist_def(verts, int(name.split("_")[1]), z_range)
    return dv

# For scale family representations, we need deformed verts on the 41x41 grid
def make_deformed_verts(divs, name):
    verts_flat=[[-W/2+W*i/divs,-H/2+H*j/divs,0.0] for i in range(divs+1) for j in range(divs+1)]
    vt=torch.tensor(verts_flat,dtype=torch.float32,device=device)
    return deform_verts(vt, name)

deform_states=["stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","twist_60"]
resp_rows=[]

for i, rep in enumerate(reprs):
    if rep.name == "R41": continue
    eq_status = next(r["status"] for r in eq_rows if r["name"]==rep.name)
    if eq_status != "CANONICAL_EQUIVALENT": continue
    
    # Compute canonical median tau_eff
    can_te_vals=[]
    for ci, cid in enumerate(cam_ids_list):
        A = render_repr(rep, ci)
        mask = get_interior_mask(ref_rep, ci)
        if mask.sum()>0:
            te = -math.log(1 - A[mask].median().item())
            can_te_vals.append(te)
    can_te = np.median(can_te_vals)
    
    for st in deform_states:
        # Deform geometry
        if "scale" in rep.name.lower() or rep.name=="REF_41":
            dv = make_deformed_verts(40, st)
            sc = rep.scale.clone()
        else:  # resolution family
            dv = make_deformed_verts(rep.divs, st)
            sc = rep.scale.clone()
        
        # Build deformed representation
        class DefAdapter:
            def __init__(self):
                self._xyz=dv; self._scaling=torch.log(sc.clamp(min=1e-8))
                self._rotation=rep.rot; self._tau_raw=rep.tau_raw; self._color_raw=rep.color_raw
                self.active_sh_degree=0; self.max_sh_degree=0; self.use_app=False
            @property
            def get_xyz(self): return self._xyz
            @property
            def get_scaling(self): return torch.exp(self._scaling)
            @property
            def get_rotation(self): return self._rotation/self._rotation.norm(dim=1,keepdim=True).clamp(min=1e-8)
            @property
            def get_opacity(self): return 1-torch.exp(-F.softplus(self._tau_raw))
            @property
            def get_transparency(self): return torch.full((dv.shape[0],1),0.5,device=device)
            @property
            def get_features(self): return torch.sigmoid(self._color_raw).unsqueeze(1)
        
        gm=DefAdapter()
        def_te_vals=[]
        for ci, cid in enumerate(cam_ids_list):
            r2=render(film_cams[ci],gm,pipe,bg_color,app_model=None,override_color=torch.ones_like(torch.sigmoid(rep.color_raw)),return_plane=False,return_depth_normal=False)
            A=r2["render"].mean(dim=0,keepdim=True).clamp(0,1)
            mask = get_interior_mask(ref_rep, ci)
            if mask.sum()>0:
                te = -math.log(1 - A[mask].median().item())
                def_te_vals.append(te)
        def_te = np.median(def_te_vals) if def_te_vals else can_te
        R = def_te / max(can_te, 1e-10)
        resp_rows.append({"repr":rep.name,"state":st,"can_te":can_te,"def_te":def_te,"R":R})

# Compute D_R
for st in deform_states:
    vals=[r["R"] for r in resp_rows if r["state"]==st]
    if vals:
        D_R = max(vals) - min(vals)
        log(f"  {st:20s}: D_R={D_R:.4f} (min={min(vals):.4f}, max={max(vals):.4f})")

log("\n=== Stage 3.2.6 complete ===")
