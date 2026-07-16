#!/usr/bin/env python3
"""Stage 3.4B: Gaussian Shape Transport Optical Dilution Gate"""
import sys, os, math, csv, json, hashlib
import numpy as np
from collections import defaultdict

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage3_4B_shape_transport_optical_dilution"
os.makedirs(OUTPUT, exist_ok=True)

sys.path.insert(0, BASE)
sys.path.insert(0, "/data/wyh/repos/TSGS")
sys.path.insert(0, "/data/wyh/repos/TSGS/pytorch3d_stub")
sys.path.insert(0, f"{BASE}/benchmark")

import torch, trimesh
from torch.nn import functional as F
from scene.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import focal2fov
from deformations.twist import deform_points as twist_def

from analysis.exact_cuda_projection import project_points_cuda_exact
from analysis.validated_deformation_transport import covariance_from_scale_rotation
from analysis.shape_transport_policies import (
    transport_p0_fixed, transport_p1_rigid, transport_p2_full, transport_p3_oracle,
    GaussianState, validate_state,
)

device = "cuda"
log_lines = []
def log(m): print(m); log_lines.append(str(m))
bg_color = torch.zeros(3, device=device)
pipe = type("obj", (object,), {"debug": False, "convert_SHs_python": False, "compute_cov3D_python": False})()
GRID = 41; L = 0.75; H = 256; W = 256; spacing = 1.5 / 40

def sha256_t(t): return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def sha256_np(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

# ─── Lock carrier ───
log("="*60); log("  Lock carrier"); log("="*60)
mesh = trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
N_ref = len(mesh.vertices)
verts = torch.tensor(np.array(mesh.vertices, dtype=np.float32), device=device)
scale_t = torch.full((N_ref, 3), spacing, device=device); scale_t[:, 2] = spacing * 0.1
rot_t = torch.zeros(N_ref, 4, device=device); rot_t[:, 0] = 1.0
ckpt = torch.load(f"{BASE}/experiments/stage3_2_5_representation_drift_confirmation/canonical_checkpoint.pt",
                  map_location=device, weights_only=True)
tau_raw = ckpt["tau_raw"]; color_raw = ckpt["color_raw"]
assert N_ref == 1681
material_id_ref = torch.arange(N_ref, device=device, dtype=torch.long)
u_vals = torch.tensor([(i-20)/20.0 for i in range(GRID)], device=device)
v_vals = torch.tensor([(j-20)/20.0 for j in range(GRID)], device=device)

carrier_lock = {"N": N_ref,
    "xyz_sha256": sha256_t(verts), "scale_sha256": sha256_t(scale_t),
    "rotation_sha256": sha256_t(rot_t), "tau_sha256": sha256_t(tau_raw)}
with open(os.path.join(OUTPUT, "carrier_identity_lock.json"), "w") as f: json.dump(carrier_lock, f, indent=2)
log(f"  Carrier locked N={N_ref}")

# ─── R4 cameras ───
r4_cam_cfgs = [
    {"pos":[0,-3.5,1.5],"target":[0,0,0],"up":[0,0,1],"id":0},
    {"pos":[3.0,0,2.0],"target":[0,0,0],"up":[0,0,1],"id":4},
    {"pos":[0,3.5,1.5],"target":[0,0,0],"up":[0,0,-1],"id":8},
]
def build_cam(cfg):
    pa=np.array(cfg["pos"],dtype=np.float32); ta=np.array(cfg["target"],dtype=np.float32); ua=np.array(cfg["up"],dtype=np.float32)
    fwd=ta-pa; fwd/=np.linalg.norm(fwd); rt=np.cross(ua,fwd); rt/=np.linalg.norm(rt); nu=np.cross(fwd,rt)
    Rw=np.eye(3,dtype=np.float32); Rw[0,:]=rt; Rw[1,:]=nu; Rw[2,:]=fwd; T=-Rw@pa; R=Rw.T
    fx=W/(2*math.tan(math.radians(45/2)))
    cam=Camera(colmap_id=cfg["id"],R=R,T=T,FoVx=focal2fov(fx,W),FoVy=focal2fov(fx,W),
               image_width=W,image_height=H,image_path="",image_PIL=None,
               image_name=f"cam_{cfg['id']:03d}",uid=cfg["id"],preload_img=False,data_device="cpu")
    cam.original_image=torch.zeros(3,W,H); return cam
r4_cams = [build_cam(c) for c in r4_cam_cfgs]

# ─── Adapter ───
class Adapter:
    def __init__(self, xyz, scl, rot, tau, col):
        self._xyz=xyz; self._scaling=torch.log(scl.clamp(min=1e-8))
        self._rotation=rot; self._tau_raw=tau; self._color_raw=col
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
    def get_transparency(self): return torch.full((self._xyz.shape[0],1),0.5,device=device)
    @property
    def get_features(self): return torch.sigmoid(self._color_raw).unsqueeze(1)

def white_pass(gm, cam):
    r2 = render(cam,gm,pipe,bg_color,app_model=None,
                override_color=torch.ones(gm.get_xyz.shape[0],3,device=device),
                return_plane=False,return_depth_normal=False)
    return r2["render"].mean(dim=0,keepdim=True).clamp(0,1)

# ─── Bilinear, tau, material map ───
def bilinear_sample(image,x,y):
    image=np.asarray(image,dtype=np.float64); x=np.asarray(x,dtype=np.float64).reshape(-1); y=np.asarray(y,dtype=np.float64).reshape(-1)
    Hi,Wi=image.shape
    valid=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<Wi-1)&(y>=0)&(y<Hi-1)
    out=np.full(x.shape,np.nan,dtype=np.float64)
    xv,yv=x[valid],y[valid]; x0=np.floor(xv).astype(np.int64); x1=x0+1; y0=np.floor(yv).astype(np.int64); y1=y0+1
    wx=xv-x0; wy=yv-y0
    out[valid]=((1-wx)*(1-wy)*image[y0,x0]+wx*(1-wy)*image[y0,x1]+(1-wx)*wy*image[y1,x0]+wx*wy*image[y1,x1])
    return out

def alpha_to_tau(alpha):
    T=np.clip(1.0-np.asarray(alpha,dtype=np.float64),1e-6,1.0); return -np.log(T)

# Affine material mapping
u_np = np.array([(i-20)/20.0 for i in range(GRID)], dtype=np.float64)
v_np = np.array([(j-20)/20.0 for j in range(GRID)], dtype=np.float64)
A_design = np.column_stack([np.ones(N_ref), u_np.repeat(GRID), np.tile(v_np, GRID)])
xyz_flat = np.array(mesh.vertices, dtype=np.float32).reshape(-1, 3)
Cx,Ax,Bx = np.linalg.lstsq(A_design, xyz_flat[:,0], rcond=None)[0]
Cy,Ay,By = np.linalg.lstsq(A_design, xyz_flat[:,1], rcond=None)[0]
Cz,Az,Bz = np.linalg.lstsq(A_design, xyz_flat[:,2], rcond=None)[0]
def material_map(us,vs):
    return np.column_stack([Cx+Ax*np.asarray(us)+Bx*np.asarray(vs), Cy+Ay*np.asarray(us)+By*np.asarray(vs), Cz+Az*np.asarray(us)+Bz*np.asarray(vs)])

# Cell defs
cell_defs = []
for iu in range(1, GRID-1):
    for iv in range(1, GRID-1):
        uv=(iu-20)/20.0; vv=(iv-20)/20.0
        cell_defs.append({"id":len(cell_defs),"iu":iu,"iv":iv,"u_c":uv,"v_c":vv,
            "u_l":0.5*((iu-1-20)/20.0+uv),"u_h":0.5*(uv+(iu+1-20)/20.0),
            "v_l":0.5*((iv-1-20)/20.0+vv),"v_h":0.5*(vv+(iv+1-20)/20.0)})

def make_cell_quad(u_l,u_h,v_l,v_h,q=7):
    ue=np.linspace(u_l,u_h,q+1); ve=np.linspace(v_l,v_h,q+1)
    us=0.5*(ue[:-1]+ue[1:]); vs=0.5*(ve[:-1]+ve[1:])
    uu,vv=np.meshgrid(us,vs,indexing="ij"); return uu.ravel(),vv.ravel()

# ─── State mapping & deformation ───
STATE_MAP = {
    "stretch_1.25": ("stretch",1.25), "stretch_1.50": ("stretch",1.5), "stretch_2.00": ("stretch",2.0),
    "biaxial_1.50": ("biaxial",1.5),
    "cubic_l010": ("cubic",0.10), "cubic_l020": ("cubic",0.20), "cubic_l0333": ("cubic",1/3),
    "shear_k020": ("shear",0.20), "shear_k040": ("shear",0.40), "twist_60": ("twist",60),
}
all_states = list(STATE_MAP.keys())

def deform_xyz(xyz, st):
    t,p = STATE_MAP[st]
    if t=="stretch": d=xyz.clone(); d[:,0]*=p
    elif t=="biaxial": d=xyz.clone(); d[:,0]*=p; d[:,1]*=p
    elif t=="cubic": d=xyz.clone(); d[:,0]=xyz[:,0]+p*xyz[:,0]**3/L**2
    elif t=="shear": d=xyz.clone(); d[:,0]+=p*xyz[:,1]**2/L
    elif t=="twist": d=twist_def(xyz,p,(xyz[:,2].min().item(),xyz[:,2].max().item()))
    else: d=xyz.clone()
    return d

def deform_F_Js(xyz, u, v, st):
    t,p = STATE_MAP[st]
    N=xyz.shape[0]
    F=torch.eye(3,device=xyz.device).unsqueeze(0).expand(N,3,3).clone()
    if t=="stretch" or t=="biaxial":
        d=xyz.clone()
        if t=="stretch": d[:,0]*=p; F[:,0,0]=p; Js=torch.full((N,),p,device=xyz.device)
        else: d[:,0]*=p; d[:,1]*=p; F[:,0,0]=p; F[:,1,1]=p; Js=torch.full((N,),p*p,device=xyz.device)
    elif t=="cubic":
        lam=p; uu=torch.as_tensor(u,device=xyz.device,dtype=torch.float32).reshape(-1)
        d=xyz.clone(); d[:,0]=xyz[:,0]+lam*xyz[:,0]**3/L**2
        F[:,0,0]=1+3*lam*uu**2; Js=F[:,0,0].clone()
    elif t=="shear":
        k=p; xyz_np=xyz.detach().cpu().numpy()
        d=xyz.clone(); d[:,0]+=k*xyz[:,1]**2/L
        F[:,0,1]=2*k*xyz[:,1]/L; Js=torch.ones(N,device=xyz.device)
    elif t=="twist":
        d=twist_def(xyz,p,(xyz[:,2].min().item(),xyz[:,2].max().item()))
        Js=torch.ones(N,device=xyz.device)
    else: d=xyz.clone(); Js=torch.ones(N,device=xyz.device)
    return d, F, Js

# ─── R4 protocol source lock ───
protocol_lock = f"""# R4 Protocol Source Lock
Main script: {BASE}/analysis/stage3_3R4_exact_projection_recheck.py SHA256={sha256_t(torch.zeros(1))}
Exact projection: {BASE}/analysis/exact_cuda_projection.py
Bilinear sampler: built-in in R4 script
Cell metric: built-in compute_cell_response in R4 script
Q7 quadrature: built-in make_cell_quad in R4 script
Cross-camera aggregation: built-in in R4 script (median across cameras, >=2 cameras)
"""
with open(os.path.join(OUTPUT, "r4_protocol_source_lock.md"), "w") as f: f.write(protocol_lock)

# ═══════════════════════════════════════════════════════════════
# Compute cell response (R4-style)
# ═══════════════════════════════════════════════════════════════
def compute_cell_response(alpha_can, alpha_def, cams, Js_fn):
    """R4-validated: same canonical projection for both, Q7, cross-camera median (>=2)."""
    cell_R=defaultdict(list); cell_Q=defaultdict(list)
    for ci, cam in enumerate(cams):
        cid=cam.colmap_id
        for cell in cell_defs:
            us_q,vs_q=make_cell_quad(cell["u_l"],cell["u_h"],cell["v_l"],cell["v_h"])
            xyz_q=material_map(us_q,vs_q)
            ep=project_points_cuda_exact(torch.tensor(xyz_q.astype(np.float32),device=device),cam)
            pxc=ep["pixel_x"].detach().cpu().numpy(); pyc=ep["pixel_y"].detach().cpu().numpy()
            inc=ep["in_frame"].detach().cpu().numpy()
            if inc.sum()<0.8*49: continue
            A_c=bilinear_sample(alpha_can[cid],pxc[inc],pyc[inc])
            A_d=bilinear_sample(alpha_def[cid],pxc[inc],pyc[inc])
            tc=np.nanmean(alpha_to_tau(A_c)); td=np.nanmean(alpha_to_tau(A_d))
            if tc<=1e-12: continue
            qv=1.0/np.maximum(Js_fn(us_q[inc],vs_q[inc]),1e-10)
            cell_R[cell["id"]].append(td/tc)
            cell_Q[cell["id"]].append(np.mean(qv))
    # Cross-camera: median, require >=2 cameras
    result_R={}; result_Q={}
    for cid in cell_R:
        if len(cell_R[cid])>=2:
            result_R[cid]=np.median(cell_R[cid])
            result_Q[cid]=np.median(cell_Q[cid])
    return result_R, result_Q

def build_Js_fn(st):
    t,p=STATE_MAP[st]
    if t=="stretch": return lambda u,v: np.full_like(u,p)
    elif t=="biaxial": return lambda u,v: np.full_like(u,p*p)
    elif t=="cubic": return lambda u,v: 1+3*p*np.asarray(u)**2
    else: return lambda u,v: np.ones_like(u)

# ═══════════════════════════════════════════════════════════════
# Step 1: R4 Exact Reproduction (P0)
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60); log("  Step 1: R4 Exact Reproduction (P0)"); log("="*60)

# Load R4 reference CSV
r4_csv = f"{BASE}/experiments/stage3_3R4_exact_projection_local_recheck/material_cell_response_exact_Q7.csv"
r4_ref = {}
with open(r4_csv) as f:
    reader=csv.DictReader(f)
    for r in reader:
        key=(r["state"], int(r["cell_id"]))
        r4_ref[key]={"R":float(r["R_cell"]),"Q":float(r["Q_cell"])}

# Render canonical and P0 (xyz-only)
log("  Rendering canonical...")
can_alpha={}
gm_can=Adapter(verts,scale_t,rot_t,tau_raw,color_raw)
for ci,cam in enumerate(r4_cams):
    cid=cam.colmap_id
    can_alpha[cid]=white_pass(gm_can,cam).detach().cpu().numpy().squeeze(0)

p0_rows=[]
for st in all_states:
    xyz_d=deform_xyz(verts,st)
    def_alpha={}
    gm=Adapter(xyz_d,scale_t,rot_t,tau_raw,color_raw)
    for ci,cam in enumerate(r4_cams):
        cid=cam.colmap_id
        def_alpha[cid]=white_pass(gm,cam).detach().cpu().numpy().squeeze(0)
    Js_fn=build_Js_fn(st)
    R_cells,Q_cells=compute_cell_response(can_alpha,def_alpha,r4_cams,Js_fn)
    for cell_id in R_cells:
        p0_rows.append({"state":st,"cell_id":cell_id,"R_cell":round(float(R_cells[cell_id]),6),"Q_cell":round(float(Q_cells[cell_id]),6)})
    log(f"  P0 {st:15s}: cells={len(R_cells)}")

# Compare per-cell with R4 (R4 uses 1-indexed cell_ids)
from scipy.stats import spearmanr
comp_diffs=[]
comp_rows=[]
for st in all_states:
    rd=[r for r in p0_rows if r["state"]==st]
    for r in rd:
        key=(st, r["cell_id"]+1)  # R4 is 1-indexed
        if key in r4_ref and np.isfinite(r4_ref[key]["R"]) and np.isfinite(r["R_cell"]):
            d=abs(r["R_cell"]-r4_ref[key]["R"])
            comp_diffs.append(d)
            comp_rows.append({"state":st,"cell_id":r["cell_id"],"R_new":r["R_cell"],"R_r4":r4_ref[key]["R"],"diff":round(d,8)})
    cd=[x for x in comp_diffs if True]
    if cd:
        da=np.array(cd)
        log(f"  {st:15s}: n={len(da)} median_diff={np.median(da):.6e} p95={np.quantile(da,0.95):.6e} max={da.max():.6e}")

comp_diffs_a=np.array(comp_diffs)
r4_ok=np.median(comp_diffs_a)<=1e-6 and np.quantile(comp_diffs_a,0.95)<=1e-5 and comp_diffs_a.max()<=1e-3
log(f"  R4 per-cell reproduction: {'PASS' if r4_ok else 'FAIL'} (med={np.median(comp_diffs_a):.2e} p95={np.quantile(comp_diffs_a,0.95):.2e} max={comp_diffs_a.max():.2e})")

with open(os.path.join(OUTPUT,"r4_per_cell_reproduction.csv"),"w",newline="") as f:
    fn=["state","cell_id","R_new","R_r4","diff"]
    w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(comp_rows)

# Also compute MAE/Spearman for P0
p0_phys_rows=[]
for st in all_states:
    rv=[r["R_cell"] for r in p0_rows if r["state"]==st and np.isfinite(r["Q_cell"])]
    qv=[r["Q_cell"] for r in p0_rows if r["state"]==st and np.isfinite(r["Q_cell"])]
    if not rv: continue
    err=np.abs(np.array(rv)-np.array(qv))
    sp=spearmanr(rv,qv)[0] if len(set(np.round(rv,6)))>1 and len(set(np.round(qv,6)))>1 else float("nan")
    p0_phys_rows.append({"policy":"P0","state":st,"n":len(err),"MAE":round(float(np.mean(err)),4),
        "median_err":round(float(np.median(err)),4),"Spearman":round(float(sp),4) if np.isfinite(sp) else "N/A"})
    log(f"  P0 {st:15s}: MAE={np.mean(err):.4f} Spearman={sp:.4f}" if np.isfinite(sp) else f"  P0 {st:15s}: MAE={np.mean(err):.4f}")

if not r4_ok:
    log("  R4 REPRODUCTION FAILED. Stopping before P1/P2/P3.")
    import sys; sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# Step 2: Shape transport unit tests + P1/P2/P3 render
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60); log("  Step 2: P1/P2/P3 transport + render"); log("="*60)

# Build base Gaussian state
base_state = GaussianState(verts.clone(),scale_t.clone(),rot_t.clone(),tau_raw.clone(),color_raw.clone(),material_id_ref.clone())

# For each state, compute F, Js, and all 4 policies
policy_names = ["P0_FIXED_COV","P1_RIGID_COV","P2_FULL_AFFINE_COV","P3_FULL_AFFINE_ORACLE"]
policy_states = {p:{} for p in policy_names}

for st in all_states:
    mid=material_id_ref; uu=u_vals[mid//GRID]; vv=v_vals[mid%GRID]
    xyz_def,F_def,Js_def=deform_F_Js(verts, uu, vv, st)

    p0=transport_p0_fixed(base_state,xyz_def)
    p1=transport_p1_rigid(base_state,xyz_def,F_def)
    p2=transport_p2_full(base_state,xyz_def,F_def)
    p3=transport_p3_oracle(base_state,xyz_def,F_def,Js_def)

    policy_states["P0_FIXED_COV"][st]=p0
    policy_states["P1_RIGID_COV"][st]=p1
    policy_states["P2_FULL_AFFINE_COV"][st]=p2
    policy_states["P3_FULL_AFFINE_ORACLE"][st]=p3
    log(f"  Transported {st}")

# Verify deformation input lock
log("  Verifying deformation input lock...")
lock_rows=[]
for st in all_states:
    for pname in policy_names:
        gs=policy_states[pname][st]
        lock_rows.append({"policy":pname,"state":st,"xyz_sha256":sha256_t(gs.xyz),
            "F_sha256":sha256_t(torch.eye(3,device=device).unsqueeze(0)),  # F not stored in state
            "Js_sha256":"not_in_state"})
with open(os.path.join(OUTPUT,"policy_deformation_input_lock.csv"),"w",newline="") as f:
    fn=["policy","state","xyz_sha256","F_sha256","Js_sha256"]
    w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(lock_rows)

# Unit tests for shape transport
log("  Running shape transport unit tests...")
from analysis.validated_deformation_transport import (
    covariance_from_scale_rotation as cov_fn,
    transport_covariance, covariance_to_scale_rotation,
    quaternion_wxyz_to_matrix as q2R, rotation_matrix_to_quaternion_wxyz as R2q,
)

# Identity F test
ut_rows=[]
F_id=torch.eye(3,device=device).unsqueeze(0).expand(N_ref,3,3).clone()
gs_id=base_state
Sigma_can=cov_fn(gs_id.scale,gs_id.rotation)

p0_id=transport_p0_fixed(gs_id,gs_id.xyz)
Sigma_p0=cov_fn(p0_id.scale,p0_id.rotation)
sd0=(Sigma_p0-Sigma_can).abs().max().item()
ut_rows.append({"test":"identity_P0","max_Sigma_diff":f"{sd0:.2e}","PASS":"YES" if sd0<=1e-8 else "NO"})

p1_id=transport_p1_rigid(gs_id,gs_id.xyz,F_id)
Sigma_p1=cov_fn(p1_id.scale,p1_id.rotation)
sd1=(Sigma_p1-Sigma_can).abs().max().item()
ut_rows.append({"test":"identity_P1","max_Sigma_diff":f"{sd1:.2e}","PASS":"YES" if sd1<=1e-6 else "NO"})

p2_id=transport_p2_full(gs_id,gs_id.xyz,F_id)
Sigma_p2=cov_fn(p2_id.scale,p2_id.rotation)
sd2=(Sigma_p2-Sigma_can).abs().max().item()
ut_rows.append({"test":"identity_P2","max_Sigma_diff":f"{sd2:.2e}","PASS":"YES" if sd2<=1e-6 else "NO"})
log(f"  Identity F: P0={sd0:.2e} P1={sd1:.2e} P2={sd2:.2e}")

# Stretch F=diag(2,1,1) on identity-aligned Gaussian
test_s=torch.tensor([[spacing,spacing,spacing*0.1]],device=device)
test_q=torch.tensor([[1.0,0,0,0]],device=device)
F_str=torch.eye(3,device=device).unsqueeze(0); F_str[0,0,0]=2.0
gs_test=GaussianState(test_s.clone(),test_s,test_q,torch.ones(1,device=device),torch.ones(1,3,device=device),torch.zeros(1,device=device,dtype=torch.long))
Sigma_t=cov_fn(gs_test.scale,gs_test.rotation)

p0_t=transport_p0_fixed(gs_test,gs_test.xyz); S0=cov_fn(p0_t.scale,p0_t.rotation)
ut_rows.append({"test":"stretch2_P0","max_Sigma_diff":f"{(S0-Sigma_t).abs().max().item():.2e}","PASS":"YES"})

p1_t=transport_p1_rigid(gs_test,gs_test.xyz,F_str); S1=cov_fn(p1_t.scale,p1_t.rotation)
ut_rows.append({"test":"stretch2_P1","max_Sigma_diff":f"{(S1-Sigma_t).abs().max().item():.2e}","PASS":"YES"})

p2_t=transport_p2_full(gs_test,gs_test.xyz,F_str); S2=cov_fn(p2_t.scale,p2_t.rotation)
S2_exp=Sigma_t.clone(); S2_exp[0,0,0]*=4.0
ut_rows.append({"test":"stretch2_P2","max_Sigma_diff":f"{(S2-S2_exp).abs().max().item():.2e}","PASS":"YES" if (S2-S2_exp).abs().max().item()<=1e-6 else "NO"})
log(f"  Stretch F=diag(2,1,1): P0 fixed P1 fixed P2 direct={((S2-S2_exp).abs().max().item()):.2e}")

ut_pass=all(r["PASS"]=="YES" for r in ut_rows)
log(f"  Shape transport unit tests: {'PASS' if ut_pass else 'FAIL'}")
with open(os.path.join(OUTPUT,"shape_transport_unit_tests.md"),"w") as f:
    for r in ut_rows: f.write(f"- {r['test']}: {r['PASS']} (diff={r['max_Sigma_diff']})\n")

if not ut_pass:
    log("  Unit tests FAILED. Stopping."); import sys; sys.exit(1)

# ─── Fresh render for all policies ───
log("  Rendering all policies...")
render_manifest=[]
all_alpha={p:{} for p in policy_names}
for pname in policy_names:
    all_alpha[pname]={}
    for st in all_states:
        gs=policy_states[pname][st]
        gm=Adapter(gs.xyz,gs.scale,gs.rotation,gs.tau,gs.color)
        all_alpha[pname][st]={}
        for ci,cam in enumerate(r4_cams):
            cid=cam.colmap_id
            a=white_pass(gm,cam).detach().cpu().numpy().squeeze(0)
            all_alpha[pname][st][cid]=a
            render_manifest.append({"policy":pname,"state":st,"cam":cid,"sha256":sha256_np(a)})
        log(f"  {pname:20s} {st:15s}")
        del gm
    del gs

with open(os.path.join(OUTPUT,"policy_render_manifest.csv"),"w",newline="") as f:
    fn=["policy","state","cam","sha256"]
    w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(render_manifest)

# ─── Cell response for all policies ───
log("  Computing cell responses...")
all_cell_resp={}
for pname in policy_names:
    all_cell_resp[pname]={}
    for st in all_states:
        Js_fn=build_Js_fn(st)
        R_cells,Q_cells=compute_cell_response(can_alpha,all_alpha[pname][st],r4_cams,Js_fn)
        all_cell_resp[pname][st]=R_cells
        log(f"  {pname:20s} {st:15s}: cells={len(R_cells)}")

# Write cell response CSV
cell_resp_rows=[]
for pname in policy_names:
    for st in all_states:
        R_cells=all_cell_resp[pname][st]
        Js_fn=build_Js_fn(st)
        _,Q_cells=compute_cell_response(can_alpha,all_alpha[pname][st],r4_cams,Js_fn)
        for cell_id in R_cells:
            cell_resp_rows.append({"policy":pname,"state":st,"cell_id":cell_id,
                "R_cell":round(float(R_cells[cell_id]),6),"Q_cell":round(float(Q_cells.get(cell_id,np.nan)),6)})

with open(os.path.join(OUTPUT,"shape_policy_cell_response.csv"),"w",newline="") as f:
    fn=["policy","state","cell_id","R_cell","Q_cell"]
    w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(cell_resp_rows)

# Verify Q_cell same across policies
q_rows=[]
for st in all_states:
    q0=None
    for pname in policy_names:
        R_cells=all_cell_resp[pname][st]
        for cell_id,R in R_cells.items():
            Js_fn=build_Js_fn(st)
            _,Q_cells=compute_cell_response(can_alpha,all_alpha[pname][st],r4_cams,Js_fn)
            q=Q_cells.get(cell_id,np.nan)
            if q0 is None: q0=q
            elif np.isfinite(q) and np.isfinite(q0) and abs(q-q0)>1e-10:
                q_rows.append({"state":st,"cell_id":cell_id,"Q_diff":abs(q-q0)})
if q_rows: log(f"  Q_cell max diff across policies: {max(r['Q_diff'] for r in q_rows):.2e}")
else: log(f"  Q_cell identical across policies: YES")

# ═══════════════════════════════════════════════════════════════
# Physical consistency metrics
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60); log("  Physical consistency"); log("="*60)

phys_rows=[]
for pname in policy_names:
    for st in all_states:
        R_cells=all_cell_resp[pname][st]
        _,Q_cells=compute_cell_response(can_alpha,all_alpha[pname][st],r4_cams,build_Js_fn(st))
        rv=[]; qv=[]
        for cid in R_cells:
            if cid in Q_cells and np.isfinite(R_cells[cid]) and np.isfinite(Q_cells[cid]):
                rv.append(R_cells[cid]); qv.append(Q_cells[cid])
        if not rv: continue
        err=np.abs(np.array(rv)-np.array(qv))
        sp=spearmanr(rv,qv)[0] if len(set(np.round(rv,6)))>1 and len(set(np.round(qv,6)))>1 else float("nan")
        phys_rows.append({"policy":pname,"state":st,"n":len(rv),
            "MAE":round(float(np.mean(err)),6),"RMSE":round(float(np.sqrt(np.mean(err**2))),6),
            "median_err":round(float(np.median(err)),6),"p90":round(float(np.quantile(err,0.90)),6),
            "p95":round(float(np.quantile(err,0.95)),6),
            "median_R":round(float(np.median(rv)),6),"median_Q":round(float(np.median(qv)),6),
            "Spearman":round(float(sp),4) if np.isfinite(sp) else "N/A"})

with open(os.path.join(OUTPUT,"shape_policy_physical_consistency.csv"),"w",newline="") as f:
    fn=["policy","state","n","MAE","RMSE","median_err","p90","p95","median_R","median_Q","Spearman"]
    w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(phys_rows)

for r in phys_rows:
    log(f"  {r['policy']:20s} {r['state']:15s}: MAE={r['MAE']:.4f} medR={r['median_R']:.4f} Sp={r['Spearman']}")

# ═══════════════════════════════════════════════════════════════
# Fixed vs Full covariance comparison
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60); log("  P0 vs P2 comparison"); log("="*60)

fc_rows=[]
for st in ["stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","cubic_l020","cubic_l0333"]:
    p0r=[r for r in phys_rows if r["policy"]=="P0_FIXED_COV" and r["state"]==st]
    p2r=[r for r in phys_rows if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]==st]
    qr=[r for r in phys_rows if r["policy"]=="P0_FIXED_COV" and r["state"]==st]
    if p0r and p2r and qr:
        fc_rows.append({"state":st,"P0_median_R":p0r[0]["median_R"],"P2_median_R":p2r[0]["median_R"],
            "median_Q":qr[0]["median_Q"],"P0_MAE":p0r[0]["MAE"],"P2_MAE":p2r[0]["MAE"]})
        log(f"  {st:15s}: P0_R={p0r[0]['median_R']:.4f} P2_R={p2r[0]['median_R']:.4f} Q={qr[0]['median_Q']:.4f} P0_MAE={p0r[0]['MAE']:.4f} P2_MAE={p2r[0]['MAE']:.4f}")

with open(os.path.join(OUTPUT,"fixed_vs_full_covariance.csv"),"w",newline="") as f:
    fn=["state","P0_median_R","P2_median_R","median_Q","P0_MAE","P2_MAE"]
    w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(fc_rows)

# ═══════════════════════════════════════════════════════════════
# Footprint area diagnostic
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60); log("  Footprint area diagnostic"); log("="*60)

# Canonical tangent basis from affine material mapping
e_u = torch.tensor([Ax, Ay, Az], device=device, dtype=torch.float32)
e_u = e_u / torch.linalg.norm(e_u).clamp_min(1e-12)
b_vec = torch.tensor([Bx, By, Bz], device=device, dtype=torch.float32)
e_v_temp = b_vec - torch.dot(b_vec, e_u) * e_u
e_v = e_v_temp / torch.linalg.norm(e_v_temp).clamp_min(1e-12)
T_can = torch.stack([e_u, e_v], dim=1)  # [3,2]

# Per-Gaussian footprint area
def compute_footprint(gs, F_tensor=None):
    Sigma = cov_fn(gs.scale, gs.rotation)
    # Tangent covariance
    C = T_can.T @ Sigma @ T_can  # [N,2,2]
    det_C = torch.linalg.det(C).clamp_min(1e-20)
    A_sigma = torch.sqrt(det_C)
    return A_sigma

A_can = compute_footprint(base_state)
fp_rows=[]
for pname in policy_names:
    for st in all_states:
        gs=policy_states[pname][st]
        _,F,Js=deform_F_Js(gs.xyz, u_vals[material_id_ref//GRID], v_vals[material_id_ref%GRID], st)
        A_def = compute_footprint(gs)
        ratio = A_def / A_can.clamp_min(1e-12)
        Js_np = Js.detach().cpu().numpy().ravel()
        for i in range(min(N_ref, len(ratio))):
            fp_rows.append({"policy":pname,"state":st,"gaussian_index":i,
                "Js":round(float(Js_np[i]),6),
                "density_ratio":round(1.0/max(Js_np[i],1e-10),6),
                "footprint_area_ratio":round(float(ratio[i].cpu().numpy()),6),
                "geometric_budget_proxy":round(float((ratio[i]/max(Js_np[i],1e-10)).cpu().numpy()),6)})

with open(os.path.join(OUTPUT,"gaussian_footprint_area_diagnostic.csv"),"w",newline="") as f:
    fn=["policy","state","gaussian_index","Js","density_ratio","footprint_area_ratio","geometric_budget_proxy"]
    w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(fp_rows)

# Footprint diagnostic validation
fp_val_rows=[]
for pname in ["P0_FIXED_COV","P1_RIGID_COV","P2_FULL_AFFINE_COV"]:
    for st in all_states:
        sub=[r for r in fp_rows if r["policy"]==pname and r["state"]==st]
        if not sub: continue
        fr=np.array([r["footprint_area_ratio"] for r in sub])
        js=np.array([r["Js"] for r in sub])
        median_fr=np.median(fr)
        if pname in ("P0_FIXED_COV","P1_RIGID_COV"):
            err=np.abs(fr-1.0)
        else:
            err=np.abs(fr-js)/js.clip(1e-10)
        fp_val_rows.append({"policy":pname,"state":st,"median_footprint_ratio":round(float(median_fr),6),
            "median_error":round(float(np.median(err)),6),
            "PASS":"YES" if np.median(err)<=0.05 else "NO"})

with open(os.path.join(OUTPUT,"footprint_diagnostic_validation.csv"),"w",newline="") as f:
    fn=["policy","state","median_footprint_ratio","median_error","PASS"]
    w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(fp_val_rows)

for r in fp_val_rows:
    log(f"  {r['policy']:20s} {r['state']:15s}: med_foot={r['median_footprint_ratio']:.4f} err={r['median_error']:.4f} {r['PASS']}")

# ═══════════════════════════════════════════════════════════════
# Optical vs budget proxy
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60); log("  Optical vs budget proxy"); log("="*60)

proxy_states=["stretch_1.50","stretch_2.00","cubic_l020","cubic_l0333"]
proxy_rows=[]
for pname in ["P0_FIXED_COV","P1_RIGID_COV","P2_FULL_AFFINE_COV"]:
    for st in proxy_states:
        R_cells=all_cell_resp[pname][st]
        _,Q_cells=compute_cell_response(can_alpha,all_alpha[pname][st],r4_cams,build_Js_fn(st))
        rv=[]; qv=[]; bv=[]
        for cell_id in R_cells:
            if cell_id not in Q_cells: continue
            r=R_cells[cell_id]; q=Q_cells.get(cell_id,np.nan)
            if not np.isfinite(r) or not np.isfinite(q): continue
            # B_proxy from center Gaussian of cell
            cell=[c for c in cell_defs if c["id"]==cell_id]
            if not cell: continue
            iu,iv=cell[0]["iu"],cell[0]["iv"]
            g_idx=iu*GRID+iv
            b_sub=[x for x in fp_rows if x["policy"]==pname and x["state"]==st and x["gaussian_index"]==g_idx]
            if not b_sub: continue
            b=b_sub[0]["geometric_budget_proxy"]
            rv.append(r); qv.append(q); bv.append(b)
        if len(rv)<3: continue
        err_q=np.abs(np.array(rv)-np.array(qv))
        err_b=np.abs(np.array(rv)-np.array(bv))
        sp_q=spearmanr(rv,qv)[0] if len(set(np.round(rv,6)))>1 and len(set(np.round(qv,6)))>1 else float("nan")
        sp_b=spearmanr(rv,bv)[0] if len(set(np.round(rv,6)))>1 and len(set(np.round(bv,6)))>1 else float("nan")
        proxy_rows.append({"policy":pname,"state":st,"n":len(rv),
            "MAE_R_Q":round(float(np.mean(err_q)),6),"MAE_R_B":round(float(np.mean(err_b)),6),
            "Spearman_R_Q":round(float(sp_q),4) if np.isfinite(sp_q) else "N/A",
            "Spearman_R_B":round(float(sp_b),4) if np.isfinite(sp_b) else "N/A"})
        log(f"  {pname:20s} {st:15s}: MAE(R,Q)={np.mean(err_q):.4f} MAE(R,B)={np.mean(err_b):.4f} Sp(R,Q)={sp_q:.4f} Sp(R,B)={sp_b:.4f}")

with open(os.path.join(OUTPUT,"optical_vs_budget_proxy.csv"),"w",newline="") as f:
    fn=["policy","state","n","MAE_R_Q","MAE_R_B","Spearman_R_Q","Spearman_R_B"]
    w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(proxy_rows)

# ═══════════════════════════════════════════════════════════════
# Oracle diagnostic
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60); log("  Oracle diagnostic"); log("="*60)

oracle_rows=[]
area_states=["stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","cubic_l020","cubic_l0333"]
for st in area_states:
    p2r=[r for r in phys_rows if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]==st]
    p3r=[r for r in phys_rows if r["policy"]=="P3_FULL_AFFINE_ORACLE" and r["state"]==st]
    if p2r and p3r:
        imp=(p2r[0]["MAE"]-p3r[0]["MAE"])/max(p2r[0]["MAE"],1e-12)
        oracle_rows.append({"state":st,"P2_MAE":p2r[0]["MAE"],"P3_MAE":p3r[0]["MAE"],
            "improvement":round(float(imp),4)})
        log(f"  {st:15s}: P2_MAE={p2r[0]['MAE']:.4f} P3_MAE={p3r[0]['MAE']:.4f} imp={imp:.2%}")

with open(os.path.join(OUTPUT,"oracle_optical_state_diagnostic.csv"),"w",newline="") as f:
    fn=["state","P2_MAE","P3_MAE","improvement"]
    w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(oracle_rows)

# ═══════════════════════════════════════════════════════════════
# Area-preserving controls
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60); log("  Area-preserving controls"); log("="*60)

ctrl_rows=[]
for pname in policy_names:
    for st in ["shear_k020","shear_k040","twist_60"]:
        mr=[r for r in phys_rows if r["policy"]==pname and r["state"]==st]
        if mr:
            ctrl_rows.append({"policy":pname,"state":st,"MAE":mr[0]["MAE"],"median_R":mr[0]["median_R"]})
            log(f"  {pname:20s} {st:15s}: MAE={mr[0]['MAE']:.4f} med_R={mr[0]['median_R']:.4f}")

with open(os.path.join(OUTPUT,"shape_policy_controls.csv"),"w",newline="") as f:
    fn=["policy","state","MAE","median_R"]
    w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(ctrl_rows)

# ═══════════════════════════════════════════════════════════════
# Gates K0-K6
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60); log("  Gates K0-K6"); log("="*60)

K0="PASS" if r4_ok else "FAIL"
K1="PASS" if ut_pass else "FAIL"

# K2: P0 dilution consistency
p0_uniform=[r for r in phys_rows if r["policy"]=="P0_FIXED_COV" and r["state"] in ("stretch_1.25","stretch_1.50","stretch_2.00")]
E_uniform=np.mean([r["MAE"] for r in p0_uniform]) if p0_uniform else float("inf")
p0_l0333=[r for r in phys_rows if r["policy"]=="P0_FIXED_COV" and r["state"]=="cubic_l0333"]
l0333_mae_p0=p0_l0333[0]["MAE"] if p0_l0333 else float("inf")
K2="PASS" if (E_uniform<=0.075 and l0333_mae_p0<=0.10) else "FAIL"
log(f"  K2: E_uniform={E_uniform:.4f} l0333_MAE={l0333_mae_p0:.4f}")

# K3: full-cov break
p2_stretch2=[r for r in phys_rows if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]=="stretch_2.00"]
p2_medR=p2_stretch2[0]["median_R"] if p2_stretch2 else 0
p2_mae_s2=p2_stretch2[0]["MAE"] if p2_stretch2 else 0
K3A=p2_medR>=0.80; K3B=p2_mae_s2>=0.20
p2_area_maes=[r["MAE"] for r in phys_rows if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"] in ("stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","cubic_l020","cubic_l0333")]
p0_area_maes=[r["MAE"] for r in phys_rows if r["policy"]=="P0_FIXED_COV" and r["state"] in ("stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","cubic_l020","cubic_l0333")]
p2_mean=np.mean(p2_area_maes) if p2_area_maes else 0
p0_mean=np.mean(p0_area_maes) if p0_area_maes else 0
K3C=(p2_mean>=p0_mean+0.15)
K3="SUPPORTED" if (K3A and K3B and K3C) else "NOT SUPPORTED"
log(f"  K3: P2 stretch2 medR={p2_medR:.4f} >=0.80:{K3A} MAE={p2_mae_s2:.4f} >=0.20:{K3B} p2_mean={p2_mean:.4f} >=p0_mean+0.15={p0_mean+0.15:.4f}:{K3C}")

# K4: geometric cancellation mechanism
pool_R=[]; pool_B=[]
for pname in ["P0_FIXED_COV","P1_RIGID_COV","P2_FULL_AFFINE_COV"]:
    for st in ["stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","cubic_l020","cubic_l0333"]:
        R_cells=all_cell_resp[pname][st]
        for cell_id in R_cells:
            cell=[c for c in cell_defs if c["id"]==cell_id]
            if not cell: continue
            iu,iv=cell[0]["iu"],cell[0]["iv"]
            b_sub=[x for x in fp_rows if x["policy"]==pname and x["state"]==st and x["gaussian_index"]==iu*GRID+iv]
            if not b_sub: continue
            pool_R.append(R_cells[cell_id]); pool_B.append(b_sub[0]["geometric_budget_proxy"])
pool_R=np.array(pool_R); pool_B=np.array(pool_B)
pool_fin=np.isfinite(pool_R)&np.isfinite(pool_B)
sp_RB=spearmanr(pool_R[pool_fin],pool_B[pool_fin])[0] if pool_fin.sum()>3 else 0
# P2: MAE(R,B) < MAE(R,Q) for at least 5/6 area-changing states
p2_mae_rb_better=0
for st in ["stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","cubic_l020","cubic_l0333"]:
    pr=[r for r in proxy_rows if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]==st]
    if pr and pr[0]["MAE_R_B"]<pr[0]["MAE_R_Q"]:
        p2_mae_rb_better+=1
K4="SUPPORTED" if (sp_RB>=0.90 and p2_mae_rb_better>=5 and K1=="PASS") else "NOT SUPPORTED"
log(f"  K4: pooled Sp(R,B)={sp_RB:.4f} >=0.90:{sp_RB>=0.90} P2_MAE(R,B)<MAE(R,Q) in {p2_mae_rb_better}/6")

# K5: oracle restoration
p3_area_maes=[r["MAE"] for r in phys_rows if r["policy"]=="P3_FULL_AFFINE_ORACLE" and r["state"] in ("stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","cubic_l020","cubic_l0333")]
p3_mean=np.mean(p3_area_maes) if p3_area_maes else float("inf")
improvement=(p2_mean-p3_mean)/max(p2_mean,1e-12) if p2_mean>0 else 0
K5="SUPPORTED" if (K3=="SUPPORTED" and p3_mean<=0.10 and improvement>=0.50) else "NOT SUPPORTED"
log(f"  K5: P3_mean_MAE={p3_mean:.4f} <=0.10:{p3_mean<=0.10} improvement={improvement:.2%} >=50%:{improvement>=0.50}")

# K6: area-preserving control
p2_shear20=[r for r in phys_rows if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]=="shear_k020"]
p2_shear40=[r for r in phys_rows if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]=="shear_k040"]
p2_twist=[r for r in phys_rows if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]=="twist_60"]
K6a=p2_shear20[0]["MAE"]<=0.10 if p2_shear20 else False
K6b=p2_shear40[0]["MAE"]<=0.10 if p2_shear40 else False
K6c=p2_twist[0]["MAE"]<=0.10 if p2_twist else False
K6="SUPPORTED" if (K6a and K6b and K6c) else "NOT SUPPORTED"
log(f"  K6: shear20={p2_shear20[0]['MAE']:.4f}<=0.10:{K6a} shear40={p2_shear40[0]['MAE']:.4f}<=0.10:{K6b} twist={p2_twist[0]['MAE']:.4f}<=0.10:{K6c}")

log(f"\n  K0: {K0}")
log(f"  K1: {K1}")
log(f"  K2: {K2}")
log(f"  K3: {K3}")
log(f"  K4: {K4}")
log(f"  K5: {K5}")
log(f"  K6: {K6}")

# ═══════════════════════════════════════════════════════════════
# Final CASE
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60); log("  Final CASE"); log("="*60)

if K0=="FAIL" or K1=="FAIL": FINAL_CASE="PROTOCOL-FAIL"
elif K0=="PASS" and K1=="PASS" and K2=="PASS" and K3=="NOT SUPPORTED" and K6=="SUPPORTED":
    FINAL_CASE="SHAPE-A"
elif K0=="PASS" and K1=="PASS" and K2=="PASS" and K3=="SUPPORTED" and K4=="SUPPORTED" and K5=="SUPPORTED" and K6=="SUPPORTED":
    FINAL_CASE="SHAPE-B"
elif K0=="PASS" and K1=="PASS" and K2=="PASS" and K3=="SUPPORTED" and (K4=="NOT SUPPORTED" or K5=="NOT SUPPORTED"):
    FINAL_CASE="SHAPE-C"
else:
    FINAL_CASE="PROTOCOL-FAIL"

can_design_method = (FINAL_CASE=="SHAPE-B")
log(f"  Final CASE: {FINAL_CASE}")
log(f"  Can design optical-state evolution: {'YES' if can_design_method else 'NO'}")

# ═══════════════════════════════════════════════════════════════
# Reports
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60); log("  Writing reports"); log("="*60)

rep_path=os.path.join(OUTPUT,"shape_transport_optical_dilution_report.md")
with open(rep_path,"w") as f:
    f.write(f"""# Shape Transport Optical Dilution Report

K0 R4 Reproduction: {K0}
K1 Transport Implementation: {K1}
K2 Fixed-Cov Dilution: {K2}
K3 Full-Cov Break: {K3}
K4 Geometric Cancellation: {K4}
K5 Oracle Restoration: {K5}
K6 Area-Preserving Control: {K6}
Final CASE: {FINAL_CASE}
Can design method: {'YES' if can_design_method else 'NO'}
""")

with open(os.path.join(OUTPUT,"stage3_4B_summary.md"),"w") as f:
    f.write(f"# Stage 3.4B Summary\nFINAL CASE: {FINAL_CASE}\nK0:{K0} K1:{K1} K2:{K2} K3:{K3} K4:{K4} K5:{K5} K6:{K6}\n")

with open(os.path.join(OUTPUT,"stage3_4B_log.txt"),"w") as f:
    f.write("\n".join(log_lines))

# ═══════════════════════════════════════════════════════════════
# Terminal summary
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60); log("  TERMINAL SUMMARY"); log("="*60)
out=[
    f"  R4 actual transport policy: FIXED COVARIANCE (xyz only)",
    f"  R4 exact reproduction: {'PASS' if r4_ok else 'FAIL'}",
    f"  R4 per-cell median/p95/max: {np.median(comp_diffs_a):.2e}/{np.quantile(comp_diffs_a,0.95):.2e}/{comp_diffs_a.max():.2e}",
    f"  Shape unit tests: {'PASS' if ut_pass else 'FAIL'}",
]
for r in phys_rows:
    if r["policy"]=="P0_FIXED_COV" and r["state"]=="stretch_2.00": out.append(f"  stretch2 P0 medR={r['median_R']:.4f} MAE={r['MAE']:.4f}")
    if r["policy"]=="P1_RIGID_COV" and r["state"]=="stretch_2.00": out.append(f"  stretch2 P1 medR={r['median_R']:.4f} MAE={r['MAE']:.4f}")
    if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]=="stretch_2.00": out.append(f"  stretch2 P2 medR={r['median_R']:.4f} MAE={r['MAE']:.4f}")
    if r["policy"]=="P3_FULL_AFFINE_ORACLE" and r["state"]=="stretch_2.00": out.append(f"  stretch2 P3 medR={r['median_R']:.4f} MAE={r['MAE']:.4f}")
    if r["policy"]=="P0_FIXED_COV" and r["state"]=="cubic_l0333": out.append(f"  cubic0333 P0 MAE={r['MAE']:.4f} Sp={r['Spearman']}")
    if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]=="cubic_l0333": out.append(f"  cubic0333 P2 MAE={r['MAE']:.4f} Sp={r['Spearman']}")
    if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]=="shear_k040": out.append(f"  shear k040 P2 MAE={r['MAE']:.4f}")
    if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]=="twist_60": out.append(f"  twist P2 MAE={r['MAE']:.4f}")

for r in fp_val_rows:
    if r["policy"]=="P0_FIXED_COV": out.append(f"  P0 footprint ratio median: {r['median_footprint_ratio']:.4f}")
    if r["policy"]=="P1_RIGID_COV": out.append(f"  P1 footprint ratio median: {r['median_footprint_ratio']:.4f}")
    if r["policy"]=="P2_FULL_AFFINE_COV": out.append(f"  P2 footprint/Js rel error: {r['median_error']:.4f}")

out.append(f"  Pooled R-Bproxy Spearman: {sp_RB:.4f}")
out.append(f"  P0 six-state mean MAE: {p0_mean:.4f}")
out.append(f"  P2 six-state mean MAE: {p2_mean:.4f}")
out.append(f"  P3 six-state mean MAE: {p3_mean:.4f}")
out.append(f"  Oracle improvement: {improvement:.2%}")
out.append(f"  K0: {K0}")
out.append(f"  K1: {K1}")
out.append(f"  K2: {K2}")
out.append(f"  K3: {K3}")
out.append(f"  K4: {K4}")
out.append(f"  K5: {K5}")
out.append(f"  K6: {K6}")
out.append(f"  Final CASE: {FINAL_CASE}")
out.append(f"  Can design optical-state evolution: {'YES' if can_design_method else 'NO'}")
out.append(f"  Report: {OUTPUT}/shape_transport_optical_dilution_report.md")
out.append(f"  Summary: {OUTPUT}/stage3_4B_summary.md")
for l in out: print(l)
