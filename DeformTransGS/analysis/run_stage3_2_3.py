#!/usr/bin/env python3
"""Stage 3.2.3: Implicit optical transport mechanism validation"""
import sys, os, csv, math, hashlib, json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import spearmanr

BASE = "/data/wyh/DeformTransGS"
OUTPUT = f"{BASE}/experiments/stage3_2_3_implicit_transport_mechanism"
os.makedirs(OUTPUT, exist_ok=True)

sys.path.insert(0, "/data/wyh/repos/TSGS")
sys.path.insert(0, "/data/wyh/repos/TSGS/pytorch3d_stub")
sys.path.insert(0, f"{BASE}/benchmark")

import torch, trimesh
from torch.nn import functional as F
from scene.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import focal2fov
from deformations.twist import deform_points as twist_def

device = "cuda"
log_lines = []
def log(m): print(m); log_lines.append(str(m))
def hdr(t): log("="*60); log(f"  {t}"); log("="*60)

# ═══════════════════════════════════════════════════════════
# 0. Setup
# ═══════════════════════════════════════════════════════════
mesh = trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
N = len(mesh.vertices)
verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
spacing = 1.5/40
scale_p = torch.full((N,3), spacing, device=device); scale_p[:,2] = spacing*0.1
rot_init = torch.zeros(N,4,device=device); rot_init[:,0]=1.0

camera_cfgs = [{"pos":[0,-3.5,1.5],"target":[0,0,0],"up":[0,0,1],"id":0},
               {"pos":[3.0,0,2.0],"target":[0,0,0],"up":[0,0,1],"id":4},
               {"pos":[0,3.5,1.5],"target":[0,0,0],"up":[0,0,-1],"id":8}]

def build_cam(cfg):
    pa=np.array(cfg["pos"],dtype=np.float32); ta=np.array(cfg["target"],dtype=np.float32); ua=np.array(cfg["up"],dtype=np.float32)
    fwd=(ta-pa)/np.linalg.norm(ta-pa); rt=np.cross(ua,fwd); rt/=np.linalg.norm(rt); nu=np.cross(fwd,rt)
    Rw=np.eye(3,dtype=np.float32); Rw[0,:]=rt; Rw[1,:]=nu; Rw[2,:]=fwd
    T=-Rw@pa; R=Rw.T
    fx=256/(2*math.tan(math.radians(45/2)))
    cam=Camera(colmap_id=cfg["id"],R=R,T=T,FoVx=focal2fov(fx,256),FoVy=focal2fov(fx,256),image_width=256,image_height=256,image_path="",image_PIL=None,image_name=f"cam_{cfg['id']:03d}",uid=cfg["id"],preload_img=False,data_device="cpu")
    cam.original_image=torch.zeros(3,256,256); return cam

film_cams=[build_cam(c) for c in camera_cfgs]
bg_color=torch.zeros(3,device=device)
pipe=type('obj',(object,),{"debug":False,"convert_SHs_python":False,"compute_cov3D_python":False})()

class Adapter:
    def __init__(self,xyz,scale,rot,tau_raw,color_raw):
        self._xyz=xyz; self._scaling=torch.log(scale.clamp(min=1e-8))
        self._rotation=rot; self._tau_raw=tau_raw; self._color_raw=color_raw
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
    def get_transparency(self): return torch.full((N,1),0.5,device=self._xyz.device)
    @property
    def get_features(self): return torch.sigmoid(self._color_raw).unsqueeze(1)

def white_pass(adapter,cam):
    r2=render(cam,adapter,pipe,bg_color,app_model=None,override_color=torch.ones_like(torch.sigmoid(adapter._color_raw)),return_plane=False,return_depth_normal=False)
    return r2["render"].mean(dim=0,keepdim=True).clamp(0,1), r2["radii"]

# Load canonical checkpoint (from Stage 3.2 full training)
try:
    ckpt=torch.load(f"{BASE}/experiments/stage3_2_fixed_optical_necessity/canonical_checkpoint.pt")
    can_tau_raw=ckpt["tau_raw"].to(device)
    can_color_raw=ckpt["color_raw"].to(device)
    log("Loaded canonical checkpoint")
except:
    log("No checkpoint found, training fresh...")
    tr=torch.full((N,1),0.0,device=device,requires_grad=True)
    cr=torch.zeros(N,3,device=device,requires_grad=True)
    opt=torch.optim.Adam([tr,cr],lr=1e-2)
    GT_DYN=f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/repaired_render/tau1.0_dynamic"
    for it in range(3000):
        opt.zero_grad(); loss=0
        for ci,cam in enumerate(film_cams):
            cid=[0,4,8][ci]
            adpt=Adapter(verts,scale_p,rot_init,tr,cr)
            r2=render(cam,adpt,pipe,bg_color,app_model=None,override_color=torch.sigmoid(cr),return_plane=False,return_depth_normal=False)
            C=r2["render"]; white=torch.ones_like(torch.sigmoid(cr))
            rw=render(cam,adpt,pipe,bg_color,app_model=None,override_color=white,return_plane=False,return_depth_normal=False)
            A=rw["render"].mean(dim=0,keepdim=True).clamp(0,1)
            bg=torch.tensor(np.array(Image.open(f"{BASE}/experiments/stage3_2_fixed_optical_necessity/background_only/cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0,device=device).permute(2,0,1)
            pred=(C+(1-A)*bg).clamp(0,1)
            gt=torch.tensor(np.array(Image.open(f"{GT_DYN}/canonical_cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0,device=device).permute(2,0,1)
            loss+=(pred-gt).abs().mean()+0.2*(1-((2*pred*gt+0.01)/(pred**2+gt**2+0.01)).mean())
        loss.backward(); opt.step()
        if it%1000==0: log(f"  iter {it}: loss={loss.item():.6f}")
    can_tau_raw=tr.detach().clone(); can_color_raw=cr.detach().clone()
    torch.save({"tau_raw":can_tau_raw,"color_raw":can_color_raw},f"{OUTPUT}/canonical_checkpoint.pt")

# ═══════════════════════════════════════════════════════════
# 1. Tau_eff formula correction
# ═══════════════════════════════════════════════════════════
hdr("1. Tau_eff correction & unit test")
log("tau_eff = -log(1 - A)")
for A_test in [0, 0.1, 0.5, 0.9]:
    te = -math.log(1 - max(A_test, 1e-10))
    log(f"  A={A_test:.1f} -> tau_eff={te:.6f}")

# ═══════════════════════════════════════════════════════════
# 2. Geometry variants
# ═══════════════════════════════════════════════════════════
hdr("2. Geometry variant integrity")
verts_t = torch.tensor(np.array(mesh.vertices), dtype=torch.float32, device=device)
z_range = (verts_t[:,2].min().item(), verts_t[:,2].max().item())

def get_state(name):
    if name=="canonical": return verts, scale_p, rot_init
    s=float(name.split("_")[1])
    if name.startswith("stretch"):
        dv=verts_t.clone(); dv[:,0]*=s; return dv, scale_p, rot_init
    elif name.startswith("biaxial"):
        dv=verts_t.clone(); dv[:,0]*=s; dv[:,1]*=s; return dv, scale_p*s, rot_init
    elif name.startswith("twist"):
        dv=twist_def(verts_t, int(name.split("_")[1]), z_range); return dv, scale_p, rot_init

def get_Js(name):
    if name=="canonical": return 1.0
    s=float(name.split("_")[1])
    return s if name.startswith("stretch") else s*s if name.startswith("biaxial") else 1.0

ablate_states = ["stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","twist_60"]
geo_rows = []
for st in ablate_states:
    dv,sc,rt = get_state(st)
    # G1: position only
    G1 = Adapter(dv, scale_p, rot_init, can_tau_raw, can_color_raw)
    # G2: covariance only
    G2 = Adapter(verts, sc, rot_init, can_tau_raw, can_color_raw)
    # G3: full
    G3 = Adapter(dv, sc, rot_init, can_tau_raw, can_color_raw)
    G0 = Adapter(verts, scale_p, rot_init, can_tau_raw, can_color_raw)
    
    for label,gm in [("G0_canonical",G0),("G1_position",G1),("G2_covariance",G2),("G3_full",G3)]:
        sha_xyz = hashlib.sha256(gm._xyz.detach().cpu().numpy().tobytes()).hexdigest()[:8]
        sha_sc = hashlib.sha256(gm._scaling.detach().cpu().numpy().tobytes()).hexdigest()[:8]
        geo_rows.append({"state":st,"variant":label,"xyz_sha":sha_xyz,"scale_sha":sha_sc})
        A_list=[]; radii_list=[]
        for ci,cam in enumerate(film_cams):
            cid=[0,4,8][ci]
            A,radii = white_pass(gm, cam)
            diff = (torch.tensor(np.array(Image.open(f"{BASE}/experiments/stage3_2_fixed_optical_necessity/background_only/cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0,device=device).permute(2,0,1).abs().mean(dim=0)>0.01).float()
            mask = binary_dilation(diff.cpu().numpy(), iterations=2)
            interior = binary_erosion(mask, iterations=5)
            if interior.sum()>0:
                A_list.append(A[0,interior].mean().item())
                radii_list.append(radii[radii>0].float().mean().item() if (radii>0).sum()>0 else 0)
        if A_list:
            geo_rows[-1].update({"A_mean":np.mean(A_list),"tau_eff":-math.log(1-max(np.mean(A_list),1e-10)),"radii_mean":np.mean(radii_list)})
    
    log(f"  {st}: G0_A={[r['A_mean'] for r in geo_rows if r['state']==st and r['variant']=='G0_canonical']}")

# ═══════════════════════════════════════════════════════════
# 3. Surface density
# ═══════════════════════════════════════════════════════════
hdr("3. Surface density")
density_rows = []
for st in ablate_states+["canonical"]:
    dv,_,_ = get_state(st)
    Js = get_Js(st)
    # Surface area via mesh triangles
    if st == "canonical":
        area0 = mesh.area
    else:
        m2 = trimesh.Trimesh(vertices=dv.cpu().numpy(), faces=mesh.faces, process=False)
        area_s = m2.area
    r = (mesh.area/area_s) if st != "canonical" else 1.0
    density_rows.append({"state":st,"Js":Js,"density_ratio":r,"expected_1_over_Js":1.0/max(Js,1e-8) if Js>0 else 1})
    if st != "canonical":
        log(f"  {st:20s}: density_ratio={r:.4f}  1/Js={1/Js:.4f}")

# ═══════════════════════════════════════════════════════════
# 4. Alpha / tau_eff for full state set
# ═══════════════════════════════════════════════════════════
hdr("4. Alpha/tau_eff across all states")
states_all = ["canonical"] + [f"stretch_{s:.2f}" for s in [1.10,1.25,1.50,2.00]] + \
    [f"biaxial_{s:.2f}" for s in [1.10,1.25,1.50]] + ["twist_30","twist_60"]
alpha_rows = []
for st in states_all:
    dv,sc,rt = get_state(st)
    Js = get_Js(st)
    gm = Adapter(dv,sc,rt,can_tau_raw,can_color_raw)
    A_vals=[]
    for ci,cam in enumerate(film_cams):
        cid=[0,4,8][ci]
        A,_ = white_pass(gm, cam)
        diff = (torch.tensor(np.array(Image.open(f"{BASE}/experiments/stage3_2_fixed_optical_necessity/background_only/cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0,device=device).permute(2,0,1).abs().mean(dim=0)>0.01).float()
        interior=binary_erosion(binary_dilation(diff.cpu().numpy(),iterations=2),iterations=5)
        if interior.sum()>0: A_vals.append(A[0,interior].mean().item())
    A_m=np.mean(A_vals) if A_vals else 0
    te=-math.log(1-max(A_m,1e-10)) if A_m<1 else 10
    alpha_rows.append({"state":st,"Js":Js,"A_interior":A_m,"tau_eff":te})
    if st!="canonical":
        log(f"  {st:20s}: Js={Js:.3f} A={A_m:.4f} tau_eff={te:.4f} ratio={te/alpha_rows[0]['tau_eff']:.4f}")

# ═══════════════════════════════════════════════════════════
# 5. Sampling resolution ablation
# ═══════════════════════════════════════════════════════════
hdr("5. Sampling resolution ablation")
resolution_rows=[]
W,H=1.5,1.5
for res_name, divs in [("21x21",20),("41x41",40),("81x81",80)]:
    N_r=(divs+1)**2
    verts_r=torch.tensor([[-W/2+W*i/divs,-H/2+H*j/divs,0.0] for i in range(divs+1) for j in range(divs+1)],dtype=torch.float32,device=device)
    sp=W/divs
    sc_r=torch.full((N_r,3),sp,device=device); sc_r[:,2]=sp*0.1
    ro_r=torch.zeros(N_r,4,device=device); ro_r[:,0]=1.0
    tau_r=torch.full((N_r,1),2.0,device=device,requires_grad=True)  # fixed global tau
    
    for st in ["canonical","stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","twist_60"]:
        if st=="canonical": dv_r=verts_r
        else:
            s=float(st.split("_")[1])
            dv_r=verts_r.clone()
            if st.startswith("stretch"): dv_r[:,0]*=s
            elif st.startswith("biaxial"): dv_r[:,0]*=s; dv_r[:,1]*=s
        
        # Use fixed tau with global calibration
        class AdapterR:
            def __init__(self,xyz):
                self._xyz=xyz; self._scaling=torch.log(sc_r.clamp(min=1e-8))
                self._rotation=ro_r; self._tau_raw=tau_r; self._color_raw=torch.zeros(N_r,3,device=device)
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
            def get_transparency(self): return torch.full((N_r,1),0.5,device=device)
            @property
            def get_features(self): return torch.sigmoid(self._color_raw).unsqueeze(1)
        
        gm_r=AdapterR(dv_r); A_vals_r=[]
        for ci,cam in enumerate(film_cams):
            cid=[0,4,8][ci]; A,_=white_pass(gm_r,cam)
            diff=(torch.tensor(np.array(Image.open(f"{BASE}/experiments/stage3_2_fixed_optical_necessity/background_only/cam{cid:03d}.png").convert("RGB")).astype(np.float32)/255.0,device=device).permute(2,0,1).abs().mean(dim=0)>0.01).float()
            interior=binary_erosion(binary_dilation(diff.cpu().numpy(),iterations=2),iterations=5)
            if interior.sum()>0: A_vals_r.append(A[0,interior].mean().item())
        A_m_r=np.mean(A_vals_r) if A_vals_r else 0
        te_r=-math.log(1-max(A_m_r,1e-10))
        resolution_rows.append({"resolution":res_name,"state":st,"A":A_m_r,"tau_eff":te_r})
    
    log(f"  {res_name}: canonical A={[r['A'] for r in resolution_rows if r['resolution']==res_name and r['state']=='canonical']}")

# ═══════════════════════════════════════════════════════════
# 6. Mechanism Gate
# ═══════════════════════════════════════════════════════════
hdr("6. Mechanism Gate")
canon_te = alpha_rows[0]["tau_eff"]
uniaxial_rates = []
for st in [f"stretch_{s:.2f}" for s in [1.10,1.25,1.50,2.00]]:
    r = next(x for x in alpha_rows if x["state"]==st)
    uniaxial_rates.append(r["tau_eff"]/canon_te)
js_vals = [get_Js(st) for st in [f"stretch_{s:.2f}" for s in [1.10,1.25,1.50,2.00]]]
rho,_ = spearmanr(js_vals, uniaxial_rates) if len(set(uniaxial_rates))>1 else (0,1)
log(f"  Uniaxial tau_eff ratio vs Js Spearman: {rho:.4f}")

# Physical consistency
for st in ["stretch_1.50","stretch_2.00","biaxial_1.50"]:
    r = next(x for x in alpha_rows if x["state"]==st)
    Js = get_Js(st)
    ratio = r["tau_eff"]/canon_te
    log(f"  {st:20s}: ratio={ratio:.4f}  1/Js={1/Js:.4f}  abs_err={abs(ratio-1/Js):.4f}")

t60 = next(x for x in alpha_rows if x["state"]=="twist_60")
t60_ratio = t60["tau_eff"]/canon_te
log(f"  twist_60: ratio={t60_ratio:.4f} (should be ~1)")

gate = "SUPPORTED" if rho >= 0.9 else "PARTIAL"
phys = "SUPPORTED" if all(abs(r["tau_eff"]/canon_te - 1/get_Js(r["state"])) <= 0.10 for r in alpha_rows if r["state"] in ["stretch_1.50","stretch_2.00","biaxial_1.50"]) else "PARTIAL"
log(f"  Implicit Transport Gate: {gate}")
log(f"  Physical Consistency: {phys}")

log("\n=== Stage 3.2.3 complete ===")
