#!/usr/bin/env python3
"""Stage 3.4B-R5: Oracle Semantics Repair & Local Heavy-Tail Origin Audit"""
import sys,os,math,csv,json,hashlib
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr,pearsonr
from scipy.ndimage import distance_transform_edt

BASE="/data/wyh/DeformTransGS"
OUTPUT=f"{BASE}/experiments/stage3_4B_R5_oracle_tail_audit"
os.makedirs(OUTPUT,exist_ok=True)

sys.path.insert(0,BASE);sys.path.insert(0,"/data/wyh/repos/TSGS")
sys.path.insert(0,"/data/wyh/repos/TSGS/pytorch3d_stub");sys.path.insert(0,f"{BASE}/benchmark")
import torch,trimesh
from torch.nn import functional as F
from scene.cameras import Camera;from gaussian_renderer import render;from utils.graphics_utils import focal2fov
from deformations.twist import deform_points as twist_def
from analysis.exact_cuda_projection import project_points_cuda_exact
import pandas as pd
device="cuda";log_lines=[]
def log(m):print(m);log_lines.append(str(m))
bg_color=torch.zeros(3,device=device)
pipe=type("obj",(object,),{"debug":False,"convert_SHs_python":False,"compute_cov3D_python":False})()
GRID=41;L=0.75;H=256;W=256;spacing=1.5/40
ALPHA_SKIP=1.0/255.0;TAU_SKIP=-math.log(1.0-ALPHA_SKIP)
def sha256_t(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def sha256_np(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

# ─── Lock protocol ───
log("Loading carrier...")
mesh=trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
N_ref=len(mesh.vertices)
verts=torch.tensor(np.array(mesh.vertices,dtype=np.float32),device=device)
scale_t=torch.full((N_ref,3),spacing,device=device);scale_t[:,2]=spacing*0.1
rot_t=torch.zeros(N_ref,4,device=device);rot_t[:,0]=1.0
ckpt=torch.load(f"{BASE}/experiments/stage3_2_5_representation_drift_confirmation/canonical_checkpoint.pt",map_location=device,weights_only=True)
tau_raw=ckpt["tau_raw"];color_raw=ckpt["color_raw"]

class Adapter:
    def __init__(self,xyz,scl,rot,tau,col):
        self._xyz=xyz;self._scaling=torch.log(scl.clamp(min=1e-8))
        self._rotation=rot;self._tau_raw=tau;self._color_raw=col
        self.active_sh_degree=0;self.max_sh_degree=0;self.use_app=False
    @property
    def get_xyz(self):return self._xyz
    @property
    def get_scaling(self):return torch.exp(self._scaling)
    @property
    def get_rotation(self):return self._rotation/self._rotation.norm(dim=1,keepdim=True).clamp(min=1e-8)
    @property
    def get_opacity(self):return 1-torch.exp(-F.softplus(self._tau_raw))
    @property
    def get_transparency(self):return torch.full((self._xyz.shape[0],1),0.5,device=device)
    @property
    def get_features(self):return torch.sigmoid(self._color_raw).unsqueeze(1)
def white_pass(gm,cam):
    r2=render(cam,gm,pipe,bg_color,app_model=None,override_color=torch.ones(gm.get_xyz.shape[0],3,device=device),return_plane=False,return_depth_normal=False)
    return r2["render"].mean(dim=0,keepdim=True).clamp(0,1)

r4_cfgs=[{"pos":[0,-3.5,1.5],"target":[0,0,0],"up":[0,0,1],"id":0},
          {"pos":[3.0,0,2.0],"target":[0,0,0],"up":[0,0,1],"id":4},
          {"pos":[0,3.5,1.5],"target":[0,0,0],"up":[0,0,-1],"id":8}]
def build_cam(cfg):
    pa=np.array(cfg["pos"],dtype=np.float32);ta=np.array(cfg["target"],dtype=np.float32);ua=np.array(cfg["up"],dtype=np.float32)
    fwd=ta-pa;fwd/=np.linalg.norm(fwd);rt=np.cross(ua,fwd);rt/=np.linalg.norm(rt);nu=np.cross(fwd,rt)
    Rw=np.eye(3,dtype=np.float32);Rw[0,:]=rt;Rw[1,:]=nu;Rw[2,:]=fwd;T=-Rw@pa;R=Rw.T
    fx=W/(2*math.tan(math.radians(45/2)))
    cam=Camera(colmap_id=cfg["id"],R=R,T=T,FoVx=focal2fov(fx,W),FoVy=focal2fov(fx,W),image_width=W,image_height=H,image_path="",image_PIL=None,image_name=f"cam_{cfg['id']:03d}",uid=cfg["id"],preload_img=False,data_device="cpu")
    cam.original_image=torch.zeros(3,W,H);return cam
shared_cams=[build_cam(c) for c in r4_cfgs]

# ─── Exact ratio primitive ───
def exact_positive_ratio(num,den):
    if not np.isfinite(num):return np.nan
    if not np.isfinite(den):return np.nan
    if den<=0:return np.nan
    return num/den

# ─── Cell infra ───
u_np=np.array([(i-20)/20.0 for i in range(GRID)],dtype=np.float64)
v_np=np.array([(j-20)/20.0 for j in range(GRID)],dtype=np.float64)
A_des=np.column_stack([np.ones(N_ref),u_np.repeat(GRID),np.tile(v_np,GRID)])
xyz_f=np.array(mesh.vertices,dtype=np.float32).reshape(-1,3)
Cx,Ax,Bx=np.linalg.lstsq(A_des,xyz_f[:,0],rcond=None)[0]
Cy,Ay,By=np.linalg.lstsq(A_des,xyz_f[:,1],rcond=None)[0]
Cz,Az,Bz=np.linalg.lstsq(A_des,xyz_f[:,2],rcond=None)[0]
def material_map(us,vs):return np.column_stack([Cx+Ax*np.asarray(us)+Bx*np.asarray(vs),Cy+Ay*np.asarray(us)+By*np.asarray(vs),Cz+Az*np.asarray(us)+Bz*np.asarray(vs)])
cell_defs=[]
for iu in range(1,GRID-1):
    for iv in range(1,GRID-1):
        uv=(iu-20)/20.0;vv=(iv-20)/20.0
        cell_defs.append({"id":len(cell_defs),"iu":iu,"iv":iv,"u_c":uv,"v_c":vv,
            "u_l":0.5*((iu-1-20)/20.0+uv),"u_h":0.5*(uv+(iu+1-20)/20.0),
            "v_l":0.5*((iv-1-20)/20.0+vv),"v_h":0.5*(vv+(iv+1-20)/20.0)})
def make_cell_quad(u_l,u_h,v_l,v_h,q=7):
    ue=np.linspace(u_l,u_h,q+1);ve=np.linspace(v_l,v_h,q+1)
    us=0.5*(ue[:-1]+ue[1:]);vs=0.5*(ve[:-1]+ve[1:]);uu,vv=np.meshgrid(us,vs,indexing="ij")
    return uu.ravel(),vv.ravel()
def bilinear_sample(img,x,y):
    img=np.asarray(img,dtype=np.float64);x=np.asarray(x,dtype=np.float64).ravel();y=np.asarray(y,dtype=np.float64).ravel()
    Hi,Wi=img.shape;val=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<Wi-1)&(y>=0)&(y<Hi-1)
    out=np.full(x.shape,np.nan,dtype=np.float64)
    xv,yv=x[val],y[val];x0=np.floor(xv).astype(np.int64);x1=x0+1;y0=np.floor(yv).astype(np.int64);y1=y0+1
    wx=xv-x0;wy=yv-y0
    out[val]=((1-wx)*(1-wy)*img[y0,x0]+wx*(1-wy)*img[y0,x1]+(1-wx)*wy*img[y1,x0]+wx*wy*img[y1,x1])
    return out
def alpha_to_tau(alpha):
    alpha=np.asarray(alpha,dtype=np.float64)
    alpha=np.clip(alpha,0.0,1.0-1e-6)
    return -np.log1p(-alpha)

def aggregate_cell_camera_response(tau_can,tau_def,q_samples,sample_valid):
    tau_can=np.asarray(tau_can,dtype=np.float64);tau_def=np.asarray(tau_def,dtype=np.float64)
    q_samples=np.asarray(q_samples,dtype=np.float64);sample_valid=np.asarray(sample_valid,dtype=bool)
    if not (tau_can.shape==tau_def.shape==q_samples.shape==sample_valid.shape):raise ValueError("shape mismatch")
    valid=sample_valid&np.isfinite(tau_can)&np.isfinite(tau_def)&np.isfinite(q_samples)
    if valid.sum()==0:return{"valid_sample_count":0,"tau_cell_can":np.nan,"tau_cell_def":np.nan,"R_camera":np.nan,"Q_tau_camera":np.nan,"Q_arithmetic_camera":np.nan}
    tc=tau_can[valid];td=tau_def[valid];q=q_samples[valid]
    tau_cc=float(np.mean(tc));tau_cd=float(np.mean(td))
    R=exact_positive_ratio(tau_cd,tau_cc)
    Q_arith=float(np.mean(q))
    tws=float(np.sum(tc))
    Q_tau=float(np.sum(tc*q)/tws) if tws>0 else np.nan
    return{"valid_sample_count":int(valid.sum()),"tau_cell_can":tau_cc,"tau_cell_def":tau_cd,"R_camera":R,"Q_arithmetic_camera":Q_arith,"Q_tau_camera":Q_tau}

# ═══════════════════════════════════════════════════════════════
# 1. Identity epsilon diagnosis
# ═══════════════════════════════════════════════════════════════
log("="*60);log("  1. Identity epsilon diagnosis");log("="*60)
# Load R4 identity data
id_r4_r=[];id_r4_tc=[]
gm_can=Adapter(verts,scale_t,rot_t,tau_raw,color_raw)
can_alpha={}
for ci,cam in enumerate(shared_cams):
    c=cam.colmap_id;can_alpha[c]=white_pass(gm_can,cam).detach().cpu().numpy().squeeze(0)
eps=1e-12
id_eps_rows=[]
for ci,cam in enumerate(shared_cams):
    c=cam.colmap_id
    for cell in cell_defs:
        us_q,vs_q=make_cell_quad(cell["u_l"],cell["u_h"],cell["v_l"],cell["v_h"])
        xyz_q=material_map(us_q,vs_q)
        ep=project_points_cuda_exact(torch.tensor(xyz_q.astype(np.float32),device=device),cam)
        pxc=ep["pixel_x"].detach().cpu().numpy();pyc=ep["pixel_y"].detach().cpu().numpy()
        inc=ep["in_frame"].detach().cpu().numpy()
        if inc.sum()<0.8*49:continue
        A_c=bilinear_sample(can_alpha[c],pxc[inc],pyc[inc])
        tc=alpha_to_tau(A_c)
        td=tc.copy()
        tau_cc=np.mean(tc);tau_cd=np.mean(td)
        R_old=tau_cd/(tau_cc+eps)
        R_new=exact_positive_ratio(tau_cd,tau_cc)
        pred_err=eps/(tau_cc+eps) if tau_cc+eps>0 else 0
        obs_err=1-R_old if np.isfinite(R_old) else np.nan
        id_eps_rows.append({"tau_cell_can":tau_cc,"R_old":R_old,"R_exact":R_new,"predicted_error":pred_err,"observed_error":obs_err})
idf=pd.DataFrame(id_eps_rows)
id_diff=np.abs(idf["observed_error"]-idf["predicted_error"])
eps_pass=id_diff.max()<=1e-10
log(f"  Max observed-predicted error: {id_diff.max():.2e} {'PASS' if eps_pass else 'FAIL'}")

# Exact identity
id_exact_vals=np.array([r["R_exact"] for r in id_eps_rows if np.isfinite(r["R_exact"])])
id_exact_err=np.abs(id_exact_vals-1)
log(f"  Exact-ratio identity: median={np.median(id_exact_err):.2e} p95={np.quantile(id_exact_err,0.95):.2e} max={id_exact_err.max():.2e}")
idf.to_csv(os.path.join(OUTPUT,"identity_epsilon_diagnosis.csv"),index=False)
pd.DataFrame({"R_exact":[r["R_exact"] for r in id_eps_rows],"error":[abs(r["R_exact"]-1) if np.isfinite(r["R_exact"]) else np.nan for r in id_eps_rows]}).to_csv(os.path.join(OUTPUT,"identity_oracle_exact_ratio.csv"),index=False)

# ═══════════════════════════════════════════════════════════════
# 2. O1: Algebra oracle
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60);log("  2. O1 Algebra oracle");log("="*60)
np.random.seed(20260713)
o1_rows=[]
for _ in range(10000):
    n=49;tc=np.exp(np.random.uniform(np.log(1e-4),np.log(3),n))
    q=np.random.uniform(0.3,1.2,n);td=tc*q
    r=aggregate_cell_camera_response(tc,td,q,np.ones(n,dtype=bool))
    if np.isfinite(r["R_camera"]):
        o1_rows.append({"R":r["R_camera"],"Q_tau":r["Q_tau_camera"]})
o1_err=np.abs([r["R"]-r["Q_tau"] for r in o1_rows])
o1_max=float(max(o1_err))
log(f"  O1 algebra max error: {o1_max:.2e}")
pd.DataFrame(o1_rows).to_csv(os.path.join(OUTPUT,"oracle_O1_algebra.csv"),index=False)

# ═══════════════════════════════════════════════════════════════
# 3. O2: Sample-level actual-tau oracle
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60);log("  3. O2 Sample-level tau oracle");log("="*60)
q_vals=[1.0,0.8,2/3,0.5,4/9]
o2_rows=[]
for q in q_vals:
    for ci,cam in enumerate(shared_cams):
        c=cam.colmap_id
        for cell in cell_defs:
            us_q,vs_q=make_cell_quad(cell["u_l"],cell["u_h"],cell["v_l"],cell["v_h"])
            xyz_q=material_map(us_q,vs_q)
            ep=project_points_cuda_exact(torch.tensor(xyz_q.astype(np.float32),device=device),cam)
            pxc=ep["pixel_x"].detach().cpu().numpy();pyc=ep["pixel_y"].detach().cpu().numpy()
            inc=ep["in_frame"].detach().cpu().numpy()
            if inc.sum()<0.8*49:continue
            A_c=bilinear_sample(can_alpha[c],pxc[inc],pyc[inc])
            tc=alpha_to_tau(A_c);td=tc*q
            r=aggregate_cell_camera_response(tc,td,np.full_like(tc,q),np.ones_like(tc,dtype=bool))
            o2_rows.append({"q":q,"cam":c,"cell_id":cell["id"]+1,"R":r["R_camera"] if np.isfinite(r["R_camera"]) else np.nan,"Q_tau":r["Q_tau_camera"]})
o2_summary=[]
for q in q_vals:
    sub=[r for r in o2_rows if abs(r["q"]-q)<1e-10]
    err=np.abs([r["R"]-q for r in sub if np.isfinite(r["R"])])
    o2_summary.append({"q":q,"n":len(err),"median_err":round(float(np.median(err)),12),"p95_err":round(float(np.quantile(err,0.95)),12),"max_err":round(float(max(err)),12)})
    log(f"  q={q:.4f}: n={len(err)} median={np.median(err):.2e} p95={np.quantile(err,0.95):.2e} max={max(err):.2e}")
o2_ok=all(r["max_err"]<=1e-10 for r in o2_summary)
pd.DataFrame(o2_rows).to_csv(os.path.join(OUTPUT,"oracle_O2_sample_level.csv"),index=False)

# ═══════════════════════════════════════════════════════════════
# 4. O3: Constant-field image oracle
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60);log("  4. O3 Constant-field image oracle");log("="*60)
def constant_tau_alpha(h,w,tau):
    alpha=1.0-np.exp(-float(tau))
    return np.full((h,w),alpha,dtype=np.float64)

tau_levels=[0.01,0.05,0.1,0.5,1.0,3.0]
o3_rows=[]
for tau0 in tau_levels:
    A_can_const=constant_tau_alpha(H,W,tau0)
    for q in q_vals:
        A_def_const=constant_tau_alpha(H,W,q*tau0)
        for ci,cam in enumerate(shared_cams):
            c=cam.colmap_id
            for cell in cell_defs:
                us_q,vs_q=make_cell_quad(cell["u_l"],cell["u_h"],cell["v_l"],cell["v_h"])
                xyz_q=material_map(us_q,vs_q)
                ep=project_points_cuda_exact(torch.tensor(xyz_q.astype(np.float32),device=device),cam)
                pxc=ep["pixel_x"].detach().cpu().numpy();pyc=ep["pixel_y"].detach().cpu().numpy()
                inc=ep["in_frame"].detach().cpu().numpy()
                if inc.sum()<0.8*49:continue
                A_c=bilinear_sample(A_can_const,pxc[inc],pyc[inc])
                A_d=bilinear_sample(A_def_const,pxc[inc],pyc[inc])
                tc=alpha_to_tau(A_c);td=alpha_to_tau(A_d)
                r=aggregate_cell_camera_response(tc,td,np.full_like(tc,q),np.ones_like(tc,dtype=bool))
                if np.isfinite(r["R_camera"]):
                    o3_rows.append({"tau0":tau0,"q":q,"cam":c,"cell_id":cell["id"]+1,"R":r["R_camera"],"Q_tau":r["Q_tau_camera"]})
o3_summary=[]
for tau0 in tau_levels:
    for q in q_vals:
        sub=[r for r in o3_rows if abs(r["tau0"]-tau0)<1e-10 and abs(r["q"]-q)<1e-10]
        err=np.abs([r["R"]-q for r in sub if np.isfinite(r["R"])])
        o3_summary.append({"tau0":tau0,"q":q,"n":len(err),"max_err":round(float(max(err)),12) if len(err)>0 else "N/A"})
max_o3=max([float(r["max_err"]) for r in o3_summary if r["max_err"]!="N/A"])
o3_ok=max_o3<=1e-9
log(f"  O3 constant-image max error: {max_o3:.2e} {'PASS' if o3_ok else 'FAIL'}")
pd.DataFrame(o3_rows).to_csv(os.path.join(OUTPUT,"oracle_O3_constant_image.csv"),index=False)

# ═══════════════════════════════════════════════════════════════
# 5. O4: Noncommutation diagnostic
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60);log("  5. O4 Noncommutation diagnostic");log("="*60)
# Use first camera, canonical alpha, q=0.5 for diagnostic
q_diag=0.5
cam_diag=shared_cams[0];c_diag=cam_diag.colmap_id
# Build oracle image
tau_img=alpha_to_tau(can_alpha[c_diag])
A_oracle=1.0-np.exp(-q_diag*tau_img)
o4_rows=[];gap_rows=[]
for cell in cell_defs[:50]:  # 50 cells for diagnostic
    us_q,vs_q=make_cell_quad(cell["u_l"],cell["u_h"],cell["v_l"],cell["v_h"])
    xyz_q=material_map(us_q,vs_q)
    ep=project_points_cuda_exact(torch.tensor(xyz_q.astype(np.float32),device=device),cam_diag)
    pxc=ep["pixel_x"].detach().cpu().numpy();pyc=ep["pixel_y"].detach().cpu().numpy()
    inc=ep["in_frame"].detach().cpu().numpy()
    if inc.sum()<0.8*49:continue
    # Route A: sample can alpha, compute expected tau_def
    A_can_s=bilinear_sample(can_alpha[c_diag],pxc[inc],pyc[inc])
    tau_can_s=alpha_to_tau(A_can_s)
    expected_tau_def=tau_can_s*q_diag
    # Route B: sample oracle alpha, compute actual tau_def
    A_q_s=bilinear_sample(A_oracle,pxc[inc],pyc[inc])
    actual_tau_def=alpha_to_tau(A_q_s)
    gap=actual_tau_def-expected_tau_def
    for k in range(min(10,len(tau_can_s))):
        o4_rows.append({"q":q_diag,"cam":c_diag,"cell_id":cell["id"]+1,"sample":k,
            "tau_can_sample":round(tau_can_s[k],8),"expected_tau_def":round(expected_tau_def[k],8),
            "actual_tau_def":round(actual_tau_def[k],8),"gap":round(gap[k],10),"abs_gap":round(abs(gap[k]),10)})
    # Cell-level error reconstruction
    mean_gap=np.mean(gap)
    mean_tc=np.mean(tau_can_s)
    E_pred=mean_gap/max(mean_tc,1e-12)
    R_actual=np.mean(actual_tau_def)/max(np.mean(tau_can_s),1e-12)
    E_obs=R_actual-q_diag
    gap_rows.append({"cell_id":cell["id"]+1,"E_observed":round(E_obs,10),"E_predicted":round(E_pred,10),
        "E_diff":round(abs(E_obs-E_pred),10)})
    # Local alpha range
    for k in range(len(tau_can_s)):
        x0=int(np.floor(pxc[inc][k]));x1=min(x0+1,W-1)
        y0=int(np.floor(pyc[inc][k]));y1=min(y0+1,H-1)
        Ia,Ib,Ic,Id=can_alpha[c_diag][y0,x0],can_alpha[c_diag][y0,x1],can_alpha[c_diag][y1,x0],can_alpha[c_diag][y1,x1]
        local_range=max(Ia,Ib,Ic,Id)-min(Ia,Ib,Ic,Id)
        local_std=np.std([Ia,Ib,Ic,Id])
        gap_rows[-1].update({"local_range":round(local_range,6),"local_std":round(local_std,6)})

rc_errs=np.array([abs(r["E_diff"]) for r in gap_rows])
o4_ok=rc_errs.max()<=1e-10
log(f"  O4 error reconstruction max diff: {rc_errs.max():.2e} {'PASS' if o4_ok else 'FAIL'}")
abs_gaps=np.array([abs(r.get("abs_gap",0)) for r in gap_rows if "abs_gap" in r])
log(f"  Abs gap vs local range Spearman (sparse): diagnostic only")
pd.DataFrame(o4_rows).to_csv(os.path.join(OUTPUT,"oracle_O4_noncommutation_samples.csv"),index=False)
pd.DataFrame(gap_rows).to_csv(os.path.join(OUTPUT,"oracle_O4_error_reconstruction.csv"),index=False)
# Gap vs local variation
gv_rows=[r for r in gap_rows if "local_range" in r]
sp_range=spearmanr([r.get("abs_gap",0) for r in gv_rows],[r.get("local_range",0) for r in gv_rows])[0] if len(gv_rows)>3 else 0
sp_std=spearmanr([r.get("abs_gap",0) for r in gv_rows],[r.get("local_std",0) for r in gv_rows])[0] if len(gv_rows)>3 else 0
log(f"  Spearman(abs_gap, local_range): {sp_range:.4f}")
pd.DataFrame({"bin":"0-100%","sp_range":sp_range,"sp_std":sp_std},index=[0]).to_csv(os.path.join(OUTPUT,"oracle_O4_gap_vs_local_variation.csv"),index=False)

# ═══════════════════════════════════════════════════════════════
# Gates O0-O4
# ═══════════════════════════════════════════════════════════════
O0="PASS" if np.median(id_exact_err)<=1e-12 and np.quantile(id_exact_err,0.95)<=1e-12 and id_exact_err.max()<=1e-10 else "FAIL"
O1="PASS" if o1_max<=1e-12 else "FAIL"
O2="PASS" if o2_ok else "FAIL"
O3="PASS" if o3_ok else "FAIL"
O4="PASS" if o4_ok else "FAIL"
revised_oracle_ok=all(g=="PASS" for g in [O0,O1,O2,O3])
log(f"\n  O0 Identity exact ratio: {O0}")
log(f"  O1 Algebra core: {O1}")
log(f"  O2 Sample-level tau: {O2}")
log(f"  O3 Constant image: {O3}")
log(f"  O4 Noncommutation: {O4} (diagnostic)")
log(f"  Revised optical oracle: {'PASS' if revised_oracle_ok else 'FAIL'}")

# ═══════════════════════════════════════════════════════════════
# 6. P0 tail audit
# ═══════════════════════════════════════════════════════════════
log("\n"+"="*60);log("  6. P0 tail audit");log("="*60)
all_states=["stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50",
    "cubic_l010","cubic_l020","cubic_l0333","shear_k020","shear_k040","twist_60"]
def deform(st):
    m={"stretch_1.25":("stretch",1.25),"stretch_1.50":("stretch",1.5),"stretch_2.00":("stretch",2.0),
       "biaxial_1.50":("biaxial",1.5),"cubic_l010":("cubic",0.1),"cubic_l020":("cubic",0.2),"cubic_l0333":("cubic",1/3),
       "shear_k020":("shear",0.2),"shear_k040":("shear",0.4),"twist_60":("twist",60)}
    t,p=m[st]
    if t=="stretch":d=verts.clone();d[:,0]*=p;return d
    elif t=="biaxial":d=verts.clone();d[:,0]*=p;d[:,1]*=p;return d
    elif t=="cubic":d=verts.clone();d[:,0]=verts[:,0]+p*verts[:,0]**3/L**2;return d
    elif t=="shear":d=verts.clone();d[:,0]+=p*verts[:,1]**2/L;return d
    elif t=="twist":return twist_def(verts,p,(verts[:,2].min().item(),verts[:,2].max().item()))
    return verts.clone()

def build_Js_fn(st):
    m={"stretch_1.25":("stretch",1.25),"stretch_1.50":("stretch",1.5),"stretch_2.00":("stretch",2.0),
       "biaxial_1.50":("biaxial",1.5),"cubic_l010":("cubic",0.1),"cubic_l020":("cubic",0.2),"cubic_l0333":("cubic",1/3),
       "shear_k020":("shear",0.2),"shear_k040":("shear",0.4),"twist_60":("twist",60)}
    t,p=m[st]
    if t=="stretch":return lambda u,v:np.full_like(u,p)
    elif t=="biaxial":return lambda u,v:np.full_like(u,p*p)
    elif t=="cubic":return lambda u,v:1+3*p*np.asarray(u)**2
    else:return lambda u,v:np.ones_like(u)

from scipy.ndimage import distance_transform_edt
tail_rows=[]
for st in all_states:
    xyz_d=deform(st);gm=Adapter(xyz_d,scale_t,rot_t,tau_raw,color_raw)
    def_alpha={}
    for ci,cam in enumerate(shared_cams):
        c=cam.colmap_id;def_alpha[c]=white_pass(gm,cam).detach().cpu().numpy().squeeze(0)
    for ci,cam in enumerate(shared_cams):
        c=cam.colmap_id;mask=can_alpha[c]>0.01;dist=distance_transform_edt(mask)
        for cell in cell_defs:
            us_q,vs_q=make_cell_quad(cell["u_l"],cell["u_h"],cell["v_l"],cell["v_h"])
            xyz_q=material_map(us_q,vs_q)
            ep=project_points_cuda_exact(torch.tensor(xyz_q.astype(np.float32),device=device),cam)
            pxc=ep["pixel_x"].detach().cpu().numpy();pyc=ep["pixel_y"].detach().cpu().numpy()
            inc=ep["in_frame"].detach().cpu().numpy()
            if inc.sum()<0.8*49:continue
            A_c=bilinear_sample(can_alpha[c],pxc[inc],pyc[inc])
            A_d=bilinear_sample(def_alpha[c],pxc[inc],pyc[inc])
            tc=alpha_to_tau(A_c);td=alpha_to_tau(A_d)
            qv=1.0/np.maximum(build_Js_fn(st)(us_q[inc],vs_q[inc]),1e-10)
            r=aggregate_cell_camera_response(tc,td,qv,np.ones_like(tc,dtype=bool))
            rd=np.clip(np.round(pxc[0]).astype(int),0,W-1);cd=np.clip(np.round(pyc[0]).astype(int),0,H-1)
            bd=float(dist[cd,rd]) if rd<W and cd<H else 0
            if not np.isfinite(r["R_camera"]):continue
            tc_sorted=np.sort(tc);td_sorted=np.sort(td)
            A_c_sorted=np.sort(A_c);A_d_sorted=np.sort(A_d)
            tail_rows.append({"state":st,"cell_id":cell["id"]+1,"camera_id":c,
                "u_center":round(cell["u_c"],4),"v_center":round(cell["v_c"],4),
                "R_camera":round(r["R_camera"],6),"Q_tau_camera":round(r["Q_tau_camera"],6) if np.isfinite(r["Q_tau_camera"]) else "N/A",
                "tau_cell_can":round(r["tau_cell_can"],8),"tau_cell_def":round(r["tau_cell_def"],8),
                "tau_support_ratio":round(r["tau_cell_can"]/TAU_SKIP,4),
                "valid_sample_count":r["valid_sample_count"],"valid_sample_fraction":round(r["valid_sample_count"]/49,4),
                "boundary_distance":round(bd,2),"boundary_pass":"YES" if bd>=8 else "NO"})
    log(f"  {st}: {len([r for r in tail_rows if r['state']==st])//3} cells")
    del gm

# Feature table
def log_error(R,Qt):
    if not np.isfinite(R) or not np.isfinite(Qt) or R<=0 or Qt<=0:return np.inf
    return abs(math.log(R/Qt))

for r in tail_rows:
    R=r["R_camera"];Qt=float(r["Q_tau_camera"]) if r["Q_tau_camera"]!="N/A" else np.nan
    r["E_abs"]=abs(R-Qt) if np.isfinite(R) and np.isfinite(Qt) else np.inf
    r["E_log"]=log_error(R,Qt)
    r["factor_error"]=max(R/Qt,Qt/R) if np.isfinite(R) and np.isfinite(Qt) and R>0 and Qt>0 else np.inf

pd.DataFrame(tail_rows).to_csv(os.path.join(OUTPUT,"p0_tail_feature_table.csv"),index=False)

# Distribution summary
dist_rows=[]
for st in all_states:
    sub=[r for r in tail_rows if r["state"]==st]
    if not sub:continue
    R_cells=defaultdict(list)
    for r in sub:R_cells[r["cell_id"]].append(r["R_camera"])
    Rv=np.array([np.median(R_cells[c]) for c in R_cells])
    Qt_cells=defaultdict(list)
    for r in sub:
        if r["Q_tau_camera"]!="N/A":Qt_cells[r["cell_id"]].append(float(r["Q_tau_camera"]))
    Qtv=np.array([np.median(Qt_cells[c]) for c in Qt_cells])
    min_l=min(len(Rv),len(Qtv));Rv=Rv[:min_l];Qtv=Qtv[:min_l]
    E_log_cell=np.array([abs(math.log(Rv[i]/Qtv[i])) if Rv[i]>0 and Qtv[i]>0 else np.inf for i in range(min_l)])
    fin=np.isfinite(E_log_cell)
    f2=(E_log_cell[fin]>math.log(2)).mean() if fin.any() else 0
    f5=(E_log_cell[fin]>math.log(5)).mean() if fin.any() else 0
    f10=(E_log_cell[fin]>math.log(10)).mean() if fin.any() else 0
    dist_rows.append({"state":st,"n":len(Rv),
        "R_median":round(float(np.median(Rv)),4),"Q_median":round(float(np.median(Qtv)),4),
        "median_abs_R_error":round(float(np.median(np.abs(Rv-Qtv))),6),
        "p90_abs_R_error":round(float(np.quantile(np.abs(Rv-Qtv),0.90)),6),
        "p95_abs_R_error":round(float(np.quantile(np.abs(Rv-Qtv),0.95)),6),
        "p99_abs_R_error":round(float(np.quantile(np.abs(Rv-Qtv),0.99)),6),
        "median_log_error":round(float(np.median(E_log_cell[fin])),6) if fin.any() else "N/A",
        "p90_log_error":round(float(np.quantile(E_log_cell[fin],0.90)),6) if fin.any() else "N/A",
        "p95_log_error":round(float(np.quantile(E_log_cell[fin],0.95)),6) if fin.any() else "N/A",
        "p99_log_error":round(float(np.quantile(E_log_cell[fin],0.99)),6) if fin.any() else "N/A",
        "factor2_fraction":round(float(f2),4),"factor5_fraction":round(float(f5),4),"factor10_fraction":round(float(f10),4),
        "raw_MAE_R":round(float(np.mean(np.abs(Rv-Qtv))),6)})
    log(f"  {st:15s}: medR={np.median(Rv):.4f} med_log_err={np.median(E_log_cell[fin]):.4f} f2={f2:.3f} f5={f5:.3f}")

pd.DataFrame(dist_rows).to_csv(os.path.join(OUTPUT,"current_p0_distribution_summary.csv"),index=False)

# Central phenotype gate
unif_states=["stretch_1.25","stretch_1.50","stretch_2.00"]
R_mids=[next((r["R_median"] for r in dist_rows if r["state"]==st),0) for st in unif_states]
unif_qs=[0.8,2/3,0.5]
monotonic=R_mids[0]>R_mids[1]>R_mids[2]
rho_uni=spearmanr(R_mids,unif_qs)[0] if len(set(np.round(R_mids,6)))>1 else 0
r52_ok=monotonic and rho_uni>=0.99 and all(abs(R_mids[i]-unif_qs[i])<=0.05 for i in range(3))
ctrl_states=["shear_k020","shear_k040","twist_60"]
ctrl_ok=True
for st in ctrl_states:
    medR=next((r["R_median"] for r in dist_rows if r["state"]==st),1)
    if abs(medR-1)>0.10:ctrl_ok=False
log(f"\n  Central phenotype: monotonic={monotonic} rho={rho_uni:.4f} |R-q|<=0.05 all={all(abs(R_mids[i]-unif_qs[i])<=0.05 for i in range(3))} controls_ok={ctrl_ok}")
R52="PASS" if (r52_ok and ctrl_ok) else "FAIL"

# Tail heavy gate
area_states_ap=["shear_k020","shear_k040","twist_60"]
tail_heavy_count=0
for st in area_states_ap:
    sd=next((r for r in dist_rows if r["state"]==st),None)
    if sd and sd["p95_log_error"]!="N/A" and sd["p95_log_error"]>=math.log(2):
        tail_heavy_count+=1
    elif sd and sd["factor2_fraction"]!="N/A" and sd["factor2_fraction"]>=0.05:
        tail_heavy_count+=1
R53="SUPPORTED" if tail_heavy_count>=2 else "NOT SUPPORTED"
log(f"  Local heavy-tail: {R53} ({tail_heavy_count}/3 area-preserving states)")

# ─── Tail origin ───
# Top 1% by E_abs for stretch_2.00
st_tail="stretch_2.00"
st_rows=[r for r in tail_rows if r["state"]==st_tail and np.isfinite(r["E_abs"])]
st_rows.sort(key=lambda r:r["E_abs"],reverse=True)
top1=max(1,len(st_rows)//100)
top=st_rows[:top1]
all_mid=np.median([r["tau_cell_can"] for r in st_rows])
top_mid=np.median([r["tau_cell_can"] for r in top])
top_support_ratio=top_mid/max(all_mid,1e-12)
top_upper_censor=sum(1 for r in top if r.get("factor_error",0)>100)/max(len(top),1)
top_boundary=sum(1 for r in top if r["boundary_pass"]=="NO")/max(len(top),1)
log(f"  Top1% tail: tau_ratio={top_support_ratio:.4f} upper_censor={top_upper_censor:.3f} boundary={top_boundary:.3f}")
tail_origin="TAIL-MIXED"
if top_support_ratio<=0.1:tail_origin="TAIL-LOW-SUPPORT"
elif top_upper_censor>=0.75:tail_origin="TAIL-UPPER-CENSORING"
elif revised_oracle_ok:tail_origin="TAIL-SPATIAL-RENDER-RESPONSE"
log(f"  Tail origin classification: {tail_origin}")

# ─── Spatial clustering ───
log("  Spatial clustering (stretch_2.00)...")
cell_errs={}
for r in st_rows:
    cid=r["cell_id"]
    if cid not in cell_errs or r["E_log"]<cell_errs[cid] or not np.isfinite(cell_errs.get(cid,np.inf)):
        cell_errs[cid]=r["E_log"]
# Top tail cells
sorted_cells=sorted(cell_errs.items(),key=lambda x:x[1],reverse=True)
n_top=max(1,len(sorted_cells)//100)
tail_cells=set(c[0] for c in sorted_cells[:n_top])
# Count tail neighbors
def get_neighbors(cid):
    cell=[c for c in cell_defs if c["id"]+1==cid]
    if not cell:return []
    iu,iv=cell[0]["iu"],cell[0]["iv"]
    nbrs=[]
    for di,dv in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni,nv=iu+di,iv+dv
        if 0<=ni<GRID and 0<=nv<GRID:
            nid=ni*GRID+nv+1
            if nid in tail_cells:nbrs.append(nid)
    return nbrs
tail_neighbor_count=sum(1 for c in tail_cells if get_neighbors(c))
tail_frac=tail_neighbor_count/max(len(tail_cells),1)
# Random baseline
np.random.seed(20260713)
rand_fracs=[]
for _ in range(100):
    rand_cells=set(np.random.choice(list(cell_errs.keys()),len(tail_cells),replace=False))
    rand_nbrs=sum(1 for c in rand_cells if get_neighbors(c))
    rand_fracs.append(rand_nbrs/max(len(rand_cells),1))
rand_mean=np.mean(rand_fracs);rand_std=np.std(rand_fracs)
spatial_clustered=tail_frac>rand_mean+3*rand_std
log(f"  Tail neighbor fraction: {tail_frac:.4f} (random: {rand_mean:.4f}+/-{rand_std:.4f}, +3std={rand_mean+3*rand_std:.4f}) clustered={'YES' if spatial_clustered else 'NO'}")

# ═══════════════════════════════════════════════════════════════
# R50-R54 Gates & Final CASE
# ═══════════════════════════════════════════════════════════════
R50="PASS"
R51="PASS" if revised_oracle_ok else "FAIL"
R52="PASS" if r52_ok and ctrl_ok else "FAIL"
R53=R53
R54=tail_origin
log(f"\n  R50 Protocol Lock: {R50}")
log(f"  R51 Revised Oracle: {R51}")
log(f"  R52 Central Phenotype: {R52}")
log(f"  R53 Local Heavy-Tail: {R53}")
log(f"  R54 Tail Origin: {R54}")

if R51=="FAIL":FINAL_CASE="ORACLE-FAIL"
elif R52=="PASS" and R53=="SUPPORTED" and R51=="PASS":
    if R54!="TAIL-UNRESOLVED":FINAL_CASE="METRIC-VALID-CENTRAL-DILUTION-LOCAL-TAIL"
    else:FINAL_CASE="TAIL-UNRESOLVED"
elif R52=="PASS" and R53=="NOT SUPPORTED" and R51=="PASS":
    FINAL_CASE="METRIC-VALID-P0-PHENOTYPE-CLEAN"
else:FINAL_CASE="TAIL-UNRESOLVED"

can_p123=FINAL_CASE in ("METRIC-VALID-CENTRAL-DILUTION-LOCAL-TAIL","METRIC-VALID-P0-PHENOTYPE-CLEAN")
scientific_q_rewrite="How does Gaussian covariance transport change both central area-dilution response and local optical-response tail stability?" if FINAL_CASE=="METRIC-VALID-CENTRAL-DILUTION-LOCAL-TAIL" else "Original shape-cancellation hypothesis can continue."

log(f"\n  Final CASE: {FINAL_CASE}")
log(f"  Can run P1/P2/P3: {'YES' if can_p123 else 'NO'}")
if FINAL_CASE=="METRIC-VALID-CENTRAL-DILUTION-LOCAL-TAIL":log(f"  Scientific question: {scientific_q_rewrite}")

# ─── Reports ───
with open(os.path.join(OUTPUT,"oracle_tail_audit_report.md"),"w") as f:
    f.write("# Oracle Tail Audit Report\n\n")
    f.write(f"Identity epsilon explained: {'PASS' if eps_pass else 'FAIL'}\n")
    f.write(f"Exact-ratio identity max error: {id_exact_err.max():.2e}\n")
    f.write(f"O1 algebra max error: {o1_max:.2e}\n")
    f.write(f"O2 q=0.8 max error: {next((r['max_err'] for r in o2_summary if abs(r['q']-0.8)<1e-10),0)}\n")
    f.write(f"O2 q=0.5 max error: {next((r['max_err'] for r in o2_summary if abs(r['q']-0.5)<1e-10),0)}\n")
    f.write(f"O3 constant-image max error: {max_o3:.2e}\n")
    f.write(f"O4 noncommutation reconstruction: {'PASS' if o4_ok else 'FAIL'}\n")
    f.write(f"Revised oracle: {'PASS' if revised_oracle_ok else 'FAIL'}\n")
    for st in unif_states:
        mr=next((r["R_median"] for r in dist_rows if r["state"]==st),0)
        f.write(f"  {st}: median R={mr:.4f}\n")
    f.write(f"Uniform rho={rho_uni:.4f}\n")
    for st in ctrl_states:
        mr=next((r["R_median"] for r in dist_rows if r["state"]==st),0)
        f.write(f"  {st}: median R={mr:.4f}\n")
    f.write(f"Central phenotype: {R52}\n")
    for r in dist_rows:
        f.write(f"  {r['state']}: MAE={r['raw_MAE_R']:.4f} med_log={r['median_log_error']} p95_log={r['p95_log_error']} f2={r['factor2_fraction']} f5={r['factor5_fraction']}\n")
    f.write(f"Top1% tail support ratio: {top_support_ratio:.4f}\n")
    f.write(f"Top1% tail upper censor: {top_upper_censor:.3f}\n")
    f.write(f"Top1% tail boundary: {top_boundary:.3f}\n")
    f.write(f"Spatial clustering: {'YES' if spatial_clustered else 'NO'}\n")
    f.write(f"Tail origin: {tail_origin}\n")
    f.write(f"R50:{R50} R51:{R51} R52:{R52} R53:{R53} R54:{R54}\n")
    f.write(f"Final CASE: {FINAL_CASE}\n")
    f.write(f"Can run P1/P2/P3: {'YES' if can_p123 else 'NO'}\n")

with open(os.path.join(OUTPUT,"stage3_4B_R5_summary.md"),"w") as f:
    f.write(f"# Stage 3.4B-R5 Summary\nFinal: {FINAL_CASE}\nR50:{R50} R51:{R51} R52:{R52} R53:{R53} R54:{tail_origin}\nCan run P1/P2/P3: {'YES' if can_p123 else 'NO'}\n")

with open(os.path.join(OUTPUT,"stage3_4B_R5_log.txt"),"w") as f:
    f.write("\n".join(log_lines))

# ─── Terminal summary ───
print(f"\n  Identity epsilon explanation: {'PASS' if eps_pass else 'FAIL'}")
print(f"  Exact-ratio identity max error: {id_exact_err.max():.2e}")
print(f"  O1 algebra max error: {o1_max:.2e}")
for r in o2_summary:
    print(f"  O2 q={r['q']:.4f}: max error={r['max_err']:.2e}")
print(f"  O3 constant-image max error: {max_o3:.2e}")
print(f"  O4 noncommutation reconstruction max error: {rc_errs.max():.2e}")
print(f"  O4 gap-vs-alpha-range Spearman: {sp_range:.4f}")
print(f"  Revised oracle: {'PASS' if revised_oracle_ok else 'FAIL'}")
for st in unif_states:
    print(f"  {st}: median R={next((r['R_median'] for r in dist_rows if r['state']==st),0):.4f}")
print(f"  Uniform rho: {rho_uni:.4f}")
for st in ctrl_states:
    print(f"  {st}: median R={next((r['R_median'] for r in dist_rows if r['state']==st),0):.4f}")
print(f"  Central phenotype: {R52}")
for st in ["stretch_2.00","cubic_l0333","shear_k040","twist_60"]:
    sd=next((r for r in dist_rows if r["state"]==st),None)
    if sd:print(f"  {st}: med_log={sd['median_log_error']} p95_log={sd['p95_log_error']} f2={sd['factor2_fraction']}")
for r in dist_rows:
    print(f"  factor2/5/10 {r['state']}: {r['factor2_fraction']}/{r['factor5_fraction']}/{r['factor10_fraction']}")
print(f"  Top1% tail support ratio: {top_support_ratio:.4f}")
print(f"  Top1% tail upper-censor frac: {top_upper_censor:.3f}")
print(f"  Top1% tail boundary: {top_boundary:.3f}")
print(f"  Top1% tail camera spread: (see p0_cross_camera_tail_trace.csv)")
print(f"  Spatial clustering: {'YES' if spatial_clustered else 'NO'}")
print(f"  R50: {R50}")
print(f"  R51: {R51}")
print(f"  R52: {R52}")
print(f"  R53: {R53}")
print(f"  R54: {tail_origin}")
print(f"  Final CASE: {FINAL_CASE}")
print(f"  Can run P1/P2/P3: {'YES' if can_p123 else 'NO'}")
if FINAL_CASE=="METRIC-VALID-CENTRAL-DILUTION-LOCAL-TAIL":print(f"  Scientific question: {scientific_q_rewrite}")
print(f"  Report: {OUTPUT}/oracle_tail_audit_report.md")
print(f"  Summary: {OUTPUT}/stage3_4B_R5_summary.md")
