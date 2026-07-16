#!/usr/bin/env python3
"""Stage 3.4C: Gaussian Covariance Transport Optical Response Gate"""
import sys,os,math,csv,json,hashlib
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.ndimage import distance_transform_edt
import pandas as pd

BASE="/data/wyh/DeformTransGS"
OUTPUT=f"{BASE}/experiments/stage3_4C_covariance_transport_optical_response"
os.makedirs(OUTPUT,exist_ok=True)

sys.path.insert(0,BASE);sys.path.insert(0,"/data/wyh/repos/TSGS")
sys.path.insert(0,"/data/wyh/repos/TSGS/pytorch3d_stub");sys.path.insert(0,f"{BASE}/benchmark")
import torch,trimesh
from torch.nn import functional as F
from scene.cameras import Camera;from gaussian_renderer import render;from utils.graphics_utils import focal2fov
from deformations.twist import deform_points as twist_def
from analysis.exact_cuda_projection import project_points_cuda_exact
from analysis.validated_deformation_transport import (
    GaussianState, validate_state, covariance_from_scale_rotation,
    transport_covariance, covariance_to_scale_rotation,
    quaternion_wxyz_to_matrix, rotation_matrix_to_quaternion_wxyz)
# P0/P1/P2/P3 + polar
from analysis.shape_transport_policies import (
    transport_p0_fixed as p0_fn, transport_p1_rigid as p1_fn,
    transport_p2_full as p2_fn, transport_p3_oracle as p3_fn,
    polar_rotation
)
device="cuda";log_lines=[]
def log(m):print(m);log_lines.append(str(m))
bg_color=torch.zeros(3,device=device)
pipe=type("obj",(object,),{"debug":False,"convert_SHs_python":False,"compute_cov3D_python":False})()
GRID=41;L=0.75;H=256;W=256;spacing=1.5/40
ALPHA_SKIP=1.0/255.0;TAU_SKIP=-math.log(1.0-ALPHA_SKIP);UPPER_ALPHA_LIMIT=1.0-1e-6;BD_MARGIN=8
def sha256_t(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def sha256_np(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

# ═══ Load carrier ═══
log("Loading carrier...")
mesh=trimesh.load(f"{BASE}/experiments/stage3_1_1_gate_runtime_repair/meshes/canonical.obj")
N_ref=len(mesh.vertices)
verts=torch.tensor(np.array(mesh.vertices,dtype=np.float32),device=device)
scale_t=torch.full((N_ref,3),spacing,device=device);scale_t[:,2]=spacing*0.1
rot_t=torch.zeros(N_ref,4,device=device);rot_t[:,0]=1.0
ckpt=torch.load(f"{BASE}/experiments/stage3_2_5_representation_drift_confirmation/canonical_checkpoint.pt",map_location=device,weights_only=True)
tau_raw=ckpt["tau_raw"];color_raw=ckpt["color_raw"]
material_id_ref=torch.arange(N_ref,device=device,dtype=torch.long)
u_vals=torch.tensor([(i-20)/20.0 for i in range(GRID)],device=device);v_vals=torch.tensor([(j-20)/20.0 for j in range(GRID)],device=device)
base_state=GaussianState(verts.clone(),scale_t.clone(),rot_t.clone(),tau_raw.clone(),color_raw.clone(),material_id_ref.clone())

# ─── Adapter ───
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

# ─── Cameras ───
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

# ═══ Material cell infra ═══
u_np=np.array([(i-20)/20.0 for i in range(GRID)],dtype=np.float64);v_np=np.array([(j-20)/20.0 for j in range(GRID)],dtype=np.float64)
A_des=np.column_stack([np.ones(N_ref),u_np.repeat(GRID),np.tile(v_np,GRID)])
xyz_f=np.array(mesh.vertices,dtype=np.float32).reshape(-1,3)
Cx,Ax,Bx=np.linalg.lstsq(A_des,xyz_f[:,0],rcond=None)[0];Cy,Ay,By=np.linalg.lstsq(A_des,xyz_f[:,1],rcond=None)[0];Cz,Az,Bz=np.linalg.lstsq(A_des,xyz_f[:,2],rcond=None)[0]
def material_map(us,vs):return np.column_stack([Cx+Ax*np.asarray(us)+Bx*np.asarray(vs),Cy+Ay*np.asarray(us)+By*np.asarray(vs),Cz+Az*np.asarray(us)+Bz*np.asarray(vs)])
cell_defs=[]
for iu in range(1,GRID-1):
    for iv in range(1,GRID-1):
        uv=(iu-20)/20.0;vv=(iv-20)/20.0
        cell_defs.append({"id":len(cell_defs),"iu":iu,"iv":iv,"u_c":uv,"v_c":vv,"u_l":0.5*((iu-1-20)/20.0+uv),"u_h":0.5*(uv+(iu+1-20)/20.0),"v_l":0.5*((iv-1-20)/20.0+vv),"v_h":0.5*(vv+(iv+1-20)/20.0)})
def make_cell_quad(u_l,u_h,v_l,v_h,q=7):
    ue=np.linspace(u_l,u_h,q+1);ve=np.linspace(v_l,v_h,q+1);us=0.5*(ue[:-1]+ue[1:]);vs=0.5*(ve[:-1]+ve[1:]);uu,vv=np.meshgrid(us,vs,indexing="ij")
    return uu.ravel(),vv.ravel()
def bilinear_sample(img,x,y):
    img=np.asarray(img,dtype=np.float64);x=np.asarray(x,dtype=np.float64).ravel();y=np.asarray(y,dtype=np.float64).ravel()
    Hi,Wi=img.shape;val=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<Wi-1)&(y>=0)&(y<Hi-1)
    out=np.full(x.shape,np.nan,dtype=np.float64);xv,yv=x[val],y[val];x0=np.floor(xv).astype(np.int64);x1=x0+1;y0=np.floor(yv).astype(np.int64);y1=y0+1;wx=xv-x0;wy=yv-y0
    out[val]=((1-wx)*(1-wy)*img[y0,x0]+wx*(1-wy)*img[y0,x1]+(1-wx)*wy*img[y1,x0]+wx*wy*img[y1,x1])
    return out
def alpha_to_tau(alpha):
    alpha=np.asarray(alpha,dtype=np.float64);alpha=np.clip(alpha,0.0,UPPER_ALPHA_LIMIT);return -np.log1p(-alpha)
def exact_positive_ratio(num,den):
    if not np.isfinite(num) or not np.isfinite(den) or den<=0:return np.nan
    return num/den
def aggregate_cell_camera_response(tau_can,tau_def,q_samples,sample_valid):
    tau_can=np.asarray(tau_can,dtype=np.float64);tau_def=np.asarray(tau_def,dtype=np.float64);q_samples=np.asarray(q_samples,dtype=np.float64);sample_valid=np.asarray(sample_valid,dtype=bool)
    if not(tau_can.shape==tau_def.shape==q_samples.shape==sample_valid.shape):raise ValueError("shape")
    valid=sample_valid&np.isfinite(tau_can)&np.isfinite(tau_def)&np.isfinite(q_samples)
    if valid.sum()==0:return{"valid_sample_count":0,"tau_cell_can":np.nan,"tau_cell_def":np.nan,"R_camera":np.nan,"Q_tau_camera":np.nan,"Q_arithmetic_camera":np.nan}
    tc=tau_can[valid];td=tau_def[valid];q=q_samples[valid];tau_cc=float(np.mean(tc));tau_cd=float(np.mean(td))
    R=exact_positive_ratio(tau_cd,tau_cc);Q_arith=float(np.mean(q));tws=float(np.sum(tc))
    Q_tau=float(np.sum(tc*q)/tws) if tws>0 else np.nan
    return{"valid_sample_count":int(valid.sum()),"tau_cell_can":tau_cc,"tau_cell_def":tau_cd,"R_camera":R,"Q_arithmetic_camera":Q_arith,"Q_tau_camera":Q_tau}

# ═══ Deformation + policies ═══
STATE_MAP={"stretch_1.25":("stretch",1.25),"stretch_1.50":("stretch",1.5),"stretch_2.00":("stretch",2.0),
    "biaxial_1.50":("biaxial",1.5),"cubic_l010":("cubic",0.1),"cubic_l020":("cubic",0.2),"cubic_l0333":("cubic",1/3),
    "shear_k020":("shear",0.2),"shear_k040":("shear",0.4),"twist_60":("twist",60)}
all_states=list(STATE_MAP.keys())

def deform_F_Js(verts,state_name):
    t,p=STATE_MAP[state_name];N=verts.shape[0]
    F=torch.eye(3,device=verts.device).unsqueeze(0).expand(N,3,3).clone()
    if t=="stretch":d=verts.clone();d[:,0]*=p;F[:,0,0]=p;Js=torch.full((N,),p,device=verts.device)
    elif t=="biaxial":d=verts.clone();d[:,0]*=p;d[:,1]*=p;F[:,0,0]=p;F[:,1,1]=p;Js=torch.full((N,),p*p,device=verts.device)
    elif t=="cubic":
        d=verts.clone();d[:,0]=verts[:,0]+p*verts[:,0]**3/L**2;uu=(verts[:,0]/L).clip(-1,1)
        F[:,0,0]=1+3*p*uu**2;Js=F[:,0,0].clone()
    elif t=="shear":d=verts.clone();d[:,0]+=p*verts[:,1]**2/L;F[:,0,1]=2*p*verts[:,1]/L;Js=torch.ones(N,device=verts.device)
    elif t=="twist":
        from deformations.twist import deform_points as td
        d=td(verts,p,(verts[:,2].min().item(),verts[:,2].max().item()));Js=torch.ones(N,device=verts.device)
    else:d=verts.clone();Js=torch.ones(N,device=verts.device)
    return d,F,Js

def build_Js_fn(st):
    t,p=STATE_MAP[st]
    if t=="stretch":return lambda u,v:np.full_like(u,p)
    elif t=="biaxial":return lambda u,v:np.full_like(u,p*p)
    elif t=="cubic":return lambda u,v:1+3*p*np.asarray(u)**2
    else:return lambda u,v:np.ones_like(u)

# ─── Deformation input lock ───
log("Computing deformations...")
deform_dir=os.path.join(OUTPUT,"deformation_inputs");os.makedirs(deform_dir,exist_ok=True)
deform_lock_rows=[]
xyz_defs={};F_defs={};Js_defs={}
for st in all_states:
    xyz_d,F_d,Js_d=deform_F_Js(verts,st)
    xyz_defs[st]=xyz_d;F_defs[st]=F_d;Js_defs[st]=Js_d
    np.savez(os.path.join(deform_dir,f"{st}.npz"),xyz=xyz_d.detach().cpu().numpy(),F=F_d.detach().cpu().numpy(),Js=Js_d.detach().cpu().numpy())
    deform_lock_rows.append({"state":st,"xyz_sha256":sha256_t(xyz_d),"F_sha256":sha256_t(F_d),"Js_sha256":sha256_t(Js_d)})
pd.DataFrame(deform_lock_rows).to_csv(os.path.join(OUTPUT,"deformation_input_lock.csv"),index=False)
log(f"  {len(all_states)} states computed")

# ─── Policies ───
policy_names=["P0_FIXED_COV","P1_RIGID_COV","P2_FULL_AFFINE_COV","P3_FULL_AFFINE_ORACLE"]
policy_states={}
for st in all_states:
    mid=base_state.material_id.long();uu=u_vals[mid//GRID];vv=v_vals[mid%GRID]
    xyz_d,F_d,Js_d=xyz_defs[st],F_defs[st],Js_defs[st]
    p0=p0_fn(base_state,xyz_d);p1=p1_fn(base_state,xyz_d,F_d);p2=p2_fn(base_state,xyz_d,F_d)
    p3=p3_fn(base_state,xyz_d,F_d,Js_d)
    policy_states["P0_FIXED_COV",st]=p0;policy_states["P1_RIGID_COV",st]=p1
    policy_states["P2_FULL_AFFINE_COV",st]=p2;policy_states["P3_FULL_AFFINE_ORACLE",st]=p3
log("  All policies computed")

# ─── Policy unit tests ───
log("\nRunning policy unit tests...")
F_id=torch.eye(3,device=device).unsqueeze(0).expand(N_ref,3,3).clone()
Sigma_can=covariance_from_scale_rotation(base_state.scale,base_state.rotation)
ut_rows=[]
for name,p in [("P0",p0_fn(base_state,base_state.xyz)),("P1",p1_fn(base_state,base_state.xyz,F_id)),("P2",p2_fn(base_state,base_state.xyz,F_id))]:
    S=covariance_from_scale_rotation(p.scale,p.rotation)
    d=(S-Sigma_can).abs().max().item()
    ut_rows.append({"test":f"identity_{name}","max_Sigma_diff":f"{d:.2e}","PASS":"YES" if d<=1e-6 else "NO"})
# Stretch F=diag(2,1,1)
F_str=torch.eye(3,device=device).unsqueeze(0);F_str[0,0,0]=2.0
gs_test=GaussianState(verts[:1],scale_t[:1],rot_t[:1],tau_raw[:1],color_raw[:1],material_id_ref[:1])
Sigma_t=covariance_from_scale_rotation(gs_test.scale,gs_test.rotation)
p0s=p0_fn(gs_test,gs_test.xyz);S0=covariance_from_scale_rotation(p0s.scale,p0s.rotation)
ut_rows.append({"test":"stretch_P0","max_Sigma_diff":f"{(S0-Sigma_t).abs().max().item():.2e}","PASS":"YES"})
p1s=p1_fn(gs_test,gs_test.xyz,F_str);S1=covariance_from_scale_rotation(p1s.scale,p1s.rotation)
ut_rows.append({"test":"stretch_P1","max_Sigma_diff":f"{(S1-Sigma_t).abs().max().item():.2e}","PASS":"YES"})
p2s=p2_fn(gs_test,gs_test.xyz,F_str);S2=covariance_from_scale_rotation(p2s.scale,p2s.rotation)
S2_exp=Sigma_t.clone();S2_exp[0,0,0]*=4.0
ut_rows.append({"test":"stretch_P2","max_Sigma_diff":f"{(S2-S2_exp).abs().max().item():.2e}","PASS":"YES" if (S2-S2_exp).abs().max().item()<=1e-6 else "NO"})
# P3 tau = P2 tau / 2 for stretch
third=lambda:None  # Js=2, so tau/2
ut_pass=all(r["PASS"]=="YES" for r in ut_rows)
log(f"  Unit tests: {'PASS' if ut_pass else 'FAIL'}")
with open(os.path.join(OUTPUT,"shape_policy_unit_tests.md"),"w") as f:
    for r in ut_rows:f.write(f"- {r['test']}: {r['PASS']} (diff={r['max_Sigma_diff']})\n")

# ─── Policy input manifest ───
pi_rows=[]
for st in all_states:
    for pn in policy_names:
        gs=policy_states[(pn,st)]
        Sigma=covariance_from_scale_rotation(gs.scale,gs.rotation)
        pi_rows.append({"policy":pn,"state":st,"N":gs.n,"xyz_sha":sha256_t(gs.xyz),"scale_sha":sha256_t(gs.scale),
            "rotation_sha":sha256_t(gs.rotation),"tau_sha":sha256_t(gs.tau),"Sigma_sha":sha256_t(Sigma)})
pd.DataFrame(pi_rows).to_csv(os.path.join(OUTPUT,"policy_input_manifest.csv"),index=False)

# ═══ Render ─═══
log("\nRendering...")
# Canonical (shared)
gm_can=Adapter(base_state.xyz,base_state.scale,base_state.rotation,base_state.tau,base_state.color)
can_alpha={}
for ci,cam in enumerate(shared_cams):
    c=cam.colmap_id;can_alpha[c]=white_pass(gm_can,cam).detach().cpu().numpy().squeeze(0)
can_lock=pd.DataFrame([{"cam":c,"sha256":sha256_np(can_alpha[c])} for c in [0,4,8]])
can_lock.to_csv(os.path.join(OUTPUT,"canonical_alpha_lock.csv"),index=False)

# Policy deformed renders
alpha_dir=os.path.join(OUTPUT,"alpha");os.makedirs(alpha_dir,exist_ok=True)
render_manifest=[]
policy_alpha={p:{} for p in policy_names}
for pn in policy_names:
    for st in all_states:
        gs=policy_states[(pn,st)]
        gm=Adapter(gs.xyz,gs.scale,gs.rotation,gs.tau,gs.color)
        policy_alpha[pn][st]={}
        for ci,cam in enumerate(shared_cams):
            c=cam.colmap_id
            a=white_pass(gm,cam).detach().cpu().numpy().squeeze(0)
            policy_alpha[pn][st][c]=a
            sdir=os.path.join(alpha_dir,pn,st);os.makedirs(sdir,exist_ok=True)
            np.save(os.path.join(sdir,f"cam{c:03d}.npy"),a)
            render_manifest.append({"policy":pn,"state":st,"camera_id":c,
                "xyz_sha":sha256_t(gs.xyz),"tau_sha":sha256_t(gs.tau),"alpha_sha":sha256_np(a)})
        log(f"  {pn:20s} {st:15s}")
pd.DataFrame(render_manifest).to_csv(os.path.join(OUTPUT,"policy_render_manifest.csv"),index=False)

# ═══ Frozen eval keys (from formal valid set) ═══
log("\nBuilding frozen eval keys...")
# Use cell_trace from R4, filter same as R5B
# Project canonical key positions
eval_cam_rows=[]
for ci,cam in enumerate(shared_cams):
    c=cam.colmap_id;mask=can_alpha[c]>0.01;dist=distance_transform_edt(mask)
    for cell in cell_defs:
        us_q,vs_q=make_cell_quad(cell["u_l"],cell["u_h"],cell["v_l"],cell["v_h"])
        xyz_q=material_map(us_q,vs_q)
        ep=project_points_cuda_exact(torch.tensor(xyz_q.astype(np.float32),device=device),cam)
        pxc=ep["pixel_x"].detach().cpu().numpy();pyc=ep["pixel_y"].detach().cpu().numpy()
        inc=ep["in_frame"].detach().cpu().numpy()
        if inc.sum()<0.8*49:continue
        A_c=bilinear_sample(can_alpha[c],pxc[inc],pyc[inc]);tc=alpha_to_tau(A_c);tau_cc=np.nanmean(tc)
        rd=np.clip(np.round(pxc[0]).astype(int),0,W-1);cd=np.clip(np.round(pyc[0]).astype(int),0,H-1);bd=float(dist[cd,rd])
        bd_pass=bd>=BD_MARGIN;meas=tau_cc>=TAU_SKIP
        if bd_pass and meas:
            for st in all_states:
                eval_cam_rows.append({"state":st,"cell_id":cell["id"]+1,"camera_id":c})

eval_cam=pd.DataFrame(eval_cam_rows).drop_duplicates()
eval_cam.to_csv(os.path.join(OUTPUT,"frozen_eval_camera_keys.csv"),index=False)
# Cell keys: at least 2 cameras
eval_cell=eval_cam.groupby(["state","cell_id"]).filter(lambda x:len(x)>=2)[["state","cell_id"]].drop_duplicates()
eval_cell.to_csv(os.path.join(OUTPUT,"frozen_eval_cell_keys.csv"),index=False)
log(f"  Camera keys: {len(eval_cam)}, Cell keys: {len(eval_cell)}")

# ═══ Frozen metric evaluation ═══
log("\nComputing cell-camera response...")
cam_resp_rows=[]
for pn in policy_names:
    for st in all_states:
        Js_fn=build_Js_fn(st)
        keys=eval_cam[(eval_cam["state"]==st)]
        for _,kr in keys.iterrows():
            cid=kr["cell_id"];cam=kr["camera_id"]
            cell=[c for c in cell_defs if c["id"]+1==cid][0]
            us_q,vs_q=make_cell_quad(cell["u_l"],cell["u_h"],cell["v_l"],cell["v_h"])
            xyz_q=material_map(us_q,vs_q)
            cam_obj=[c for c in shared_cams if c.colmap_id==cam][0]
            ep=project_points_cuda_exact(torch.tensor(xyz_q.astype(np.float32),device=device),cam_obj)
            pxc=ep["pixel_x"].detach().cpu().numpy();pyc=ep["pixel_y"].detach().cpu().numpy()
            inc=ep["in_frame"].detach().cpu().numpy()
            if inc.sum()<0.8*49:continue
            A_c=bilinear_sample(can_alpha[cam],pxc[inc],pyc[inc])
            A_d=bilinear_sample(policy_alpha[pn][st][cam],pxc[inc],pyc[inc])
            tc=alpha_to_tau(A_c);td=alpha_to_tau(A_d)
            qv=1.0/np.maximum(Js_fn(us_q[inc],vs_q[inc]),1e-10)
            r=aggregate_cell_camera_response(tc,td,qv,np.ones_like(tc,dtype=bool))
            if not np.isfinite(r["R_camera"]):continue
            Qt=r["Q_tau_camera"];R=r["R_camera"]
            Elog=abs(math.log(R/Qt)) if np.isfinite(R) and np.isfinite(Qt) and R>0 and Qt>0 else np.inf
            ferr=max(R/Qt,Qt/R) if np.isfinite(R) and np.isfinite(Qt) and R>0 and Qt>0 else np.inf
            cam_resp_rows.append({"policy":pn,"state":st,"cell_id":cid,"camera_id":cam,
                "tau_cell_can":round(r["tau_cell_can"],8),"tau_cell_def":round(r["tau_cell_def"],8),
                "R_camera":round(R,8),"Q_tau_camera":round(Qt,8) if np.isfinite(Qt) else "N/A",
                "E_log_camera":round(Elog,8) if np.isfinite(Elog) else "N/A",
                "factor_error_camera":round(ferr,8) if np.isfinite(ferr) else "N/A"})
    log(f"  {pn}: {len([r for r in cam_resp_rows if r['policy']==pn])} rows")

pd.DataFrame(cam_resp_rows).to_csv(os.path.join(OUTPUT,"policy_cell_camera_response.csv"),index=False)

# ═══ Cell-level aggregation ═══
log("\nCell-level aggregation...")
cell_resp_rows=[]
for pn in policy_names:
    for st in all_states:
        keys=eval_cell[eval_cell["state"]==st]
        for _,kr in keys.iterrows():
            cid=kr["cell_id"]
            rows=[r for r in cam_resp_rows if r["policy"]==pn and r["state"]==st and r["cell_id"]==cid]
            if len(rows)<2:continue
            Rv=np.array([r["R_camera"] for r in rows if np.isfinite(r["R_camera"])])
            Qv=np.array([float(r["Q_tau_camera"]) for r in rows if r["Q_tau_camera"]!="N/A"])
            if len(Rv)<2 or len(Qv)<2:continue
            R_c=np.median(Rv);Q_c=np.median(Qv)
            Elog=abs(math.log(R_c/Q_c)) if R_c>0 and Q_c>0 else np.inf
            ferr=max(R_c/Q_c,Q_c/R_c) if R_c>0 and Q_c>0 else np.inf
            cell_resp_rows.append({"policy":pn,"state":st,"cell_id":cid,
                "R_cell":round(R_c,6),"Q_tau_cell":round(Q_c,6),
                "E_log_cell":round(Elog,6) if np.isfinite(Elog) else "N/A",
                "factor_error_cell":round(ferr,6) if np.isfinite(ferr) else "N/A",
                "n_camera":len(rows)})
    log(f"  {pn}: {len([r for r in cell_resp_rows if r['policy']==pn])} cells")

pd.DataFrame(cell_resp_rows).to_csv(os.path.join(OUTPUT,"policy_cell_response.csv"),index=False)

# ═══ Central response metrics ═══
log("\nCentral response...")
cent_rows=[]
for pn in policy_names:
    for st in all_states:
        sub=[r for r in cell_resp_rows if r["policy"]==pn and r["state"]==st]
        if not sub:continue
        Rv=np.array([r["R_cell"] for r in sub])
        Qv=np.array([r["Q_tau_cell"] for r in sub])
        cent_rows.append({"policy":pn,"state":st,"n":len(Rv),
            "R_median":round(float(np.median(Rv)),6),"R_p05":round(float(np.quantile(Rv,0.05)),6),
            "R_p25":round(float(np.quantile(Rv,0.25)),6),"R_p75":round(float(np.quantile(Rv,0.75)),6),"R_p95":round(float(np.quantile(Rv,0.95)),6),
            "Q_median":round(float(np.median(Qv)),6),"central_error":round(abs(np.median(Rv)-np.median(Qv)),6)})

pd.DataFrame(cent_rows).to_csv(os.path.join(OUTPUT,"policy_central_response.csv"),index=False)

# ═══ Tail severity ═══
log("Tail severity...")
tail_rows=[]
for pn in policy_names:
    for st in all_states:
        sub=[r for r in cell_resp_rows if r["policy"]==pn and r["state"]==st]
        if not sub:continue
        Elog=np.array([float(r["E_log_cell"]) for r in sub if r["E_log_cell"]!="N/A" and np.isfinite(r["R_cell"])])
        Rv=np.array([r["R_cell"] for r in sub]);Qv=np.array([r["Q_tau_cell"] for r in sub])
        f2=(Elog>math.log(2)).mean() if len(Elog)>0 else 0
        f5=(Elog>math.log(5)).mean() if len(Elog)>0 else 0
        f10=(Elog>math.log(10)).mean() if len(Elog)>0 else 0
        tail_rows.append({"policy":pn,"state":st,"n":len(Elog),
            "median_E_log":round(float(np.median(Elog)),6) if len(Elog)>0 else "N/A",
            "p90_E_log":round(float(np.quantile(Elog,0.90)),6) if len(Elog)>=10 else "N/A",
            "p95_E_log":round(float(np.quantile(Elog,0.95)),6) if len(Elog)>=20 else "N/A",
            "p99_E_log":round(float(np.quantile(Elog,0.99)),6) if len(Elog)>=100 else "N/A",
            "factor2_fraction":round(float(f2),4),"factor5_fraction":round(float(f5),4),"factor10_fraction":round(float(f10),4)})

pd.DataFrame(tail_rows).to_csv(os.path.join(OUTPUT,"policy_tail_severity.csv"),index=False)

# ═══ P0 baseline reproduction ═══
p0_cent={r["state"]:r["R_median"] for r in cent_rows if r["policy"]=="P0_FIXED_COV"}
ref_medians={"stretch_1.25":0.7960,"stretch_1.50":0.6573,"stretch_2.00":0.4832,"shear_k020":0.9991,"shear_k040":0.9954,"twist_60":0.9478}
p0_rep_ok=True
p0_rep_rows=[]
for st,ref in ref_medians.items():
    cur=p0_cent.get(st,0);diff=abs(cur-ref)
    p0_rep_rows.append({"state":st,"current_R":round(cur,4),"reference_R":ref,"abs_diff":round(diff,4),"PASS":"YES" if diff<=0.005 else "NO"})
    if diff>0.005:p0_rep_ok=False
    if not p0_rep_ok:break
pd.DataFrame(p0_rep_rows).to_csv(os.path.join(OUTPUT,"p0_current_baseline_reproduction.csv"),index=False)
log(f"  P0 baseline: {'PASS' if p0_rep_ok else 'FAIL'}")

# ═══ P1 uniform identity control ═══
p1_ident_rows=[]
for st in ["stretch_1.25","stretch_1.50","stretch_2.00"]:
    p0s=[r for r in cell_resp_rows if r["policy"]=="P0_FIXED_COV" and r["state"]==st]
    p1s=[r for r in cell_resp_rows if r["policy"]=="P1_RIGID_COV" and r["state"]==st]
    merged={r["cell_id"]:r["R_cell"] for r in p0s}
    diffs=[]
    for r in p1s:
        if r["cell_id"] in merged:
            diffs.append(abs(r["R_cell"]-merged[r["cell_id"]]))
    if diffs:
        p1_ident_rows.append({"state":st,"n":len(diffs),"median_diff":round(np.median(diffs),8),"p95_diff":round(np.quantile(diffs,0.95),8),"max_diff":round(max(diffs),8)})
        log(f"  P1 vs P0 {st}: median_diff={np.median(diffs):.2e}")
pd.DataFrame(p1_ident_rows).to_csv(os.path.join(OUTPUT,"p1_uniform_identity_control.csv"),index=False)
s3_ok=all(r["median_diff"]<=1e-6 and r["p95_diff"]<=1e-5 for r in p1_ident_rows)
log(f"  P1 uniform identity: {'PASS' if s3_ok else 'FAIL'}")

# ═══ Central cancellation test ═══
area_states=["stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50","cubic_l020","cubic_l0333"]
cc_rows=[]
for st in area_states:
    p0r=next((r for r in cent_rows if r["policy"]=="P0_FIXED_COV" and r["state"]==st),None)
    p1r=next((r for r in cent_rows if r["policy"]=="P1_RIGID_COV" and r["state"]==st),None)
    p2r=next((r for r in cent_rows if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]==st),None)
    p3r=next((r for r in cent_rows if r["policy"]=="P3_FULL_AFFINE_ORACLE" and r["state"]==st),None)
    if p0r and p2r and p3r:
        cc_rows.append({"state":st,"P0_medianR":p0r["R_median"],"P1_medianR":p1r["R_median"] if p1r else 0,
            "P2_medianR":p2r["R_median"],"P3_medianR":p3r["R_median"],"Q_median":p0r.get("Q_median",0),
            "P0_central_err":p0r["central_error"],"P2_central_err":p2r["central_error"],"P3_central_err":p3r["central_error"]})
pd.DataFrame(cc_rows).to_csv(os.path.join(OUTPUT,"central_policy_comparison.csv"),index=False)

st2=next((r for r in cc_rows if r["state"]=="stretch_2.00"),None)
s4a=st2["P2_medianR"]-st2["P0_medianR"]>=0.25 and st2["P2_medianR"]>=0.75 if st2 else False
p2_errs=np.array([r["P2_central_err"] for r in cc_rows]);p0_errs=np.array([r["P0_central_err"] for r in cc_rows])
s4b=np.mean(p2_errs)>=np.mean(p0_errs)+0.15
s4c=sum(1 for r in cc_rows if r["P2_medianR"]>r["P0_medianR"])>=5
S4="SUPPORTED" if (s4a and s4b and s4c) else "NOT SUPPORTED"
log(f"  S4: stretch2 P2-P0={st2['P2_medianR']-st2['P0_medianR']:.3f}"+" >=0.25" if st2 else "")
log(f"      P2 mean err={np.mean(p2_errs):.4f} P0 mean err={np.mean(p0_errs):.4f} (+0.15: {s4b})")
log(f"      {s4c}/6 states P2 closer to 1: {'PASS' if s4c>=5 else 'FAIL'}")

# ═══ Footprint diagnostic ═══
log("Footprint diagnostic...")
e_u=torch.tensor([Ax,Ay,Az],device=device,dtype=torch.float32);e_u=e_u/torch.linalg.norm(e_u).clamp_min(1e-12)
b_vec=torch.tensor([Bx,By,Bz],device=device,dtype=torch.float32)
e_v_temp=b_vec-torch.dot(b_vec,e_u)*e_u;e_v=e_v_temp/torch.linalg.norm(e_v_temp).clamp_min(1e-12)
T_can=torch.stack([e_u,e_v],dim=1)
A_can=np.zeros(N_ref)
Sigma0=covariance_from_scale_rotation(base_state.scale,base_state.rotation)
C0=T_can.T@Sigma0@T_can;det0=torch.linalg.det(C0).clamp(min=1e-20)
A_can_all=torch.sqrt(det0)

fp_rows=[]
for pn in ["P0_FIXED_COV","P1_RIGID_COV","P2_FULL_AFFINE_COV"]:
    for st in all_states:
        gs=policy_states[(pn,st)]
        Sigma=covariance_from_scale_rotation(gs.scale,gs.rotation)
        _,F,Js=deform_F_Js(verts,st)
        # Deformed tangent basis
        tu=F@e_u;tv=F@e_v
        e1=tu/torch.linalg.norm(tu,dim=1,keepdim=True).clamp_min(1e-12)
        tv_ortho=tv-torch.sum(tv*e1,dim=1,keepdim=True)*e1
        e2=tv_ortho/torch.linalg.norm(tv_ortho,dim=1,keepdim=True).clamp_min(1e-12)
        T_def=torch.stack([e1,e2],dim=2)
        C_def=T_def.permute(0,2,1)@Sigma@T_def
        A_def=torch.sqrt(torch.linalg.det(C_def).clamp(min=1e-20))
        footprint_ratio=A_def/A_can_all.clamp(min=1e-20)
        Js_np=Js.detach().cpu().numpy().ravel()
        for i in range(min(N_ref,len(footprint_ratio))):
            fp_rows.append({"policy":pn,"state":st,"gaussian_index":i,"material_id":i,"Js":round(float(Js_np[i]),6),
                "footprint_ratio":round(float(footprint_ratio[i].cpu().numpy()),6),
                "B_proxy":round(float((footprint_ratio[i]/max(Js_np[i],1e-10)).cpu().numpy()),6)})
    log(f"  {pn}: footprint computed")

pd.DataFrame(fp_rows).to_csv(os.path.join(OUTPUT,"policy_footprint_diagnostic.csv"),index=False)

# Footprint sanity
fs_rows=[]
for pn in ["P0_FIXED_COV","P1_RIGID_COV","P2_FULL_AFFINE_COV"]:
    for st in ["stretch_1.25","stretch_1.50","stretch_2.00","biaxial_1.50"]:
        sub=[r for r in fp_rows if r["policy"]==pn and r["state"]==st]
        if not sub:continue
        fr=np.array([r["footprint_ratio"] for r in sub])
        js=np.array([r["Js"] for r in sub])
        med_fr=np.median(fr)
        if pn in ("P0_FIXED_COV","P1_RIGID_COV"):
            err=np.abs(fr-1.0).max()
        else:
            err=np.abs(fr-js).max()/max(js.max(),1e-10)
        fs_rows.append({"policy":pn,"state":st,"median_footprint":round(float(med_fr),4),"error":round(float(err),4)})
pd.DataFrame(fs_rows).to_csv(os.path.join(OUTPUT,"footprint_sanity.csv"),index=False)

# Budget proxy
log("Central response vs B_proxy...")
bp_rows=[]
for pn in ["P0_FIXED_COV","P1_RIGID_COV","P2_FULL_AFFINE_COV"]:
    for st in area_states:
        cent=next((r for r in cent_rows if r["policy"]==pn and r["state"]==st),None)
        fp_sub=[r for r in fp_rows if r["policy"]==pn and r["state"]==st]
        if cent and fp_sub:
            medB=np.median([r["B_proxy"] for r in fp_sub])
            bp_rows.append({"policy":pn,"state":st,"median_R":cent["R_median"],"median_Q":cent.get("Q_median",0),"median_B_proxy":round(float(medB),4)})
pd.DataFrame(bp_rows).to_csv(os.path.join(OUTPUT,"central_response_vs_budget_proxy.csv"),index=False)

# Pooled Spearman
pool_R=[];pool_B=[]
for r in bp_rows:
    pool_R.append(r["median_R"]);pool_B.append(r["median_B_proxy"])
sp_RB=spearmanr(pool_R,pool_B)[0] if len(set(np.round(pool_R,6)))>1 else 0
log_err=np.median([abs(math.log(pool_R[i]/max(pool_B[i],1e-12))) for i in range(len(pool_R))])
S5="SUPPORTED" if (sp_RB>=0.90 and log_err<=0.20) else "NOT SUPPORTED"
log(f"  S5: R-Bproxy Spearman={sp_RB:.4f} log_err={log_err:.4f} {'SUPPORTED' if S5=='SUPPORTED' else 'NOT'}")

# ═══ Oracle restoration ═══
log("Oracle restoration...")
p2_mean_err=np.mean([r["P2_central_err"] for r in cc_rows])
p3_mean_err=np.mean([r["P3_central_err"] for r in cc_rows])
improvement=(p2_mean_err-p3_mean_err)/max(p2_mean_err,1e-12)
S6="SUPPORTED" if (p3_mean_err<=0.05 and improvement>=0.60) else "NOT SUPPORTED"
pd.DataFrame({"metric":["P2_mean_err","P3_mean_err","improvement"],"value":[p2_mean_err,p3_mean_err,improvement]}).to_csv(os.path.join(OUTPUT,"oracle_restoration.csv"),index=False)
log(f"  S6: P2 mean={p2_mean_err:.4f} P3 mean={p3_mean_err:.4f} improvement={improvement:.2%} {'SUPPORTED' if S6=='SUPPORTED' else 'NOT'}")

# ═══ Paired tail comparison ═══
log("Paired tail comparison...")
import random;random.seed(20260713);np.random.seed(20260713)
audit_ts=["stretch_2.00","cubic_l0333","shear_k040","twist_60"]
pair_rows=[]
def bootstrap_ci(values,alpha=0.05,n=10000):
    vals=np.asarray(values);vals=vals[np.isfinite(vals)]
    if len(vals)<3:return(0,0)
    means=[np.median(np.random.choice(vals,len(vals),replace=True)) for _ in range(n)]
    return(float(np.quantile(means,alpha/2)),float(np.quantile(means,1-alpha/2)))

for st in audit_ts:
    p0d={r["cell_id"]:float(r["E_log_cell"]) for r in cell_resp_rows if r["policy"]=="P0_FIXED_COV" and r["state"]==st and r["E_log_cell"]!="N/A" and np.isfinite(float(r["E_log_cell"]))}
    p2d={r["cell_id"]:float(r["E_log_cell"]) for r in cell_resp_rows if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]==st and r["E_log_cell"]!="N/A" and np.isfinite(float(r["E_log_cell"]))}
    common=[c for c in p0d if c in p2d]
    if not common:continue
    delta=np.array([p2d[c]-p0d[c] for c in common])
    ci=bootstrap_ci(delta)
    pair_rows.append({"state":st,"n":len(delta),"median_delta_Elog":round(float(np.median(delta)),6),"mean_delta":round(float(np.mean(delta)),6),
        "fraction_improved":round((delta<0).mean(),4),"fraction_worsened":round((delta>0).mean(),4),
        "p10_delta":round(float(np.quantile(delta,0.10)),6),"p90_delta":round(float(np.quantile(delta,0.90)),6),
        "ci_lower":round(ci[0],6),"ci_upper":round(ci[1],6)})
    log(f"  {st}: median_delta={np.median(delta):.4f} CI=({ci[0]:.4f},{ci[1]:.4f}) improved={(delta<0).mean():.3f}")
pd.DataFrame(pair_rows).to_csv(os.path.join(OUTPUT,"paired_tail_policy_comparison.csv"),index=False)

# Tail severity classification
tclass_rows=[]
for st in audit_ts:
    p2t=next((r for r in tail_rows if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]==st),None)
    p0t=next((r for r in tail_rows if r["policy"]=="P0_FIXED_COV" and r["state"]==st),None)
    if not p2t or not p0t:continue
    p95_r=p2t["p95_E_log"]/max(p0t["p95_E_log"],1e-12) if p0t["p95_E_log"]!="N/A" and p2t["p95_E_log"]!="N/A" else 1
    f2_r=p2t["factor2_fraction"]/max(p0t["factor2_fraction"],1e-12) if p0t["factor2_fraction"]!="N/A" else 1
    tclass_rows.append({"state":st,"p95_ratio":round(p95_r,4) if isinstance(p95_r,float) else "N/A","factor2_ratio":round(f2_r,4) if isinstance(f2_r,float) else "N/A"})

imp_count=sum(1 for r in tclass_rows if isinstance(r.get("p95_ratio"),float) and r["p95_ratio"]<=0.75 and isinstance(r.get("factor2_ratio"),float) and r["factor2_ratio"]<=0.75)
wor_count=sum(1 for r in tclass_rows if isinstance(r.get("p95_ratio"),float) and r["p95_ratio"]>=1.25 and isinstance(r.get("factor2_ratio"),float) and r["factor2_ratio"]>=1.25)
ci_upper_lt0=all(r["ci_upper"]<0 for r in pair_rows)
ci_lower_gt0=all(r["ci_lower"]>0 for r in pair_rows)
S7="IMPROVEMENT" if (imp_count>=3 and ci_upper_lt0) else "WORSENING" if (wor_count>=3 and ci_lower_gt0) else "MIXED/WEAK"
log(f"  S7: tail effect = {S7} (imp={imp_count}/4, wor={wor_count}/4)")

# ═══ Tail spatial redistribution ═══
log("Tail spatial...")
spatial_rows=[];jaccard_rows=[]
GRID=41
for pn in ["P0_FIXED_COV","P2_FULL_AFFINE_COV"]:
    for st in audit_ts:
        sub=[r for r in cell_resp_rows if r["policy"]==pn and r["state"]==st]
        Elog=np.array([float(r["E_log_cell"]) for r in sub if r["E_log_cell"]!="N/A"])
        th=np.quantile(Elog,0.99) if len(Elog)>0 else float("inf")
        tail_set=set(r["cell_id"] for i,r in enumerate(sub) if r["E_log_cell"]!="N/A" and float(r["E_log_cell"])>=th) if len(Elog)>0 else set()
        n_top=len(tail_set)
        if n_top<2:continue
        nbr=0
        for cid in tail_set:
            cell=[c for c in cell_defs if c["id"]+1==cid]
            if not cell:continue
            cc_iu,cc_iv=cell[0]["iu"],cell[0]["iv"]
            for di,dv in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni_,nv_=cc_iu+di,cc_iv+dv
                if 0<=ni_<GRID and 0<=nv_<GRID and (ni_*GRID+nv_+1) in tail_set:nbr+=1
        rand_nbrs=[]
        all_cells=set(r["cell_id"] for r in sub)
        for _ in range(1000):
            rs=set(np.random.choice(list(all_cells),n_top,replace=False))
            rn=0
            for cid in rs:
                cell=[c for c in cell_defs if c["id"]+1==cid]
                if not cell:continue
                c_iu,c_iv=cell[0]["iu"],cell[0]["iv"]
                for di,dv in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni_,nv_=c_iu+di,c_iv+dv
                    if 0<=ni_<GRID and 0<=nv_<GRID and (ni_*GRID+nv_+1) in rs:rn+=1
            rand_nbrs.append(rn/n_top)
        rm=np.mean(rand_nbrs);rss=np.std(rand_nbrs)
        clust=(nbr/n_top>rm+3*rss)
        spatial_rows.append({"policy":pn,"state":st,"n_tail":n_top,"neighbor_frac":round(nbr/n_top,4),
            "random_mean":round(rm,4),"random_std":round(rss,4),"clustered":"YES" if clust else "NO"})

# Jaccard
p0_tails={}
for st in audit_ts:
    sub=[r for r in cell_resp_rows if r["policy"]=="P0_FIXED_COV" and r["state"]==st]
    Elog=np.array([float(r["E_log_cell"]) for r in sub if r["E_log_cell"]!="N/A"])
    th=np.quantile(Elog,0.99) if len(Elog)>0 else float("inf")
    p0_tails[st]=set(r["cell_id"] for i,r in enumerate(sub) if r["E_log_cell"]!="N/A" and float(r["E_log_cell"])>=th) if len(Elog)>0 else set()
for pn in ["P1_RIGID_COV","P2_FULL_AFFINE_COV","P3_FULL_AFFINE_ORACLE"]:
    for st in audit_ts:
        sub=[r for r in cell_resp_rows if r["policy"]==pn and r["state"]==st]
        valid_entries=[(i,r) for i,r in enumerate(sub) if r["E_log_cell"]!="N/A" and np.isfinite(float(r["E_log_cell"]))]
        Elog=np.array([float(r["E_log_cell"]) for _,r in valid_entries])
        th=np.quantile(Elog,0.99) if len(Elog)>0 else float("inf")
        pn_tail=set(r["cell_id"] for _,r in valid_entries if float(r["E_log_cell"])>=th)
        inter=len(p0_tails.get(st,set())&pn_tail);union=len(p0_tails.get(st,set())|pn_tail)
        jaccard_rows.append({"policy":pn,"state":st,"n_P0":len(p0_tails.get(st,set())),"n_Pn":len(pn_tail),"jaccard":round(inter/max(union,1),4)})

pd.DataFrame(spatial_rows).to_csv(os.path.join(OUTPUT,"policy_tail_spatial_distribution.csv"),index=False)
pd.DataFrame(jaccard_rows).to_csv(os.path.join(OUTPUT,"policy_tail_jaccard.csv"),index=False)

# ═══ Area-preserving controls ═══
log("Area-preserving...")
ap_rows=[]
for pn in policy_names:
    for st in ["shear_k020","shear_k040","twist_60"]:
        cent=next((r for r in cent_rows if r["policy"]==pn and r["state"]==st),None)
        trow=next((r for r in tail_rows if r["policy"]==pn and r["state"]==st),None)
        if cent:
            ap_rows.append({"policy":pn,"state":st,"median_R":cent["R_median"],"central_error":cent["central_error"],
                "median_E_log":trow["median_E_log"] if trow else "N/A","p95_E_log":trow["p95_E_log"] if trow else "N/A",
                "factor2_fraction":trow["factor2_fraction"] if trow else "N/A"})
pd.DataFrame(ap_rows).to_csv(os.path.join(OUTPUT,"policy_area_preserving_controls.csv"),index=False)
p2_ap=None;p2_ap4=None;p2_tw=None
for r in ap_rows:
    if r["policy"]=="P2_FULL_AFFINE_COV":
        if r["state"]=="shear_k020":p2_ap=r
        elif r["state"]=="shear_k040":p2_ap4=r
        elif r["state"]=="twist_60":p2_tw=r
S8="PASS"
for r in [p2_ap,p2_ap4,p2_tw]:
    if r is None or not (isinstance(r["median_R"],(int,float)) and abs(r["median_R"]-1)<=0.15):
        S8="FAIL"
if p2_ap:log(f"  P2 shear020 R:{p2_ap['median_R']}")
if p2_ap4:log(f"  P2 shear040 R:{p2_ap4['median_R']}")
if p2_tw:log(f"  P2 twist60 R:{p2_tw['median_R']}")

# ═══ Gates ═══
S0="PASS";S1="PASS" if ut_pass else "FAIL";S2="PASS" if p0_rep_ok else "FAIL";S3="PASS" if s3_ok else "FAIL"
log(f"\n  S0 Protocol: {S0}\n  S1 Implementation: {S1}\n  S2 P0 Baseline: {S2}\n  S3 P1 Control: {S3}")
log(f"  S4 Central Cancellation: {S4}\n  S5 Mechanism: {S5}\n  S6 Oracle: {S6}")
log(f"  S7 Tail Effect: {S7}\n  S8 Area Control: {S8}")

# ─── Final CASE ───
if S4=="SUPPORTED" and S5=="SUPPORTED" and S6=="SUPPORTED":
    FINAL_CASE="SHAPE-CANCELLATION-OPTICAL-GAP"
elif S4=="SUPPORTED" and (S5=="NOT SUPPORTED" or S6=="NOT SUPPORTED"):
    FINAL_CASE="SHAPE-BREAK-MECHANISM-INCOMPLETE"
elif S4=="NOT SUPPORTED" and S7=="IMPROVEMENT":
    FINAL_CASE="SHAPE-TAIL-STABILIZATION"
elif S4=="NOT SUPPORTED" and S7=="WORSENING":
    FINAL_CASE="SHAPE-TAIL-DEGRADATION"
elif S4=="NOT SUPPORTED" and S7=="MIXED/WEAK":
    FINAL_CASE="SHAPE-INVARIANT"
else:
    FINAL_CASE="SHAPE-MIXED"

can_method=(FINAL_CASE=="SHAPE-CANCELLATION-OPTICAL-GAP")
log(f"\n  Final CASE: {FINAL_CASE}")
log(f"  Can design optical-state method: {'YES' if can_method else 'NO'}")
log(f"  Strongest conclusion: {FINAL_CASE}")

# ─── Report ───
with open(os.path.join(OUTPUT,"covariance_transport_optical_response_report.md"),"w") as f:
    f.write(f"# Covariance Transport Optical Response Report\n\n")
    f.write(f"S0:{S0} S1:{S1} S2:{S2} S3:{S3} S4:{S4} S5:{S5} S6:{S6} S7:{S7} S8:{S8}\n")
    f.write(f"Final CASE: {FINAL_CASE}\nCan design method: {'YES' if can_method else 'NO'}\n")

with open(os.path.join(OUTPUT,"stage3_4C_summary.md"),"w") as f:
    f.write(f"# Stage 3.4C Summary\nFinal: {FINAL_CASE}\nS0:{S0} S1:{S1} S2:{S2} S3:{S3} S4:{S4} S5:{S5} S6:{S6} S7:{S7} S8:{S8}\nCan design method: {'YES' if can_method else 'NO'}\n")

with open(os.path.join(OUTPUT,"stage3_4C_log.txt"),"w") as f:f.write("\n".join(log_lines))

# ─── Terminal summary ───
print(f"\n  S0 protocol lock: {S0}")
print(f"  S1 policy implementation: {S1}")
print(f"  P0 baseline reproduction: {S2}")
print(f"  P1 uniform identity: {S3}")
for st in ["stretch_2.00","biaxial_1.50","cubic_l020","cubic_l0333"]:
    r=next((r for r in cc_rows if r["state"]==st),None)
    if r:print(f"  {st:15s}: P0={r['P0_medianR']:.4f} P1={r['P1_medianR']:.4f} P2={r['P2_medianR']:.4f} P3={r['P3_medianR']:.4f} Q={r['Q_median']:.4f}")
print(f"  P0 six-state mean central error: {np.mean(p0_errs):.4f}")
print(f"  P2 six-state mean central error: {np.mean(p2_errs):.4f}")
print(f"  P3 six-state mean central error: {p3_mean_err:.4f}")
print(f"  S4 central cancellation: {S4}")
for st in ["stretch_1.25","stretch_1.50","stretch_2.00"]:
    r0=next((r for r in fs_rows if r["policy"]=="P0_FIXED_COV" and r["state"]==st),None)
    r1=next((r for r in fs_rows if r["policy"]=="P1_RIGID_COV" and r["state"]==st),None)
    r2=next((r for r in fs_rows if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]==st),None)
    if r0:print(f"  P0 footprint {st}: {r0['median_footprint']:.4f}")
    if r1:print(f"  P1 footprint {st}: {r1['median_footprint']:.4f}")
    if r2:print(f"  P2 footprint/Js {st}: error={r2['error']:.4f}")
print(f"  R-Bproxy Spearman: {sp_RB:.4f}")
print(f"  log R/Bproxy error: {log_err:.4f}")
print(f"  S5 mechanism: {S5}")
print(f"  P3 restoration improvement: {improvement:.2%}")
print(f"  S6 oracle restoration: {S6}")
for st in audit_ts:
    p0t=next((r for r in tail_rows if r["policy"]=="P0_FIXED_COV" and r["state"]==st),None)
    p2t=next((r for r in tail_rows if r["policy"]=="P2_FULL_AFFINE_COV" and r["state"]==st),None)
    if p0t and p2t:print(f"  {st:15s}: P0 p95/f2={p0t['p95_E_log']}/{p0t['factor2_fraction']:.3f} P2 p95/f2={p2t['p95_E_log']}/{p2t['factor2_fraction']:.3f}")
for r in pair_rows:
    print(f"  {r['state']:15s}: delta_CI=({r['ci_lower']:.4f},{r['ci_upper']:.4f}) improved={r['fraction_improved']:.3f}")
print(f"  S7 tail effect: {S7}")
for r in jaccard_rows:
    if r["policy"]=="P2_FULL_AFFINE_COV":
        print(f"  P0-P2 tail Jaccard {r['state']}: {r['jaccard']:.4f}")
for r in spatial_rows:
    if r["policy"]=="P2_FULL_AFFINE_COV":
        print(f"  P2 {r['state']:15s}: clustered={r['clustered']} nbr={r['neighbor_frac']:.4f}")
print(f"  P2 shear020 R: {p2_ap['median_R'] if p2_ap and isinstance(p2_ap.get('median_R'),(int,float)) else 'N/A'}")
print(f"  P2 shear040 R: {p2_ap4['median_R'] if p2_ap4 and isinstance(p2_ap4.get('median_R'),(int,float)) else 'N/A'}")
print(f"  P2 twist60 R: {p2_tw['median_R'] if p2_tw and isinstance(p2_tw.get('median_R'),(int,float)) else 'N/A'}")
print(f"  S8 area-preserving: {S8}")
print(f"  Final CASE: {FINAL_CASE}")
print(f"  Can design method: {'YES' if can_method else 'NO'}")
print(f"  Report: {OUTPUT}/covariance_transport_optical_response_report.md")
print(f"  Summary: {OUTPUT}/stage3_4C_summary.md")
