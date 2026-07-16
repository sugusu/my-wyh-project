
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np

H0=0.08
K_PERP=np.array([0.35,0.55,0.80],dtype=np.float64)
K_PAR=np.array([1.10,0.75,0.45],dtype=np.float64)
K_NORM=np.array([0.45,0.60,0.75],dtype=np.float64)
VIEWS={"V0_NORMAL":np.array([0.,0.,1.]),"V1_T1_OBLIQUE":np.array([0.6,0.,0.8]),"V2_T2_OBLIQUE":np.array([0.,0.6,0.8]),"V3_DIAGONAL_OBLIQUE":np.array([0.4242640687,0.4242640687,0.8]),"V4_NEG_T1_OBLIQUE":np.array([-0.6,0.,0.8]),"V5_NEG_T2_OBLIQUE":np.array([0.,-0.6,0.8])}
for k in list(VIEWS): VIEWS[k]=VIEWS[k]/np.linalg.norm(VIEWS[k])
CHANNELS=("R","G","B")

def surface_eval(surface,u,v):
    if surface=="S0_PLANAR_SHEET":
        du=np.stack([np.ones_like(u),np.zeros_like(u),np.zeros_like(u)],axis=-1)
        dv=np.stack([np.zeros_like(u),np.ones_like(u),np.zeros_like(u)],axis=-1)
    else:
        dzdu=0.18*np.pi*np.cos(np.pi*u)*np.sin(np.pi*v); dzdv=0.18*np.pi*np.sin(np.pi*u)*np.cos(np.pi*v)
        du=np.stack([np.ones_like(u),np.zeros_like(u),dzdu],axis=-1); dv=np.stack([np.zeros_like(u),np.ones_like(u),dzdv],axis=-1)
    n=np.cross(du,dv); n/=np.linalg.norm(n,axis=-1,keepdims=True)+1e-30
    t1=du/(np.linalg.norm(du,axis=-1,keepdims=True)+1e-30); t2=np.cross(n,t1); t2/=np.linalg.norm(t2,axis=-1,keepdims=True)+1e-30
    return t1,t2,n

def deformed_frame(t1,t2,F):
    ft1=t1@F.T; ft2=t2@F.T
    n=np.cross(ft1,ft2); j=np.linalg.norm(n,axis=-1); n/=j[:,None]+1e-30
    t1p=ft1/(np.linalg.norm(ft1,axis=-1,keepdims=True)+1e-30); t2p=np.cross(n,t1p); t2p/=np.linalg.norm(t2p,axis=-1,keepdims=True)+1e-30
    return t1p,t2p,n,j

def ortho_basis(d):
    ref=np.tile(np.array([0.,0.,1.]),(len(d),1)); alt=np.tile(np.array([0.,1.,0.]),(len(d),1))
    mask=np.abs(np.sum(d*ref,axis=1))>0.9; ref[mask]=alt[mask]
    e1=np.cross(d,ref); e1/=np.linalg.norm(e1,axis=1,keepdims=True)+1e-30
    e2=np.cross(d,e1); e2/=np.linalg.norm(e2,axis=1,keepdims=True)+1e-30
    return e1,e2

def load_micro(out, fam):
    short=fam.split('_')[0]
    rows=list(csv.DictReader((Path(out)/"D1N2_microstructure"/f"{short}.csv").open(newline='',encoding='utf-8')))
    return np.array([[float(r["m1"]),float(r["m2"])] for r in rows]), np.array([float(r["weight"]) for r in rows])

def compute_K(t1,t2,F,dirs,weights):
    t1p,t2p,n,j=deformed_frame(t1,t2,F)
    K=np.zeros((len(t1),3,3,3)); I=np.eye(3)
    P=I[None,:,:]-n[:,:,None]*n[:,None,:]
    if float(np.max(weights)-np.min(weights))<=1e-14:
        for c in range(3):
            tangent_iso=K_PERP[c]+0.5*(K_PAR[c]-K_PERP[c])
            K[:,c]=tangent_iso*P+K_NORM[c]*(n[:,:,None]*n[:,None,:])
    else:
        for m,w in zip(dirs,weights):
            a=m[0]*t1+m[1]*t2; b=a@F.T; b/=np.linalg.norm(b,axis=1,keepdims=True)+1e-30
            outer=b[:,:,None]*b[:,None,:]
            for c in range(3): K[:,c]+=w*(K_PERP[c]*I[None,:,:]+(K_PAR[c]-K_PERP[c])*outer)
        for c in range(3):
            K[:,c]=P@K[:,c]@P+K_NORM[c]*(n[:,:,None]*n[:,None,:]); K[:,c]=0.5*(K[:,c]+np.swapaxes(K[:,c],1,2))
    return K,t1p,t2p,n,j

def transmission(K,n,j,lv,t1p,t2p):
    d=lv[0]*t1p+lv[1]*t2p+lv[2]*n; d/=np.linalg.norm(d,axis=1,keepdims=True)+1e-30
    e1,e2=ortho_basis(d); cos=np.maximum(np.abs(np.sum(n*d,axis=1)),0.15); h=H0/np.maximum(j,1e-30)
    mu=np.zeros((len(d),3,2)); T=np.zeros((len(d),3)); tau=np.zeros((len(d),3))
    for c in range(3):
        a=np.einsum("ni,nij,nj->n",e1,K[:,c],e1); b=np.einsum("ni,nij,nj->n",e1,K[:,c],e2); cc=np.einsum("ni,nij,nj->n",e2,K[:,c],e2)
        tr=a+cc; disc=np.sqrt(np.maximum((a-cc)**2+4*b*b,0.0)); mu[:,c,0]=0.5*(tr+disc); mu[:,c,1]=0.5*(tr-disc)
        T[:,c]=0.5*(np.exp(-h*mu[:,c,0]/cos)+np.exp(-h*mu[:,c,1]/cos)); tau[:,c]=-np.log(np.clip(T[:,c],1e-12,1.0))
    return d,h,mu,T,tau

def run_replay(out, base, limit=100000):
    out=Path(out); base=Path(base)
    bank=list(csv.DictReader((out/"D1N2_unique_deformation_lock.csv").open(newline='',encoding='utf-8')))
    F={r["matrix_key"]:np.array(json.loads(r["F"]),dtype=np.float64) for r in bank}
    samples={}
    for s,rel in [("S0_PLANAR_SHEET","S0.csv"),("S1_WAVY_MEMBRANE","S1.csv")]:
        rows=list(csv.DictReader((base/"experiments/stageD0_deformable_optical_transport_feasibility/D0_material_samples"/rel).open(newline='',encoding='utf-8')))
        u=np.array([float(r["u"]) for r in rows]); v=np.array([float(r["v"]) for r in rows]); t1,t2,_=surface_eval(s,u,v); samples[s]=(rows,t1,t2)
    cache={}
    errs={k:[] for k in ["K","mu","T","tau"]}
    rows_out=[]; checked=0
    for row in csv.DictReader((out/"D1N2_microstructure_optical_oracle.csv").open(newline='',encoding='utf-8')):
        if checked>=limit: break
        key=(row["surface"],row["deformation_matrix_key"],row["microstructure_family"],row["view_key"])
        if key not in cache:
            _,t1,t2=samples[row["surface"]]; dirs,w=load_micro(out,row["microstructure_family"])
            K,t1p,t2p,n,j=compute_K(t1,t2,F[row["deformation_matrix_key"]],dirs,w)
            cache[key]=(K,*transmission(K,n,j,VIEWS[row["view_key"]],t1p,t2p))
        K,d,h,mu,T,tau=cache[key]; i=int(row["sample_id"]); c=CHANNELS.index(row["channel"])
        Krow=np.array([[float(row["K00"]),float(row["K01"]),float(row["K02"])],[float(row["K10"]),float(row["K11"]),float(row["K12"])],[float(row["K20"]),float(row["K21"]),float(row["K22"])]])
        errs["K"].append(float(np.max(np.abs(K[i,c]-Krow))))
        mref=np.array([float(row["mu1"]),float(row["mu2"])]); errs["mu"].append(float(np.max(np.abs(mu[i,c]-mref)/np.maximum(np.abs(mref),1e-12))))
        errs["T"].append(abs(float(row["T"])-float(T[i,c]))); errs["tau"].append(abs(float(row["tau_eff"])-float(tau[i,c])))
        checked+=1
    def stat(x): 
        a=np.array(x); return float(np.quantile(a,0.99)), float(a.max())
    Kp,Km=stat(errs["K"]); mup,mum=stat(errs["mu"]); Tp,Tm=stat(errs["T"]); tap,tam=stat(errs["tau"])
    rows_out.append({"checked_rows":checked,"K_p99":Kp,"K_max":Km,"mu_p99":mup,"mu_max":mum,"T_p99":Tp,"T_max":Tm,"tau_eff_p99":tap,"tau_eff_max":tam})
    with (out/"D1N2_oracle_replay.csv").open("w",newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,rows_out[0].keys()); w.writeheader(); w.writerows(rows_out)
    ok=Kp<=1e-10 and Km<=1e-8 and mup<=1e-9 and mum<=1e-7 and Tp<=1e-10 and Tm<=1e-8 and tap<=1e-10 and tam<=1e-8
    return {"O5":"PASS" if ok else "FAIL","K_p99":Kp,"K_max":Km,"mu_p99":mup,"mu_max":mum,"T_p99":Tp,"T_max":Tm,"tau_p99":tap,"tau_max":tam}
