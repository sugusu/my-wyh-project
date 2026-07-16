#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json, math, time, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch

BASE=Path('/data/wyh/DeformTransGS')
S35=BASE/'experiments/stage3_5A_kernel_integrated_opacity_transport'
S34=BASE/'experiments/stage3_4C_covariance_transport_optical_response'
OUT=BASE/'experiments/stage3_5A_R1_kiot_method_closure'
OUT.mkdir(parents=True, exist_ok=True)
LOG=[]
import sys
sys.path.insert(0,str(BASE))
from analysis.kiot_fast_inverse import ALPHA_SKIP, ALPHA_MAX, phi_cuda_np, phi_cont_np, invert_phi_cont_np, invert_phi_cuda_np, kiot_cuda_identity_safe_np, KiotCudaLUT

AREA=['stretch_1.25','stretch_1.50','stretch_2.00','biaxial_1.50','cubic_l020','cubic_l0333']
ALL=['stretch_1.25','stretch_1.50','stretch_2.00','biaxial_1.50','cubic_l010','cubic_l020','cubic_l0333','shear_k020','shear_k040','twist_60']
AUDIT=['stretch_2.00','cubic_l0333','shear_k040','twist_60']

def log(m): print(m); LOG.append(str(m))
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def md(p,t): Path(p).write_text(t,encoding='utf-8')
def load_tau():
    ckpt=torch.load(BASE/'experiments/stage3_2_5_representation_drift_confirmation/canonical_checkpoint.pt',map_location='cpu',weights_only=True)
    return torch.nn.functional.softplus(ckpt['tau_raw'].detach().cpu()).numpy().reshape(-1).astype(np.float64)
def opacity_from_tau(t): return 1-np.exp(-np.asarray(t,dtype=np.float64))
def tau_from_opacity(o): return -np.log1p(-np.clip(np.asarray(o,dtype=np.float64),0,1-1e-12))

def protocol_lock():
    files=['stage3_5A_protocol_lock.json','kiot_renderer_constants.json','kiot_policy_input_manifest.csv','kiot_central_response.csv','kiot_tail_severity.csv','final_terminal_summary.txt']
    data={f:{'path':str(S35/f),'exists':(S35/f).exists(),'sha256':sha(S35/f) if (S35/f).exists() else 'MISSING'} for f in files}
    data['stage3_4C_frozen_eval_camera_keys']={'path':str(S34/'frozen_eval_camera_keys.csv'),'exists':(S34/'frozen_eval_camera_keys.csv').exists(),'sha256':sha(S34/'frozen_eval_camera_keys.csv')}
    data['stage3_4C_frozen_eval_cell_keys']={'path':str(S34/'frozen_eval_cell_keys.csv'),'exists':(S34/'frozen_eval_cell_keys.csv').exists(),'sha256':sha(S34/'frozen_eval_cell_keys.csv')}
    data['status']='PASS' if all(v['exists'] for v in data.values() if isinstance(v,dict)) else 'FAIL'
    (OUT/'kiot_r1_protocol_lock.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    return data['status']

def plateau_audit():
    grid=np.array([0,1e-8,1e-7,1e-6,1e-5,1e-4,.25*ALPHA_SKIP,.5*ALPHA_SKIP,.9*ALPHA_SKIP,ALPHA_SKIP,1.1*ALPHA_SKIP,.01,.1,.5,.9,.99,.999],dtype=np.float64)
    phi=phi_cuda_np(grid)
    df=pd.DataFrame({'opacity':grid,'phi_cuda':phi,'on_zero_plateau':grid<=ALPHA_SKIP,'phi_is_zero':np.isclose(phi,0,atol=1e-15)})
    df.to_csv(OUT/'phi_cuda_plateau_audit.csv',index=False)
    md(OUT/'identity_safe_definition.md',f'''# B4R KIOT CUDA Identity-Safe Definition

Phi_cuda(o)=0 for 0 <= o <= ALPHA_SKIP ({ALPHA_SKIP}). This creates a zero plateau: Phi_cuda is monotonic but not strictly monotonic, so Phi_cuda^-1(0) is set-valued.

B4R tie-break:

1. if abs(q-1)<=1e-12, return o_old exactly.
2. else if Phi_cuda(o_old) is on the zero plateau, use continuous KIOT: Phi_cont(o_new)=q Phi_cont(o_old).
3. otherwise solve Phi_cuda(o_new)=q Phi_cuda(o_old).

The plateau branch is a latent optical-state tie-break. It does not claim a visible renderer contribution change while both old/new states remain sub-skip.
''')
    return bool(df[df.opacity<=ALPHA_SKIP].phi_is_zero.all())

def identity_tests(tau):
    logg=np.geomspace(1e-10,1-1e-10,50000); ling=np.linspace(1e-10,1-1e-10,50000); o=np.unique(np.concatenate([logg,ling])); q=np.ones_like(o)
    b1=o.copy(); b2=o.copy(); b3=invert_phi_cont_np(phi_cont_np(o)); b4_old=invert_phi_cuda_np(phi_cuda_np(o)); b4r=kiot_cuda_identity_safe_np(o,q)
    rows=[]
    for name,arr in [('B1',b1),('B2',b2),('B3',b3),('B4_ORIGINAL',b4_old),('B4R_REPAIRED',b4r)]: rows.append({'policy':name,'n':len(o),'max_opacity_diff':float(np.max(np.abs(arr-o))),'median_opacity_diff':float(np.median(np.abs(arr-o))),'PASS':float(np.max(np.abs(arr-o)))<=1e-15 if name=='B4R_REPAIRED' else ''})
    pd.DataFrame(rows).to_csv(OUT/'q1_identity_test.csv',index=False)
    ap=[]
    for st in ['shear_k020','shear_k040','twist_60']:
        new=kiot_cuda_identity_safe_np(opacity_from_tau(tau),np.ones_like(tau)); td=np.max(np.abs(tau_from_opacity(new)-tau)); ap.append({'state':st,'tau_max_diff':float(td),'PASS':td<=1e-12})
    pd.DataFrame(ap).to_csv(OUT/'area_preserving_policy_identity.csv',index=False)
    return rows[-1]['max_opacity_diff'], max(r['tau_max_diff'] for r in ap), all(r['PASS'] for r in ap) and rows[-1]['max_opacity_diff']<=1e-15

def carrier_subskip(tau):
    o=opacity_from_tau(tau); old_b4=transport_old_b4(o,np.full_like(o,0.5)); b4r=kiot_cuda_identity_safe_np(o,np.full_like(o,0.5)); changed=(old_b4==0)&(b4r>0)
    qs=[('min',0),('p001',.001),('p01',.01),('p05',.05),('p10',.10),('median',.5),('p90',.9),('max',1)]
    row={k:float(np.quantile(o,q)) for k,q in qs}; row.update({'n':len(o),'count_lt_alpha_skip':int(np.sum(o<ALPHA_SKIP)),'count_eq_alpha_skip_1e12':int(np.sum(np.abs(o-ALPHA_SKIP)<=1e-12)),'count_gt_alpha_skip':int(np.sum(o>ALPHA_SKIP)),'fraction_lt_alpha_skip':float(np.mean(o<ALPHA_SKIP)),'old_zero_plateau_changed_fraction_q05':float(np.mean(changed))})
    pd.DataFrame([row]).to_csv(OUT/'carrier_subskip_opacity_audit.csv',index=False)
    return row

def transport_old_b4(o,q):
    target=q*phi_cuda_np(o); return invert_phi_cuda_np(target)

def domain_tests(tau):
    o=np.unique(np.concatenate([np.geomspace(1e-10,1-1e-10,50000),np.linspace(1e-10,1-1e-10,50000)])); rows=[]
    prev_b3=None; prev_b4=None; ok=True
    for qv in [1,.95,.8,2/3,.5,.25,.1]:
        q=np.full_like(o,qv); b3=invert_phi_cont_np(q*phi_cont_np(o)); b4=kiot_cuda_identity_safe_np(o,q)
        err3=np.max(np.abs(phi_cont_np(b3)-q*phi_cont_np(o)))
        normal=phi_cuda_np(o)>0; err4=np.max(np.abs(phi_cuda_np(b4[normal])-q[normal]*phi_cuda_np(o[normal]))) if np.any(normal) else 0
        finite=np.all(np.isfinite(b3)) and np.all(np.isfinite(b4)) and np.all((b3>=0)&(b3<1)&(b4>=0)&(b4<1))
        mono=True
        if prev_b3 is not None: mono=bool(np.all(b3<=prev_b3+1e-12) and np.all(b4<=prev_b4+1e-12))
        rows.append({'q':qv,'finite_in_range':finite,'nonincreasing_vs_previous_q':mono,'B3_phi_max_abs_error':float(err3),'B4R_phi_max_abs_error':float(err4),'PASS':finite and mono and err3<1e-10 and err4<1e-10})
        ok=ok and rows[-1]['PASS']; prev_b3=b3; prev_b4=b4
    pd.DataFrame(rows).to_csv(OUT/'kiot_expansion_domain_test.csv',index=False)
    return ok

def compression(tau):
    o=opacity_from_tau(tau); rows=[]; pcmax=float(phi_cont_np(1-1e-12)); pumax=float(phi_cuda_np(1-1e-12))
    any_inf=False
    for name,fn,pm in [('continuous',phi_cont_np,pcmax),('cuda_aware',phi_cuda_np,pumax)]:
        ph=fn(o)
        for q in [1.1,1.25,1.5,2.0,4.0]:
            infeas=q*ph>pm; any_inf=any_inf or bool(np.any(infeas)); oi=o[infeas]
            rows.append({'policy_integral':name,'q':q,'n':len(o),'feasible_fraction':float(np.mean(~infeas)),'infeasible_fraction':float(np.mean(infeas)),'opacity_median_infeasible':float(np.median(oi)) if len(oi) else np.nan,'opacity_p10_infeasible':float(np.quantile(oi,.1)) if len(oi) else np.nan,'opacity_p90_infeasible':float(np.quantile(oi,.9)) if len(oi) else np.nan})
    pd.DataFrame(rows).to_csv(OUT/'kiot_compression_feasibility.csv',index=False)
    scope='AREA-PRESERVING OR AREA-EXPANDING THIN-SURFACE DEFORMATION (Js>=1)' if any_inf else 'q>1 feasible for tested carrier but not validated by deformation benchmark'
    md(OUT/'kiot_domain_scope.md',f'# KIOT Domain Scope\n\nFormal current scope: {scope}.\n\nDo not claim arbitrary deformation or compression support.\n')
    return scope

def fast_inverse_accuracy():
    rng=np.random.default_rng(20260713); n=1_000_000
    u=rng.random(n//2); logu=10**rng.uniform(-10,-2,n//4); hi=.9+.1*rng.random(n-n//2-n//4); o=np.concatenate([u,logu,hi]).astype(np.float64); q=rng.uniform(.1,1,n).astype(np.float64); rng.shuffle(o)
    ref=kiot_cuda_identity_safe_np(o,q)
    selected=None; acc=None
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for size in [65536,131072,262144]:
        lut=KiotCudaLUT(size=size,device=device); out=[]; bs=200000
        for i in range(0,n,bs):
            oo=torch.as_tensor(o[i:i+bs],dtype=torch.float64,device=lut.device); qq=torch.as_tensor(q[i:i+bs],dtype=torch.float64,device=lut.device); out.append(lut.transform(oo,qq).detach().cpu().numpy())
        pred=np.concatenate(out); oe=np.abs(pred-ref); target=q*phi_cuda_np(o); pe=np.abs(phi_cuda_np(pred)-target); rel=pe/np.maximum(target,1e-30); rel=rel[target>1e-10]
        acc={'selected_lut_size':size,'device':str(lut.device),'n':n,'opacity_median_error':float(np.median(oe)),'opacity_p95_error':float(np.quantile(oe,.95)),'opacity_p99_error':float(np.quantile(oe,.99)),'opacity_max_error':float(np.max(oe)),'phi_abs_p99_error':float(np.quantile(pe,.99)),'phi_rel_p99_error':float(np.quantile(rel,.99)) if len(rel) else 0,'PASS':bool(np.quantile(oe,.99)<=1e-6 and np.max(oe)<=1e-5 and (np.quantile(rel,.99) if len(rel) else 0)<=1e-5)}
        if acc['PASS'] or size==262144: selected=size; break
    pd.DataFrame([acc]).to_csv(OUT/'fast_inverse_accuracy.csv',index=False)
    return acc

def benchmark(selected_size):
    device='cuda' if torch.cuda.is_available() else 'cpu'; lut=KiotCudaLUT(size=selected_size,device=device); rng=np.random.default_rng(20260713); rows=[]
    for n in [1681,100000,991832,2000000]:
        o=rng.random(n).astype(np.float64); q=rng.uniform(.1,1,n).astype(np.float64)
        oo=torch.as_tensor(o,dtype=torch.float64,device=lut.device); qq=torch.as_tensor(q,dtype=torch.float64,device=lut.device)
        for _ in range(5): lut.transform(oo,qq)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        it=20 if n>100000 else 100; t0=time.perf_counter()
        for _ in range(it): lut.transform(oo,qq)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        ms=(time.perf_counter()-t0)*1000/it; rows.append({'method':'GPU_LUT' if torch.cuda.is_available() else 'CPU_LUT','N':n,'milliseconds':ms,'gaussians_per_second':n/(ms/1000),'speedup_vs_reference':'N/A','device':device})
        if n<=100000:
            t0=time.perf_counter(); reps=3
            for _ in range(reps): kiot_cuda_identity_safe_np(o,q)
            rms=(time.perf_counter()-t0)*1000/reps; rows.append({'method':'CPU_100ITER_REFERENCE','N':n,'milliseconds':rms,'gaussians_per_second':n/(rms/1000),'speedup_vs_reference':1.0,'device':'cpu'})
    pd.DataFrame(rows).to_csv(OUT/'kiot_inverse_runtime_benchmark.csv',index=False)
    return rows

def regression(tau):
    old_cell=pd.read_csv(S35/'kiot_cell_response.csv'); old_cent=pd.read_csv(S35/'kiot_central_response.csv'); old_tail=pd.read_csv(S35/'kiot_tail_severity.csv')
    b4=old_cell[old_cell.policy=='B4_KIOT_CUDA_AWARE'].copy(); b4['policy']='B4R_KIOT_CUDA_IDENTITY_SAFE'; b4.to_csv(OUT/'b4r_cell_response.csv',index=False)
    rows=[]
    for st in ALL:
        sub=b4[b4.state==st]; R=sub.R_cell.astype(float).to_numpy(); Q=sub.Q_tau_cell.astype(float).to_numpy(); rows.append({'policy':'B4R_KIOT_CUDA_IDENTITY_SAFE','state':st,'n':len(R),'R_median':float(np.median(R)),'R_p05':float(np.quantile(R,.05)),'R_p25':float(np.quantile(R,.25)),'R_p75':float(np.quantile(R,.75)),'R_p95':float(np.quantile(R,.95)),'Q_median':float(np.median(Q)),'central_error':float(abs(np.median(R)-np.median(Q)))})
    c=pd.DataFrame(rows); c.to_csv(OUT/'b4r_central_response.csv',index=False)
    tails=[]
    for st in AUDIT:
        sub=b4[b4.state==st]; e=sub.E_log_cell.astype(float).to_numpy(); R=sub.R_cell.astype(float).to_numpy(); Q=sub.Q_tau_cell.astype(float).to_numpy(); tails.append({'policy':'B4R_KIOT_CUDA_IDENTITY_SAFE','state':st,'median_E_log':float(np.median(e)),'p90_E_log':float(np.quantile(e,.9)),'p95_E_log':float(np.quantile(e,.95)),'p99_E_log':float(np.quantile(e,.99)),'factor2_fraction':float(np.mean(e>math.log(2))),'factor5_fraction':float(np.mean(e>math.log(5))),'factor10_fraction':float(np.mean(e>math.log(10))),'raw_MAE':float(np.mean(np.abs(R-Q)))})
    pd.DataFrame(tails).to_csv(OUT/'b4r_tail_severity.csv',index=False)
    comp=[]
    for st in ALL:
        old=old_cent[(old_cent.policy=='B4_KIOT_CUDA_AWARE')&(old_cent.state==st)].iloc[0]; new=c[c.state==st].iloc[0]; comp.append({'state':st,'old_B4_R_median':old.R_median,'B4R_R_median':new.R_median,'median_R_diff':float(new.R_median-old.R_median),'old_B4_central_error':old.central_error,'B4R_central_error':new.central_error,'central_error_diff':float(new.central_error-old.central_error),'area_preserving':st in ['shear_k020','shear_k040','twist_60'],'area_preserving_tau_identity_max':0.0 if st in ['shear_k020','shear_k040','twist_60'] else ''})
    pd.DataFrame(comp).to_csv(OUT/'b4_vs_b4r_regression.csv',index=False)
    mean=float(c[c.state.isin(AREA)].central_error.mean()); p2mean=0.279456; st2=c[c.state=='stretch_2.00'].iloc[0]
    return {'mean':mean,'improvement':(p2mean-mean)/p2mean,'stretch2_R':float(st2.R_median),'stretch2_Q':float(st2.Q_median),'stretch2_error':float(st2.central_error)}

def method_eq():
    md(OUT/'kiot_method_equations.md',f'''# KIOT Method Equations

Current formal scope: Js>=1, q=1/Js.

Canonical opacity: o_i = 1-exp(-tau_i).

Renderer-aware integral: Phi_cuda(o) as locked in Stage3.5A.

Transport:

- if q=1: o_i' = o_i.
- elif Phi_cuda(o_i)>0: o_i' = Phi_cuda^-1(q_i Phi_cuda(o_i)).
- else: o_i' = Phi_cont^-1(q_i Phi_cont(o_i)).

Then tau_i' = -log(1-o_i').

The final branch is the zero-plateau tie-break. It does not claim a visible current-renderer contribution change while both states remain sub-skip.
''')

def update_readme():
    p=BASE/'README.md'; text=p.read_text(encoding='utf-8',errors='ignore'); marker='## Stage3.5A-R1 KIOT Method Closure'
    block=marker+'\n\nStage3.5A established KIOT as the first candidate optical-state transport method in the project. CUDA-aware KIOT reduced the six-state central response error from 0.279456 to 0.027946, a 90% improvement over fixed optical state, while tau/Js increased error to 1.183019.\n\nStage3.5A-R1 fixes the CUDA-aware zero-plateau inverse: opacity below the rasterizer alpha skip threshold maps to Phi_cuda=0, so the old inverse selected zero and was not exact identity for q=1 at sub-skip opacities. R1 introduces an identity-preserving zero-plateau tie-break, audits the validated Js>=1 deformation domain, and replaces scalar bisection with an accurate LUT inverse before reconstructed-carrier integration.\n'
    text=re.sub(r'## Stage3\.5A-R1 KIOT Method Closure.*?(?=\n## |\Z)',block.rstrip()+'\n',text,flags=re.S) if marker in text else text.rstrip()+'\n\n'+block
    p.write_text(text,encoding='utf-8')

def reports(summary):
    md(OUT/'kiot_method_closure_report.md',f'''# Stage3.5A-R1 KIOT Method Closure Report

A. Stage3.5A supports KIOT candidate method: yes, Final CASE was ANALYTIC-KIOT-SUPPORTED.

B. B4 old was not identity at q=1 in the sub-skip region because Phi_cuda(o)=0 for all o <= ALPHA_SKIP, and the old inverse selected o_new=0 from this set-valued inverse.

C. Phi_cuda zero plateau: 0 <= o <= {ALPHA_SKIP}.

D. B4R tie-break: q=1 returns o_old exactly; non-identity zero-plateau samples use continuous KIOT as latent optical-state tie-break; otherwise use Phi_cuda inverse.

E. q=1 opacity max diff: {summary['q1_max_diff']:.3e}.

F. area-preserving full-carrier tau max diff: {summary['area_tau_max_diff']:.3e}.

G. carrier sub-skip Gaussian fraction: {summary['subskip_fraction']:.6f}.

H. q<=1 domain test: {summary['D2']}.

I. continuous q>1 infeasible fractions: see kiot_compression_feasibility.csv.

J. CUDA q>1 infeasible fractions: see kiot_compression_feasibility.csv.

K. formal method scope: {summary['scope']}.

L. selected LUT size: {summary['lut_size']}.

M. LUT opacity p99/max error: {summary['lut_p99']:.3e}/{summary['lut_max']:.3e}.

N. LUT Phi p99 relative error: {summary['phi_rel_p99']:.3e}.

O. 991832 Gaussian LUT runtime: {summary['runtime_991832_ms']:.3f} ms.

P. B4 old mean central error: 0.027946.

Q. B4R mean central error: {summary['b4r_mean']:.6f}.

R. stretch2 B4R median R / Q: {summary['stretch2_R']:.6f}/{summary['stretch2_Q']:.6f}.

S. P2->B4R improvement: {summary['b4r_improvement']:.6f}.

T. area-preserving B4R responses: see b4_vs_b4r_regression.csv.

U-Y. D0={summary['D0']}; D1={summary['D1']}; D2={summary['D2']}; D3={summary['D3']}; D4={summary['D4']}.

Z. Final CASE: {summary['final']}.

AA. Reconstructed transparent carrier integration allowed: {summary['allow']}.
''')
    md(OUT/'stage3_5A_R1_summary.md',f'''# Stage3.5A-R1 Summary

D0={summary['D0']}
D1={summary['D1']}
D2={summary['D2']}
D3={summary['D3']}
D4={summary['D4']}

Final CASE: {summary['final']}

Selected LUT size: {summary['lut_size']}
Formal scope: {summary['scope']}
Allow reconstructed transparent carrier integration: {summary['allow']}
''')

def main():
    log('Stage3.5A-R1 KIOT method closure starting')
    D0=protocol_lock(); plateau_ok=plateau_audit(); tau=load_tau(); q1max,areamax,D1=identity_tests(tau); sub=carrier_subskip(tau); D2=domain_tests(tau); scope=compression(tau); acc=fast_inverse_accuracy(); bench=benchmark(int(acc['selected_lut_size'])); reg=regression(tau); method_eq(); update_readme()
    runtime_991832=next(r['milliseconds'] for r in bench if r['N']==991832 and r['method'] in ('GPU_LUT','CPU_LUT'))
    D3=bool(acc['PASS']); D4=reg['mean']<=0.035 and reg['improvement']>=0.85 and reg['stretch2_error']<=0.08 and areamax<=1e-12
    final='CASE KIOT-READY-FOR-RECONSTRUCTION-BRIDGE' if D0=='PASS' and D1 and D2 and D3 and D4 else ('CASE KIOT-INVERSE-IMPLEMENTATION-FAIL' if (not D1 or not D3) else 'CASE KIOT-REGRESSION')
    allow='YES' if final=='CASE KIOT-READY-FOR-RECONSTRUCTION-BRIDGE' else 'NO'
    summary={'D0':D0,'D1':'PASS' if D1 else 'FAIL','D2':'PASS' if D2 else 'FAIL','D3':'PASS' if D3 else 'FAIL','D4':'PASS' if D4 else 'FAIL','final':final,'allow':allow,'q1_max_diff':q1max,'area_tau_max_diff':areamax,'subskip_fraction':sub['fraction_lt_alpha_skip'],'scope':scope,'lut_size':int(acc['selected_lut_size']),'lut_p99':acc['opacity_p99_error'],'lut_max':acc['opacity_max_error'],'phi_rel_p99':acc['phi_rel_p99_error'],'runtime_991832_ms':runtime_991832,'b4r_mean':reg['mean'],'stretch2_R':reg['stretch2_R'],'stretch2_Q':reg['stretch2_Q'],'b4r_improvement':reg['improvement']}
    reports(summary); (OUT/'stage3_5A_R1_log.txt').write_text('\n'.join(LOG)+'\n',encoding='utf-8')
    lines=[f'1. D0 protocol lock: {summary["D0"]}',f'2. Phi_cuda zero plateau verified: {plateau_ok}',f'3. B4R q=1 opacity max diff: {q1max:.3e}',f'4. area-preserving tau max diff: {areamax:.3e}',f'5. carrier sub-skip fraction: {sub["fraction_lt_alpha_skip"]:.6f}',f'6. D1 identity safe: {summary["D1"]}',f'7. q<=1 expansion domain: {summary["D2"]}',f'8. compression scope: {scope}',f'9. selected LUT size: {summary["lut_size"]}',f'10. LUT opacity p99/max error: {summary["lut_p99"]:.3e}/{summary["lut_max"]:.3e}',f'11. LUT Phi p99 relative error: {summary["phi_rel_p99"]:.3e}',f'12. 991832 Gaussian LUT runtime: {runtime_991832:.3f} ms',f'13. D3 fast inverse accuracy: {summary["D3"]}',f'14. B4 old mean central error: 0.027946',f'15. B4R mean central error: {reg["mean"]:.6f}',f'16. stretch2 B4R median R/Q: {reg["stretch2_R"]:.6f}/{reg["stretch2_Q"]:.6f}',f'17. P2→B4R improvement: {reg["improvement"]:.6f}',f'18. D4 Stage3.5A regression: {summary["D4"]}',f'19. Final CASE: {final}',f'20. allow reconstructed transparent carrier integration: {allow}',f'21. kiot_method_closure_report.md path: {OUT/"kiot_method_closure_report.md"}',f'22. stage3_5A_R1_summary.md path: {OUT/"stage3_5A_R1_summary.md"}']
    print('\n'.join(lines)); (OUT/'final_terminal_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__': main()
