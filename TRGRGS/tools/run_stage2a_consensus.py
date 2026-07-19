#!/usr/bin/env python3
import csv,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from trgr.camera_geometry import load_colmap_cameras
from trgr.view_graph import directed_view_graph
from trgr.hypothesis_reprojection import pixels_depth_to_world,world_to_pixels,local_set_residual
from trgr.hypothesis_consensus import consensus_score
from trgr.hypothesis_clustering import cluster_pixel
scene=ROOT/'data/translab/scene_01';out=ROOT/'outputs/scene_01/stage2a_consensus';per=out/'per_view';vis=out/'visualizations';summary=out/'summary'
for p in (per,vis,summary):p.mkdir(parents=True,exist_ok=True)
probe=json.load(open(ROOT/'outputs/scene_01/probe_views_r2.json'));names=probe['probe_ids'];assert names==['frame_0096.png','frame_0218.png','frame_0363.png','frame_0319.png','frame_0092.png','frame_0130.png','frame_0204.png','frame_0156.png']
cams={c.name:c for c in load_colmap_cameras(scene,ROOT/'third_party/TSGS')};data={}
for n in names:
 z=np.load(ROOT/'outputs/scene_01/depth_sweep'/f'{Path(n).stem}.npz');data[n]={k:z[k] for k in z.files};data[n]['K']=cams[n].K.copy();data[n]['K'][:2]/=2.
graph=directed_view_graph(names,cams,{n:int(data[n]['transparent_mask'].sum()) for n in names});(out/'view_graph.json').write_text(json.dumps({'directed_pair_count':56,'pairs':graph},indent=2)+'\n')

def evaluate(n,u,v,dep,save_audit=False):
 N=len(dep);valid=np.zeros(N,np.uint8);fw=np.zeros(N,np.uint8);bi=np.zeros(N,np.uint8);res=np.full((7,N),np.nan,np.float32);cp=np.full((7,N),np.nan,np.float32);cd=np.full((7,N),np.nan,np.float32);audit=[];s=cams[n];K=data[n]['K'];world=pixels_depth_to_world(u,v,dep,K,s.world_to_camera);ti=0
 for target in names:
  if target==n:continue
  t=cams[target];uv,z=world_to_pixels(world,data[target]['K'],t.world_to_camera);h,w=data[target]['transparent_mask'].shape
  initial=(z>0)&np.isfinite(z)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
  pair_valid=0
  for st in range(0,N,4096):
   sl=slice(st,min(st+4096,N));rr,bp,bt,bd=local_set_residual(uv[sl],z[sl],data[target]['depths'],data[target]['transparent_mask'].astype(bool));idx=np.arange(st,min(st+4096,N));ok=initial[sl]&np.isfinite(rr);valid[idx[ok]]+=1;pair_valid+=int(ok.sum());res[ti,idx[ok]]=rr[ok]
   sup=ok&(rr<=.005);fw[idx[sup]]+=1
   if sup.any():
    jj=idx[sup];wp=pixels_depth_to_world(bp[sup,0],bp[sup,1],bd[sup],data[target]['K'],t.world_to_camera);back,bz=world_to_pixels(wp,K,s.world_to_camera);pe=np.linalg.norm(back-np.stack([u[jj],v[jj]],1),axis=1);de=np.abs(bz-dep[jj]);cp[ti,jj]=pe;cd[ti,jj]=de;good=(pe<=1.5)&(de<=.005);bi[jj[good]]+=1
    if save_audit and len(audit)<20:
     for q in range(min(20-len(audit),len(jj))):audit.append({'source':n,'source_pixel':[int(u[jj[q]]),int(v[jj[q]])],'source_depth':float(dep[jj[q]]),'world_point':wp[q].tolist(),'target':target,'target_projected_pixel':uv[jj[q]].tolist(),'target_projected_depth':float(z[jj[q]]),'target_pixel':bp[sup][q].tolist(),'target_tau_index':int(bt[sup][q]),'target_depth':float(bd[sup][q]),'residual':float(rr[sup][q])})
   
  graph[next(i for i,x in enumerate(graph) if x['source']==n and x['target']==target)]['projected_valid_ratio']=pair_valid/max(N,1);ti+=1
 med=np.zeros(N,np.float32);p90=np.zeros(N,np.float32);mcp=np.zeros(N,np.float32);mcd=np.zeros(N,np.float32)
 for i in range(N):
  q=res[:,i];q=q[np.isfinite(q)];med[i]=np.median(q) if len(q) else 1e6;p90[i]=np.quantile(q,.9) if len(q) else 1e6;q=cp[:,i];q=q[np.isfinite(q)];mcp[i]=np.median(q) if len(q) else 1e6;q=cd[:,i];q=q[np.isfinite(q)];mcd[i]=np.median(q) if len(q) else 1e6
 score,ratio=consensus_score(valid,bi,med)
 return {'valid':valid,'forward':fw,'bidir':bi,'median_residual':med,'p90_residual':p90,'median_cycle_pixel':mcp,'median_cycle_depth':mcd,'score':score,'ratio':ratio},audit

def run_once(write=False):
 results={};audits=[];baseline_res=[];top1_res=[];set_res=[];base_ratio=[];top_ratio=[];all_scores=[];counts=[];taus=[];method_total=selected_total=formal_total=unresolved=0
 for n in names:
  d=data[n];method=d['transparent_mask'].astype(bool)&(d['num_valid_hypotheses']>=3);ys,xs=np.nonzero(method);depth=d['depths'][:,ys,xs];tau,j=np.nonzero(np.isfinite(depth)&(depth>0));dep=depth[tau,j];px=xs[j];py=ys[j];met,audit=evaluate(n,px,py,dep,write);audits.extend(audit)
  shape=method.shape;fields={k:np.zeros((6,*shape),a.dtype) for k,a in met.items()}
  for k,a in met.items():fields[k][tau,py,px]=a
  bmet,_=evaluate(n,xs,ys,d['baseline_default_depth'][ys,xs].astype(np.float32));topd=np.zeros(shape,np.float32);tops=np.zeros(shape,np.float32);top2d=np.zeros(shape,np.float32);top2s=np.zeros(shape,np.float32);cnt=np.zeros(shape,np.uint8);unres=np.zeros(shape,np.uint8);bestres=np.zeros(shape,np.float32);bestratio=np.zeros(shape,np.float32);toptau=np.full(shape,-1,np.int8)
  for j,(y,x) in enumerate(zip(ys,xs)):
   clusters=cluster_pixel(d['depths'][:,y,x],fields['score'][:,y,x],fields['ratio'][:,y,x],fields['valid'][:,y,x],fields['median_residual'][:,y,x]);formal_total+=int(bool(clusters));keep=[c for c in sorted(clusters,key=lambda c:(-c['max_score'],c['representative_tau'])) if c['max_score']>=.5 and c['support_ratio']>=.5][:2];cnt[y,x]=len(keep);selected_total+=int(bool(keep));method_total+=1
   if keep:
    q=keep[0];topd[y,x]=q['depth'];tops[y,x]=q['max_score'];toptau[y,x]=q['representative_tau'];bestres[y,x]=q['residual'];bestratio[y,x]=q['support_ratio'];top1_res.append(q['residual']);top_ratio.append(q['support_ratio']);taus.append(q['representative_tau'])
    if len(keep)>1:top2d[y,x]=keep[1]['depth'];top2s[y,x]=keep[1]['max_score']
    set_res.append(min(c['residual'] for c in keep))
   else:unres[y,x]=1;unresolved+=1
   baseline_res.append(bmet['median_residual'][j]);base_ratio.append(bmet['ratio'][j])
  all_scores.extend(tops[tops>0]);counts.extend(cnt[method]);payload={'top1_consensus_depth':topd,'top1_consensus_score':tops,'top2_consensus_depth':top2d,'top2_consensus_score':top2s,'selected_cluster_count':cnt,'unresolved_mask':unres,'best_forward_residual':bestres,'best_bidirectional_support_ratio':bestratio,'selected_tau':toptau,**{f'candidate_{k}':v for k,v in fields.items()}};results[n]=payload
  if write:
   np.savez_compressed(per/f'{Path(n).stem}.npz',**payload)
   def col(a):
    ok=np.isfinite(a)&(a>0);o=np.zeros((*a.shape,3),np.uint8)
    if ok.any():lo,hi=np.quantile(a[ok],[.02,.98]);o=cv2.applyColorMap((np.clip((a-lo)/(hi-lo+1e-8),0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO);o[~ok]=0
    return o
   tiles=[col(d['baseline_default_depth'])]+[col(x) for x in d['depths']]+[col(topd),col(top2d),col(cnt),col(bestratio),col(tops),cv2.cvtColor(unres*255,cv2.COLOR_GRAY2BGR),col(bestres)];cv2.imwrite(str(vis/f'{Path(n).stem}_consensus_grid.png'),np.vstack([np.hstack(tiles[:7]),np.hstack(tiles[7:14])]))
 B=np.asarray(baseline_res);T=np.asarray(top1_res);S=np.asarray(set_res);scores=np.asarray(all_scores);C=np.asarray(counts);taua=np.asarray(taus);metrics={'method_valid_pixels':method_total,'formal_candidate_pixel_ratio':formal_total/method_total,'consensus_pixel_ratio':selected_total/method_total,'unresolved_ratio':unresolved/method_total,'selected_cluster_count_mean':float(C.mean()),'consensus_score_std':float(scores.std()),'dominant_tau_ratio':float(max(np.mean(taua==i) for i in range(6))),'baseline_median_forward_residual':float(np.median(B)),'top1_median_forward_residual':float(np.median(T)),'selected_set_median_forward_residual':float(np.median(S)),'baseline_bidirectional_support_ratio':float(np.mean(base_ratio)),'top1_bidirectional_support_ratio':float(np.mean(top_ratio))}
 return results,metrics,audits
r1,m,aud=run_once(True);r2,_,_=run_once(False);maxdiff=0.;same=True;nonfinite=0
for n in names:
 for k in r1[n]:
  a,b=r1[n][k],r2[n][k];same&=np.array_equal(a>0,b>0);maxdiff=max(maxdiff,float(np.max(np.abs(a.astype(float)-b.astype(float)))));nonfinite+=int((~np.isfinite(a)).sum())
failed=[]
if not same or maxdiff>1e-6:failed.append('FAIL_NONDETERMINISTIC')
if nonfinite:failed.append('FAIL_NONFINITE')
if m['formal_candidate_pixel_ratio']<.9 or m['consensus_pixel_ratio']<.8:failed.append('FAIL_REPROJECTION_COVERAGE')
if not (1<=m['selected_cluster_count_mean']<=2) or m['dominant_tau_ratio']>.95 or m['consensus_score_std']<=1e-4 or m['unresolved_ratio']>=.2:failed.append('FAIL_CONSENSUS_COLLAPSE')
imp_top=1-m['top1_median_forward_residual']/m['baseline_median_forward_residual'];imp_set=1-m['selected_set_median_forward_residual']/m['baseline_median_forward_residual'];imp_sup=m['top1_bidirectional_support_ratio']-m['baseline_bidirectional_support_ratio']
if not (imp_top>=.1 or imp_set>=.15 or imp_sup>=.1):failed.append('FAIL_CROSS_VIEW_IMPROVEMENT')
status='PASS_GT_FREE_CROSS_VIEW_CONSENSUS' if not failed else failed[0];report={'status':status,'failed_checks':failed,'gt_read':False,'parameters':{'support_tolerance':.005,'cycle_pixel_tolerance':1.5,'cycle_depth_tolerance':.005,'cluster_merge_tolerance':.0025,'minimum_valid_targets':3,'score_formula':'(bidirectional_support_count/valid_target_count)*exp(-median_forward_residual/0.005)'},'repeatability':{'selected_masks_identical':same,'max_absolute_difference':maxdiff,'nonfinite_count':nonfinite},'metrics':m,'improvement':{'top1_median_ratio':imp_top,'selected_set_median_ratio':imp_set,'bidirectional_support_absolute':imp_sup},'probe_ids':names}
(ROOT/'reports/stage2a_gt_free_consensus.json').write_text(json.dumps(report,indent=2)+'\n');(out/'view_graph.json').write_text(json.dumps({'directed_pair_count':56,'pairs':graph},indent=2)+'\n');(summary/'audit_candidates.json').write_text(json.dumps(aud[:1000],indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if status.startswith('PASS') else 2)
