#!/usr/bin/env python3
import sys,csv,json,numpy as np,trimesh
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from trgr.evaluation_protocol_audit import *
from trgr.mesh_distance import closest_points_on_triangles
import matplotlib.pyplot as plt
def surface_centroid(mesh):
    v=np.asarray(mesh.vertices);t=v[np.asarray(mesh.faces)];a=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)/2
    return np.average(t.mean(1),axis=0,weights=a).tolist() if a.sum()>0 else v.mean(0).tolist()
pred_path=ROOT/'checkpoints/baseline/scene_01/mesh/tsdf_fusion_post_30000.ply';raw_path=ROOT/'checkpoints/baseline/scene_01/mesh/tsdf_fusion_30000.ply';gt_path=ROOT/'data/translab/scene_01/meshes/scene_mesh.obj';out=ROOT/'outputs/scene_01/protocol_audit';out.mkdir(parents=True,exist_ok=True)
pred=trimesh.load(pred_path,force='mesh',process=False);pv,pf=np.asarray(pred.vertices),np.asarray(pred.faces);gt_scene=trimesh.load(gt_path,force='scene',process=False)
allparts=list(gt_scene.geometry.values());full=trimesh.util.concatenate(allparts);gv,gf=np.asarray(full.vertices),np.asarray(full.faces)
transparts=[g for g in allparts if getattr(getattr(g.visual,'material',None),'name',None)=='Material.011'];tm=trimesh.util.concatenate(transparts);tv,tf=np.asarray(tm.vertices),np.asarray(tm.faces)
ps=sample_surface(pv,pf,500000,42);gs=sample_surface(gv,gf,500000,42);ts=sample_surface(tv,tf,500000,42);pb=colmap_to_blender(ps)
protocols=[('A_raw_prediction_raw_full_gt',ps,gs,'full', 'none','none',False,False,False),('B_r2_transform_material011',ps,blender_to_colmap(ts),'Material.011','none','Blender_to_COLMAP',False,False,False),('C_official_transform_no_crop',pb,gs,'full','COLMAP_to_Blender','none',False,True,False),('D_official_complete',pb[pb[:,1]>-.1],gs[gs[:,1]>-.1],'full','COLMAP_to_Blender','none',True,True,False),('E_official_prediction_material011_gt',pb,ts,'Material.011','COLMAP_to_Blender','none',False,True,False),('F_official_evaluator_full_gt',pb,gs,'full','COLMAP_to_Blender','none',False,True,False)]
rows=[]
for name,a,b,mat,pt,gttr,crop,clean,obs in protocols:
 d2,d1,pr,re,f1=nearest_sample_metrics(a,b);rows.append({'protocol_name':name,'prediction_path':str(pred_path),'gt_path':str(gt_path),'gt_material_filter':mat,'prediction_transform':pt,'gt_transform':gttr,'crop_enabled':crop,'cleaning_enabled':clean,'observation_filter_enabled':obs,'prediction_sample_count':len(a),'gt_sample_count':len(b),'d2s_mean':d2['mean'],'d2s_median':d2['median'],'d2s_p75':d2['p75'],'d2s_p90':d2['p90'],'d2s_p95':d2['p95'],'s2d_mean':d1['mean'],'s2d_median':d1['median'],'s2d_p75':d1['p75'],'s2d_p90':d1['p90'],'s2d_p95':d1['p95'],'symmetric_mean':(d2['mean']+d1['mean'])/2,'precision':pr,'recall':re,'f1':f1,'prediction_bbox_diagonal':np.linalg.norm(np.ptp(a,axis=0)),'gt_bbox_diagonal':np.linalg.norm(np.ptp(b,axis=0))})
with open(ROOT/'reports/stage06_r3_protocol_comparison.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
# GT material audit: the OBJ loader exposes each legal material-defined geometry.
mr=[]
for g in allparts:
 v=np.asarray(g.vertices);name=getattr(getattr(g.visual,'material',None),'name',None) or 'none';mr.append({'material_name':name,'triangle_count':len(g.faces),'surface_area':g.area,'bbox_min':v.min(0).tolist(),'bbox_max':v.max(0).tolist(),'centroid':surface_centroid(g),'centroid_definition':'area_weighted_surface' if g.area>0 else 'degenerate_zero_area_vertex_mean','evaluated_definition':'Material.011' if name=='Material.011' else ('full_GT_member' if name else 'geometry_without_material')})
with open(ROOT/'reports/stage06_r3_gt_target_audit.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=mr[0]);w.writeheader();w.writerows(mr)
# Connected components, selected only by area; evaluator retains all components already present in post mesh.
comps=pred.split(only_watertight=False);comps=sorted(comps,key=lambda x:x.area,reverse=True);cr=[];fig,axs=plt.subplots(4,5,figsize=(15,12))
for i,c in enumerate(comps):
 v=np.asarray(c.vertices);sample=sample_surface(v,np.asarray(c.faces),min(10000,max(1000,len(c.faces)*2)),42+i);sb=colmap_to_blender(sample);near=closest_points_on_triangles(sb,gv,gf)['distance'];cr.append({'component_id':i,'vertex_count':len(v),'triangle_count':len(c.faces),'surface_area':c.area,'bbox_min':v.min(0).tolist(),'bbox_max':v.max(0).tolist(),'centroid':surface_centroid(c),'median_distance_to_evaluator_gt':float(np.median(near)),'retained_by_official_evaluator':True})
 if i<20:
  c.export(out/f'prediction_component_{i:02d}.ply');ax=axs.flat[i];ax.scatter(v[::max(1,len(v)//3000),0],v[::max(1,len(v)//3000),2],s=.2);ax.set_title(f'#{i} area={c.area:.3g}');ax.axis('equal');ax.axis('off')
for ax in axs.flat[len(comps):]:ax.axis('off')
fig.tight_layout();fig.savefig(out/'prediction_components_montage.png',dpi=140);plt.close(fig)
with open(out/'prediction_components.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=cr[0]);w.writeheader();w.writerows(cr)
summary={'component_count':len(comps),'raw_full_prediction':mesh_info(raw_path),'post_clean_prediction':mesh_info(pred_path),'selection_variants':{'raw_full_prediction':str(raw_path),'largest_component_ids':[0],'cumulative_area_95_component_ids':list(range(next(i for i,x in enumerate(np.cumsum([c.area for c in comps])/sum(c.area for c in comps)) if x>=.95)+1)),'official_cleaning_retained_post_components':list(range(len(comps))),'official_crop_rule':'sampled transformed prediction points with y > -0.1'}}
(out/'prediction_component_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
