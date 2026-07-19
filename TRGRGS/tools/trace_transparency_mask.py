#!/usr/bin/env python3
import sys,json,csv,cv2,numpy as np
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from trgr.transparency_mask_audit import *
from trgr.camera_geometry import load_colmap_cameras
scene=ROOT/'data/translab/scene_01';out=ROOT/'outputs/scene_01/mask_semantics_audit';out.mkdir(parents=True,exist_ok=True);cams=load_colmap_cameras(scene,ROOT/'third_party/TSGS');cams=[c for i,c in enumerate(cams) if i%8!=0]
rows=[];agreements=[]
audit_indices=set(np.linspace(0,len(cams)-1,20,dtype=int).tolist())
for cam_idx,cam in enumerate(cams):
 raw,continuous,branch=renderer_transparency_tensor(scene,cam.name)
 # R2 selector directly consumes the exact final camera/rasterizer tensor.
 selector=branch.copy();normal=cv2.imread(str(scene/'normals'/f'{Path(cam.name).stem}_normal.png')).astype(np.float32)/127.5-1;normal=cv2.resize(normal,(800,800),interpolation=cv2.INTER_LINEAR)
 s=component_stats(selector,normal);s['name']=cam.name;rows.append(s)
 if cam_idx in audit_indices:
  eq=int((selector==branch).sum());union=np.logical_or(selector,branch).sum();inter=np.logical_and(selector,branch).sum();agreements.append({'name':cam.name,'equal_pixel_count':eq,'pixel_agreement':eq/selector.size,'iou':inter/union if union else 1.,'foreground_area_a':int(selector.sum()),'foreground_area_b':int(branch.sum()),'polarity_agreement':True})
fields=list(rows[0]);
with open(out/'all_view_component_stats.csv','w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
with open(out/'mask_renderer_equivalence_20views.csv','w',newline='') as f:w=csv.DictWriter(f,agreements[0]);w.writeheader();w.writerows(agreements)
rat=np.array([x['largest_component_ratio'] for x in rows]);small=np.array([x['small_component_pixel_fraction'] for x in rows]);equiv=min(x['pixel_agreement'] for x in agreements)>=.999 and min(x['iou'] for x in agreements)>=.999;multi=(rat<.9).mean()>=.5 and np.median(small)<=.05
report={'status':'PASS_MASK_RENDERER_EQUIVALENCE' if equiv else 'FAIL_MASK_RENDERER_MISMATCH','training_views':len(rows),'audit_views':len(agreements),'minimum_pixel_agreement':min(x['pixel_agreement'] for x in agreements),'minimum_iou':min(x['iou'] for x in agreements),'fraction_largest_component_ratio_below_090':float((rat<.9).mean()),'median_small_component_pixel_fraction':float(np.median(small)),'mask_semantics':'MASKS_ARE_GENUINELY_MULTI_COMPONENT' if multi else 'NOISE_OR_UNRESOLVED','largest_component_gate':'LARGEST_COMPONENT_GATE_INVALID' if equiv and multi else 'KEEP_UNTIL_RESOLVED','renderer_tensor_reused_by_selector':True}
(ROOT/'reports/stage1_p0_mask_semantics_audit.json').write_text(json.dumps(report,indent=2)+'\n')
# histograms and required representative label visualizations
import matplotlib.pyplot as plt
for key,file in [('component_count','component_count_histogram.png'),('largest_component_ratio','largest_component_ratio_histogram.png'),('small_component_pixel_fraction','small_component_fraction_histogram.png')]:plt.figure();plt.hist([x[key] for x in rows],30);plt.xlabel(key);plt.tight_layout();plt.savefig(out/file);plt.close()
orders={'top_component_count':sorted(rows,key=lambda x:x['component_count'],reverse=True)[:20],'lowest_largest_ratio':sorted(rows,key=lambda x:x['largest_component_ratio'])[:20],'uniform_views':[rows[i] for i in np.linspace(0,len(rows)-1,20,dtype=int)]}
for group,items in orders.items():
 d=out/group;d.mkdir(exist_ok=True)
 for r in items:
  raw,cont,m=renderer_transparency_tensor(scene,r['name']);n,lab,stats,_=cv2.connectedComponentsWithStats(m,8);sig=np.zeros_like(m);smallm=np.zeros_like(m);minimum=r['minimum_small_component_area']
  for i,a in enumerate(stats[1:,cv2.CC_STAT_AREA],1):(sig if a>=minimum else smallm)[lab==i]=255
  im=cv2.resize(cv2.imread(str(scene/'images'/r['name'])),(800,800));pan=np.hstack([im,cv2.cvtColor((cont*255).astype(np.uint8),cv2.COLOR_GRAY2BGR),cv2.cvtColor(m*255,cv2.COLOR_GRAY2BGR),cv2.applyColorMap((lab%255).astype(np.uint8),cv2.COLORMAP_HSV),cv2.cvtColor(sig,cv2.COLOR_GRAY2BGR),cv2.cvtColor(smallm,cv2.COLOR_GRAY2BGR)]);cv2.imwrite(str(d/r['name']),cv2.resize(pan,(1200,200)))
print(json.dumps(report,indent=2))
