#!/usr/bin/env python3
import sys,json,csv,hashlib,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from trgr.camera_geometry import load_colmap_cameras
from trgr.view_selection import view_metrics,select_probe_views
scene=ROOT/'data/translab/scene_01';cams=load_colmap_cameras(scene,ROOT/'third_party/TSGS');cams=[c for i,c in enumerate(cams) if i%8!=0]
rows=view_metrics(cams,scene);sel,cand,relax=select_probe_views(rows);names=[x['name'] for x in sel]
serial=lambda r:{**r,'center':r['center'].tolist(),'direction':r['direction'].tolist()}
ok=len(names)==8
payload={'status':'FROZEN_GT_FREE' if ok else 'FAIL_INSUFFICIENT_GT_FREE_PROBE_CANDIDATES','probe_ids':names,'candidate_count':len(cand),'required_candidate_count':16,'required_probe_count':8,'training_camera_count':len(cams),'relaxations':relax,'selection_inputs':['training cameras','transparent masks','normals','camera centers','viewing directions'],'gt_used':False}
(ROOT/'outputs/scene_01/probe_views.json').write_text(json.dumps(payload,indent=2)+'\n');(ROOT/'reports/stage1_probe_selection.json').write_text(json.dumps(payload,indent=2)+'\n')
fields=['name','mask_area_ratio','border_touch_ratio','largest_component_ratio','valid_normal_ratio','selected','candidate','center_x','center_y','center_z','direction_x','direction_y','direction_z']
with open(ROOT/'outputs/scene_01/probe_view_metrics.csv','w',newline='') as f:
 w=csv.DictWriter(f,fields);w.writeheader()
 for r in rows:w.writerow({'name':r['name'],'mask_area_ratio':r['mask_area_ratio'],'border_touch_ratio':r['border_touch_ratio'],'largest_component_ratio':r['largest_component_ratio'],'valid_normal_ratio':r['valid_normal_ratio'],'selected':r['name'] in names,'candidate':r in cand,'center_x':r['center'][0],'center_y':r['center'][1],'center_z':r['center'][2],'direction_x':r['direction'][0],'direction_y':r['direction'][1],'direction_z':r['direction'][2]})
ims=[]
for n in names:
 im=cv2.imread(str(scene/'images'/n));im=cv2.resize(im,(320,320));cv2.putText(im,n,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2);ims.append(im)
if ok: montage=np.vstack([np.hstack(ims[:4]),np.hstack(ims[4:])])
else:
 montage=np.full((640,1280,3),245,np.uint8);cv2.putText(montage,'PROBE FREEZE FAILED: only %d strict candidates'%len(cand),(70,300),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,200),3)
cv2.imwrite(str(ROOT/'outputs/scene_01/probe_view_montage.png'),montage)
print(json.dumps(payload,indent=2))
raise SystemExit(0 if ok else 2)
