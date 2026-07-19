#!/usr/bin/env python3
import sys,json,csv,cv2,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from trgr.camera_geometry import load_colmap_cameras
from trgr.view_selection import camera_center_direction
from trgr.view_selection_r2 import select_probe_views_r2
stats={r['name']:r for r in csv.DictReader(open(ROOT/'outputs/scene_01/mask_semantics_audit/all_view_component_stats.csv'))};cams=load_colmap_cameras(ROOT/'data/translab/scene_01',ROOT/'third_party/TSGS');cams=[c for i,c in enumerate(cams) if i%8!=0];rows=[]
for i,c in enumerate(cams):
 r=stats[c.name];center,direction=camera_center_direction(c);rows.append({'name':c.name,'camera_index':i,'center':center,'direction':direction,**{k:int(r[k]) if k in ('mask_pixel_count','component_count') else float(r[k]) for k in ('mask_pixel_count','mask_area_ratio','component_count','largest_component_ratio','small_component_pixel_fraction','border_touch_ratio','valid_normal_ratio_inside_mask')}})
sel,cand,hist,status=select_probe_views_r2(rows);names=[r['name'] for r in sel];payload={'status':status,'probe_ids':names,'candidate_count':len(cand),'history':hist,'seed':42,'gt_used':False,'largest_component_is_diagnostic_only':True,'frozen':len(names)==8}
(ROOT/'outputs/scene_01/probe_views_r2.json').write_text(json.dumps(payload,indent=2)+'\n');(ROOT/'reports/stage1_probe_selection_r2.json').write_text(json.dumps(payload,indent=2)+'\n')
fields=['name','camera_index','mask_pixel_count','mask_area_ratio','component_count','largest_component_ratio','small_component_pixel_fraction','border_touch_ratio','valid_normal_ratio_inside_mask','candidate','selected']
with open(ROOT/'outputs/scene_01/probe_view_metrics_r2.csv','w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();[w.writerow({k:(r['name'] in names if k=='selected' else r in cand if k=='candidate' else r[k]) for k in fields}) for r in rows]
ims=[]
for n in names:
 im=cv2.resize(cv2.imread(str(ROOT/'data/translab/scene_01/images'/n)),(320,320));cv2.putText(im,n,(8,25),0,.6,(0,0,255),2);ims.append(im)
if len(ims)==8:cv2.imwrite(str(ROOT/'outputs/scene_01/probe_view_montage_r2.png'),np.vstack([np.hstack(ims[:4]),np.hstack(ims[4:])]))
print(json.dumps(payload,indent=2));raise SystemExit(0 if len(names)==8 else 2)

