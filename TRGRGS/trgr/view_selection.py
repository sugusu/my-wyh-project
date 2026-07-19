from pathlib import Path
import cv2,numpy as np

def camera_center_direction(cam):
    c=np.linalg.inv(cam.world_to_camera)[:3,3]; d=cam.world_to_camera[:3,:3].T@np.array([0.,0.,1.]);d/=np.linalg.norm(d)
    return c,d
def view_metrics(cameras,scene,normal_folder='normals'):
    rows=[];scene=Path(scene)
    for cam in cameras:
        m=cv2.imread(str(scene/'transparent_masks'/Path(cam.name).with_suffix('.png')),0)>127; area=m.mean(); total=max(m.sum(),1)
        border=(m[0].sum()+m[-1].sum()+m[:,0].sum()+m[:,-1].sum())/total
        nlab,lab,stats,_=cv2.connectedComponentsWithStats(m.astype(np.uint8),8); largest=stats[1:,cv2.CC_STAT_AREA].max() if nlab>1 else 0
        n=cv2.imread(str(scene/normal_folder/f'{Path(cam.name).stem}_normal.png'))[::4,::4].astype(np.int16)
        valid=float((((n-128)**2).sum(2)>25).mean())
        c,d=camera_center_direction(cam);rows.append({'name':cam.name,'mask_area_ratio':float(area),'border_touch_ratio':float(border),'largest_component_ratio':float(largest/total),'valid_normal_ratio':float(valid),'center':c,'direction':d})
    return rows
def select_probe_views(rows,count=8):
    areas=np.array([r['mask_area_ratio'] for r in rows]); relax=[]
    settings=[(.10,.90,.05,.80),(.05,.95,.05,.80),(.05,.95,.10,.80),(.05,.95,.10,.70)]
    cand=[]
    for si,(lo,hi,b,n) in enumerate(settings):
        qlo,qhi=np.quantile(areas,[lo,hi]);cand=[r for r in rows if qlo<=r['mask_area_ratio']<=qhi and r['border_touch_ratio']<=b and r['largest_component_ratio']>=.90 and r['valid_normal_ratio']>=n]
        relax.append({'step':si,'is_relaxation':si>0,'area_percentiles':[lo,hi],'border_max':b,'normal_min':n,'candidate_count':len(cand)})
        if len(cand)>=16:break
    if len(cand)<count:return [],cand,relax
    C=np.stack([r['center'] for r in cand]);D=np.stack([r['direction'] for r in cand]); cd=np.linalg.norm(C[:,None]-C[None],axis=2);cd/=max(cd.max(),1e-12);ad=np.arccos(np.clip(D@D.T,-1,1))/np.pi;dist=.5*cd+.5*ad
    sel=[int(np.argmax(dist.mean(1)))]
    while len(sel)<count:sel.append(int(np.argmax(dist[:,sel].min(1))))
    return [cand[i] for i in sel],cand,relax
