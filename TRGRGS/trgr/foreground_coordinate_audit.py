from pathlib import Path
import json,numpy as np,trimesh
from scipy.spatial import cKDTree

def audit_foreground(xyz,indices,gt_mesh_path):
    fg=xyz[indices]; scene=trimesh.load(gt_mesh_path,force='scene',process=False)
    mtl=Path(gt_mesh_path).with_suffix('.mtl').read_text().splitlines(); names=set(); cur=None
    for line in mtl:
        f=line.split()
        if f[:1]==['newmtl']: cur=f[1]
        elif f[:1]==['d'] and cur and float(f[1])<.999:names.add(cur)
    parts=[g for g in scene.geometry.values() if getattr(getattr(g.visual,'material',None),'name',None) in names]
    mesh=trimesh.util.concatenate(parts); gt=np.asarray(mesh.vertices)[:,[0,2,1]].copy(); gt[:,1]*=-1
    center=fg.mean(0); mc=(gt.min(0)+gt.max(0))/2; md=np.linalg.norm(np.ptp(gt,axis=0)); fd=np.linalg.norm(np.ptp(fg,axis=0))
    dist=cKDTree(gt).query(fg,k=1,workers=-1)[0]
    d={'foreground_gaussian_count':len(fg),'all_gaussian_center':xyz.mean(0).tolist(),'foreground_center':center.tolist(),
       'transformed_gt_mesh_center':mc.tolist(),'mesh_diameter':md,'foreground_center_to_mesh_center_distance':float(np.linalg.norm(center-mc)),
       'center_offset_ratio':float(np.linalg.norm(center-mc)/md),'foreground_bbox_min':fg.min(0).tolist(),'foreground_bbox_max':fg.max(0).tolist(),
       'mesh_bbox_min':gt.min(0).tolist(),'mesh_bbox_max':gt.max(0).tolist(),'foreground_bbox_diagonal':fd,'mesh_bbox_diagonal':md,
       'bbox_diagonal_ratio':float(fd/md),'nearest_vertex_distance':{'median':float(np.median(dist)),'p90':float(np.quantile(dist,.9)),'p95':float(np.quantile(dist,.95))}}
    vals=np.array([x for x in _numbers(d)],float); finite=np.isfinite(vals).all()
    if not finite:d['status']='FAIL_NONFINITE'
    elif len(fg)<=1000:d['status']='FAIL_FOREGROUND_EMPTY'
    elif d['center_offset_ratio']>.10:d['status']='FAIL_CENTER_OFFSET'
    elif not .75<=d['bbox_diagonal_ratio']<=1.30:d['status']='FAIL_BBOX_SCALE'
    else:d['status']='PASS'
    return d
def _numbers(x):
    if isinstance(x,dict):
        for v in x.values():yield from _numbers(v)
    elif isinstance(x,(list,tuple)):
        for v in x:yield from _numbers(v)
    elif isinstance(x,(int,float,np.number)):yield x

