import numpy as np,trimesh
from pathlib import Path
from .mesh_distance import closest_points_on_triangles

def load_transparent_gt(path):
    scene=trimesh.load(path,force='scene',process=False);mtl=Path(path).with_suffix('.mtl').read_text().splitlines();names=set();cur=None
    for line in mtl:
        f=line.split()
        if f[:1]==['newmtl']:cur=f[1]
        elif f[:1]==['d'] and cur and float(f[1])<.999:names.add(cur)
    mesh=trimesh.util.concatenate([g for g in scene.geometry.values() if getattr(getattr(g.visual,'material',None),'name',None) in names]);v=np.asarray(mesh.vertices)[:,[0,2,1]].copy();v[:,1]*=-1
    return v,np.asarray(mesh.faces),sorted(names)
def area_centroid(v,f):
    tri=v[f];a=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2;c=tri.mean(1);return np.average(c,axis=0,weights=a)
def audit_core_r2(xyz,indices,opacity,support,gt_mesh_path,scene_diameter=.5481526628412927):
    pts=xyz[indices];v,f,mats=load_transparent_gt(gt_mesh_path);near=closest_points_on_triangles(pts,v,f);d=near['distance'];cent=area_centroid(v,f);q1,q99=np.quantile(pts,[.01,.99],axis=0);q5,q95=np.quantile(pts,[.05,.95],axis=0);gmin,gmax=v.min(0),v.max(0)
    raw=pts.mean(0);ow=np.average(pts,axis=0,weights=opacity[indices].reshape(-1));sw=np.average(pts,axis=0,weights=np.maximum(support[indices],1));ratio=np.linalg.norm(q99-q1)/np.linalg.norm(gmax-gmin)
    absstats={'mean':d.mean(),'median':np.median(d),'p75':np.quantile(d,.75),'p90':np.quantile(d,.9),'p95':np.quantile(d,.95),'maximum':d.max()};rel={k:float(x/scene_diameter) for k,x in absstats.items()}
    fail=[]
    vals=np.r_[pts.ravel(),d,raw,ow,sw,q1,q99,q5,q95,gmin,gmax,ratio]
    if len(indices)<1000:fail.append('FAIL_CORE_TOO_SMALL')
    if rel['median']>.02:fail.append('FAIL_MEDIAN_SURFACE_DISTANCE')
    if rel['p90']>.05:fail.append('FAIL_P90_SURFACE_DISTANCE')
    if not .65<=ratio<=1.50:fail.append('FAIL_ROBUST_BBOX_SCALE')
    if not np.isfinite(vals).all():fail.append('FAIL_NONFINITE')
    return {'status':'PASS' if not fail else 'FAIL','failed_checks':fail,'model_surface_core_count':len(indices),'gt_distance_backend':near['distance_backend'],'gt_materials':mats,'gt_surface_distance':{k:float(x) for k,x in absstats.items()},'gt_surface_distance_relative':rel,'nonfinite_count':int((~np.isfinite(vals)).sum()),'raw_center':raw.tolist(),'opacity_weighted_center':ow.tolist(),'support_weighted_center':sw.tolist(),'gt_area_weighted_surface_centroid':cent.tolist(),'center_distances_to_gt_centroid':{'raw':float(np.linalg.norm(raw-cent)),'opacity_weighted':float(np.linalg.norm(ow-cent)),'support_weighted':float(np.linalg.norm(sw-cent))},'raw_bbox':[pts.min(0).tolist(),pts.max(0).tolist()],'quantile_1_99_bbox':[q1.tolist(),q99.tolist()],'quantile_5_95_bbox':[q5.tolist(),q95.tolist()],'gt_mesh_bbox':[gmin.tolist(),gmax.tolist()],'robust_bbox_diagonal_ratio':float(ratio),'scene_diameter':scene_diameter}
