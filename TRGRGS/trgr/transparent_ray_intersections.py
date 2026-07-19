"""Exact transparent-mesh ray intersection utilities for post-method diagnostics only."""
import numpy as np,open3d as o3d

def deduplicate_sorted_depths(depths,tolerance):
    d=np.sort(np.asarray(depths,float));
    if not len(d):return d
    groups=[[d[0]]]
    for x in d[1:]:
        if x-groups[-1][-1]<=tolerance:groups[-1].append(x)
        else:groups.append([x])
    return np.array([np.mean(g) for g in groups])

def list_positive_intersections(vertices,triangles,origins,directions,w2c,tolerance):
    mesh=o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices.astype(np.float32)),o3d.core.Tensor(triangles.astype(np.int32)));scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(mesh)
    rays=np.c_[origins,directions].astype(np.float32);ans=scene.list_intersections(o3d.core.Tensor(rays));rid=ans['ray_ids'].numpy();t=ans['t_hit'].numpy();points=origins[rid]+directions[rid]*t[:,None];z=points@w2c[:3,:3].T+w2c[:3,3];z=z[:,2]
    values=[];offset=[0]
    for i in range(len(origins)):
        q=deduplicate_sorted_depths(z[(rid==i)&(z>0)&np.isfinite(z)],tolerance);values.extend(q);offset.append(len(values))
    return np.asarray(values,np.float32),np.asarray(offset,np.int64)
