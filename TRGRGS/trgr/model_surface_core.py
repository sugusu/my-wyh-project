import numpy as np
from .mesh_distance import closest_points_on_mesh

def quaternion_matrices(q):
    q=np.asarray(q,float);q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-20);w,x,y,z=q.T
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],1).reshape(-1,3,3)
def select_model_surface_core(xyz,scales,opacity,rotation,mask_indices,mask_support,eroded_support,model_mesh_path,scene_diameter=.5481526628412927,voxel_size=.002):
    """GT-free: model_mesh_path must be the model's own reconstruction."""
    pts=xyz[mask_indices];near=closest_points_on_mesh(pts,model_mesh_path);R=quaternion_matrices(rotation[mask_indices]);n=near['normals'];s=scales[mask_indices]
    sigma=np.sqrt(np.maximum(np.einsum('ni,nij,nj->n',n,R@np.array([np.diag(x*x) for x in s])@np.transpose(R,(0,2,1)),n),0))
    tol=np.clip(np.maximum(3*voxel_size,2.5*sigma),3*voxel_size,.02*scene_diameter);op=opacity[mask_indices].reshape(-1);mx=s.max(1)
    finite=np.isfinite(np.column_stack([pts,s,op[:,None],near['distance'][:,None],sigma[:,None]])).all(1)
    keep=(eroded_support[mask_indices]>=3)&(near['distance']<=tol)&(op>=.005)&(mx<=.10*scene_diameter)&finite
    metrics={'indices':mask_indices[keep],'mask_support':mask_support[mask_indices][keep],'eroded_mask_support':eroded_support[mask_indices][keep],
      'model_mesh_distance':near['distance'][keep],'adaptive_tolerance':tol[keep],'sigma_normal':sigma[keep],'opacity':op[keep],'max_scale':mx[keep],
      'closest_model_surface_points':near['closest_points'][keep],'closest_model_surface_normals':n[keep]}
    return metrics,{'distance_backend':near['distance_backend'],'mask_cone_count':len(mask_indices),'model_surface_core_count':int(keep.sum()),'nonfinite_candidate_count':int((~finite).sum())}
