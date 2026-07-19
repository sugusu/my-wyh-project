import numpy as np
def camera_center(w2c):return -w2c[:3,:3].T@w2c[:3,3]
def viewing_direction(w2c):return w2c[:3,:3].T@np.array([0.,0.,1.])
def directed_view_graph(names,cameras,mask_areas):
 out=[]
 for a in names:
  for b in names:
   if a==b:continue
   ca,cb=cameras[a],cameras[b];da,db=viewing_direction(ca.world_to_camera),viewing_direction(cb.world_to_camera)
   out.append({'source':a,'target':b,'camera_center_distance':float(np.linalg.norm(camera_center(ca.world_to_camera)-camera_center(cb.world_to_camera))),'viewing_direction_angle':float(np.arccos(np.clip(da@db,-1,1))),'relative_transform':(cb.world_to_camera@np.linalg.inv(ca.world_to_camera)).tolist(),'source_mask_area':int(mask_areas[a]),'target_mask_area':int(mask_areas[b])})
 return out
