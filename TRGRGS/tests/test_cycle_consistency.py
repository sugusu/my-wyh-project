import numpy as np
from trgr.hypothesis_reprojection import pixels_depth_to_world,world_to_pixels
def test_two_camera_cycle():
 K=np.array([[80.,0,32],[0,80,32],[0,0,1]]);a=np.eye(4);b=np.eye(4);b[0,3]=.1;p=pixels_depth_to_world(np.array([32.]),np.array([32.]),np.array([2.]),K,a);uv,z=world_to_pixels(p,K,b);q=pixels_depth_to_world(uv[:,0],uv[:,1],z,K,b);back,d=world_to_pixels(q,K,a);assert np.allclose(back,[ [32,32] ]) and np.allclose(d,2)
