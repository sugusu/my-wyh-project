import numpy as np
from trgr.hypothesis_reprojection import pixels_depth_to_world,world_to_pixels
def test_reprojection_roundtrip():
 K=np.array([[100.,0,32],[0,100,32],[0,0,1]]);w=np.eye(4);u=np.array([12.,40.]);v=np.array([22.,50.]);d=np.array([1.,2.]);p=pixels_depth_to_world(u,v,d,K,w);uv,z=world_to_pixels(p,K,w);assert np.allclose(uv,np.stack([u,v],1)) and np.allclose(z,d)
