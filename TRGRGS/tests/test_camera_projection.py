import numpy as np
from trgr.camera_geometry import Camera, roundtrip

def test_projection_unprojection_roundtrip():
    theta=.37; R=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    W=np.eye(4); W[:3,:3]=R; W[:3,3]=[.1,-.2,2]
    cam=Camera('x.png',1600,1600,np.array([[2222.,0,800],[0,2222.,800],[0,0,1.]]),W)
    rng=np.random.default_rng(42); pts=rng.normal(size=(1000,3))
    assert np.max(np.linalg.norm(roundtrip(pts,cam)-pts,axis=1)) < 1e-10

