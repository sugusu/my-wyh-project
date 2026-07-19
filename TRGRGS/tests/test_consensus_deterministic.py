import numpy as np
from trgr.hypothesis_clustering import cluster_pixel
def test_cluster_tie_uses_low_tau():
 c=cluster_pixel(np.array([1.,1.001,2,0,0,0]),np.array([.8,.8,.5,0,0,0]),np.ones(6),np.array([3,3,3,0,0,0]),np.zeros(6));assert c[0]['representative_tau']==0
