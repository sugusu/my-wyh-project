import numpy as np
from trgr.mesh_distance import closest_points_on_triangles
def test_exact_triangle_distance_not_vertex_distance():
 v=np.array([[0.,0,0],[1,0,0],[0,1,0]]);f=np.array([[0,1,2]]);p=np.array([[.25,.25,2.]])
 r=closest_points_on_triangles(p,v,f);assert np.allclose(r['closest_points'],[[.25,.25,0]],atol=1e-6);assert np.allclose(r['distance'],[2.])

