import numpy as np,open3d as o3d

def closest_points_on_mesh(points,mesh_path):
    """Exact Open3D point-to-triangle closest points, never nearest vertices."""
    legacy=o3d.io.read_triangle_mesh(str(mesh_path)); mesh=o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(mesh)
    q=o3d.core.Tensor(np.asarray(points,dtype=np.float32));ans=scene.compute_closest_points(q)
    cp=ans['points'].numpy();pid=ans['primitive_ids'].numpy().astype(np.int64);uv=ans['primitive_uvs'].numpy()
    tri=np.asarray(legacy.triangles)[pid];v=np.asarray(legacy.vertices);e1=v[tri[:,1]]-v[tri[:,0]];e2=v[tri[:,2]]-v[tri[:,0]]
    normals=np.cross(e1,e2);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
    return {'closest_points':cp,'primitive_ids':pid,'primitive_uvs':uv,'normals':normals,'distance':np.linalg.norm(np.asarray(points)-cp,axis=1),'distance_backend':'open3d_raycasting_scene_exact'}

def closest_points_on_triangles(points,vertices,triangles):
    legacy=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(triangles))
    mesh=o3d.t.geometry.TriangleMesh.from_legacy(legacy);scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(mesh)
    ans=scene.compute_closest_points(o3d.core.Tensor(np.asarray(points,dtype=np.float32)));cp=ans['points'].numpy();pid=ans['primitive_ids'].numpy().astype(np.int64);uv=ans['primitive_uvs'].numpy();tri=np.asarray(triangles)[pid]
    e1=vertices[tri[:,1]]-vertices[tri[:,0]];e2=vertices[tri[:,2]]-vertices[tri[:,0]];n=np.cross(e1,e2);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-20)
    return {'closest_points':cp,'primitive_ids':pid,'primitive_uvs':uv,'normals':n,'distance':np.linalg.norm(np.asarray(points)-cp,axis=1),'distance_backend':'open3d_raycasting_scene_exact'}
