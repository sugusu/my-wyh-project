from pathlib import Path
import cv2, numpy as np
from plyfile import PlyData, PlyElement
from .camera_geometry import world_to_camera, project

def load_gaussian_xyz(ply_path):
    v=PlyData.read(str(ply_path))['vertex'].data
    return np.column_stack((v['x'],v['y'],v['z'])).astype(np.float64)

def select_foreground_gaussians(xyz, cameras, mask_dir, min_support=3, chunk_size=8192):
    """GT-free selection. Intentionally accepts no mesh argument."""
    xyz=np.asarray(xyz); support=np.zeros(len(xyz),np.uint16); mask_dir=Path(mask_dir)
    for cam in cameras:
        mask=cv2.imread(str(mask_dir/Path(cam.name).with_suffix('.png')),cv2.IMREAD_GRAYSCALE)
        for lo in range(0,len(xyz),chunk_size):
            hi=min(lo+chunk_size,len(xyz)); pc=world_to_camera(xyz[lo:hi],cam.world_to_camera); uv,z=project(pc,cam.K)
            finite=np.isfinite(uv).all(1)&np.isfinite(z)&(z>0)
            x=np.rint(uv[:,0]).astype(np.int64); y=np.rint(uv[:,1]).astype(np.int64)
            valid=finite&(x>=2)&(x<cam.width-2)&(y>=2)&(y<cam.height-2)
            ids=np.flatnonzero(valid); ok=ids[mask[y[ids],x[ids]]>127]
            support[lo+ok]+=1
    return np.flatnonzero(support>=min_support).astype(np.int64),support

def select_mask_cone_candidates(xyz,cameras,mask_dir,min_support=3,chunk_size=8192,erosion_pixels=3):
    """GT-free mask-view-cone candidates; this does not claim surface membership."""
    xyz=np.asarray(xyz); support=np.zeros(len(xyz),np.uint16); eroded=np.zeros(len(xyz),np.uint16);mask_dir=Path(mask_dir)
    kernel=np.ones((2*erosion_pixels+1,2*erosion_pixels+1),np.uint8)
    for cam in cameras:
        mask=cv2.imread(str(mask_dir/Path(cam.name).with_suffix('.png')),0); em=cv2.erode((mask>127).astype(np.uint8),kernel)
        for lo in range(0,len(xyz),chunk_size):
            hi=min(lo+chunk_size,len(xyz));pc=world_to_camera(xyz[lo:hi],cam.world_to_camera);uv,z=project(pc,cam.K)
            x=np.rint(uv[:,0]).astype(np.int64);y=np.rint(uv[:,1]).astype(np.int64);v=np.isfinite(uv).all(1)&np.isfinite(z)&(z>0)&(x>=2)&(x<cam.width-2)&(y>=2)&(y<cam.height-2)
            ids=np.flatnonzero(v);support[lo+ids[mask[y[ids],x[ids]]>127]]+=1;eroded[lo+ids[em[y[ids],x[ids]]>0]]+=1
    return np.flatnonzero(support>=min_support).astype(np.int64),support,eroded

def write_xyz_ply(path,xyz):
    a=np.empty(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4')]); a['x'],a['y'],a['z']=xyz.T
    PlyData([PlyElement.describe(a,'vertex')]).write(str(path))
