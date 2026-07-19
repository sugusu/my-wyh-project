import numpy as np

def pixels_depth_to_world(u,v,depth,K,w2c):
    cam=np.stack([(u-K[0,2])/K[0,0]*depth,(v-K[1,2])/K[1,1]*depth,depth],-1)
    return (cam-w2c[:3,3])@w2c[:3,:3]

def world_to_pixels(points,K,w2c):
    cam=points@w2c[:3,:3].T+w2c[:3,3]
    uv=np.stack([K[0,0]*cam[:,0]/cam[:,2]+K[0,2],K[1,1]*cam[:,1]/cam[:,2]+K[1,2]],-1)
    return uv,cam[:,2]

def local_set_residual(uv,z,target_depths,target_mask):
    """Nearest legal depth in a rounded-pixel 3x3 set; no interpolation."""
    n=len(z);best=np.full(n,np.inf,np.float32);bp=np.full((n,2),-1,np.int16);bt=np.full(n,-1,np.int8);bd=np.zeros(n,np.float32)
    h,w=target_mask.shape;cx=np.rint(uv[:,0]).astype(int);cy=np.rint(uv[:,1]).astype(int)
    for dy in (-1,0,1):
      for dx in (-1,0,1):
        x=cx+dx;y=cy+dy;inside=(x>=0)&(x<w)&(y>=0)&(y<h)
        ii=np.flatnonzero(inside)
        if not len(ii):continue
        legal=target_mask[y[ii],x[ii]]
        ii=ii[legal]
        if not len(ii):continue
        vals=target_depths[:,y[ii],x[ii]].T;res=np.abs(vals-z[ii,None]);res[~np.isfinite(vals)|(vals<=0)]=np.inf
        tau=np.argmin(res,1);rr=res[np.arange(len(ii)),tau];take=rr<best[ii];jj=ii[take];tt=tau[take]
        best[jj]=rr[take];bp[jj]=np.stack([x[jj],y[jj]],1);bt[jj]=tt;bd[jj]=vals[take,tt]
    return best,bp,bt,bd
