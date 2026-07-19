import numpy as np
RENDERER_SEMANTICS_NAME='max_weight_threshold_window_depth'
DISPERSION_NAME='threshold_conditioned_depth_dispersion'
TAUS=np.array([.05,.15,.30,.50,.70,.85],np.float32)
STARTS=np.maximum(0,TAUS-.025);ENDS=np.minimum(1,TAUS+.025)
def compute_dispersion(depths):
 d=np.asarray(depths,float);valid=np.isfinite(d)&(d>0);n=valid.sum(0);masked=np.where(valid,d,np.nan);spread=np.nanmax(masked,axis=0)-np.nanmin(masked,axis=0);ref=np.nanmedian(masked,axis=0);pairs=valid[1:]&valid[:-1];j=np.where(pairs,np.abs(d[1:]-d[:-1]),np.nan);jump=np.nanmax(j,axis=0);ok=n>=3
 for a in (spread,ref,jump):a[~np.isfinite(a)]=0
 sr=np.where(ok,spread/(ref+1e-6),0);jr=np.where(ok,jump/(ref+1e-6),0);disp=np.where(ok,.7*sr+.3*jr,0)
 return {'valid':valid.astype(np.uint8),'num_valid_hypotheses':n.astype(np.uint8),'depth_spread_abs':np.where(ok,spread,0).astype(np.float32),'depth_spread_rel':sr.astype(np.float32),'max_adjacent_jump_abs':np.where(ok,jump,0).astype(np.float32),'max_adjacent_jump_rel':jr.astype(np.float32),DISPERSION_NAME:disp.astype(np.float32)}

