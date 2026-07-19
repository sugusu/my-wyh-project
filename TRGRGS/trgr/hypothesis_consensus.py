import numpy as np
def consensus_score(valid,bidir,residual,tolerance=.005):
 ratio=np.divide(bidir,valid,out=np.zeros_like(bidir,dtype=np.float32),where=valid>0)
 return ratio*np.exp(-residual/tolerance),ratio
def deterministic_equal(a,b,tol=1e-6):
 return all(np.array_equal(a[k],b[k]) if a[k].dtype.kind in 'biu' else np.max(np.abs(a[k]-b[k]))<=tol for k in a)
