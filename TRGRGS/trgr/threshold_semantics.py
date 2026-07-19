import numpy as np

SEMANTICS_NAME='max_weight_threshold_window_depth'
def alpha_compositing_reference(depth,alpha,taus):
    depth=np.asarray(depth,float);alpha=np.asarray(alpha,float);T=np.r_[1.,np.cumprod(1-alpha)[:-1]];w=T*alpha;cum=np.cumsum(w)
    crossing=np.array([depth[np.searchsorted(cum,t)] if t<=cum[-1] else np.nan for t in taus])
    return {'transmittance':T,'weights':w,'cumulative_opacity':cum,'crossing_depth':crossing}
def threshold_window_reference(depth,weights,start,end,window_size):
    """CPU semantic model of forward.cu renderDepthCUDA transparent branch."""
    order=np.argsort(depth);d=np.asarray(depth,float)[order];w=np.asarray(weights,float)[order];c=np.r_[0,np.cumsum(w)]
    starts=np.flatnonzero(c[1:]>start)
    if not len(starts):return 0.
    a=starts[0];ends=np.flatnonzero(c[1:]>end);b=ends[0] if len(ends) else len(d)-1
    best=None
    for i in range(a,b+1):
        j=i
        while j+1<=b and d[j+1]-d[i]<=window_size:j+=1
        score=w[i:j+1].sum()
        if best is None or score>best[0]:best=(score,i,j)
    score,i,j=best
    return float(np.sum(d[i:j+1]*w[i:j+1])/score) if score>0 else 0.

