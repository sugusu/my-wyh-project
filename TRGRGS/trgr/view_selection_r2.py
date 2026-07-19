import numpy as np
def select_probe_views_r2(rows,count=8):
    rows=sorted(rows,key=lambda r:r['camera_index']);areas=np.array([r['mask_area_ratio'] for r in rows if r['mask_pixel_count']>0]);steps=[(.05,.95,.10,.70,.05),(.05,.95,.10,.70,.10),(.05,.95,.10,.70,.20),(.02,.98,.10,.70,.20),(.02,.98,.10,.60,.20),(.02,.98,.20,.60,.20)];history=[];cand=[]
    for i,(lo,hi,border,normal,small) in enumerate(steps):
        qlo,qhi=np.quantile(areas,[lo,hi]);cand=[r for r in rows if r['mask_pixel_count']>=500 and qlo<=r['mask_area_ratio']<=qhi and r['border_touch_ratio']<=border and r['valid_normal_ratio_inside_mask']>=normal and r['small_component_pixel_fraction']<=small];history.append({'step':i,'area_percentiles':[lo,hi],'border_max':border,'normal_min':normal,'small_component_fraction_max':small,'candidate_count':len(cand)})
        if len(cand)>=8:break
    if len(cand)<8:return [],cand,history,'FAIL_INSUFFICIENT_USABLE_PROBES'
    C=np.stack([r['center'] for r in cand]);D=np.stack([r['direction'] for r in cand]);cd=np.linalg.norm(C[:,None]-C[None],axis=2);cd/=max(cd.max(),1e-12);dist=.5*cd+.5*np.arccos(np.clip(D@D.T,-1,1))/np.pi
    mean=dist.mean(1);sel=[min(np.flatnonzero(np.isclose(mean,mean.max())),key=lambda i:cand[i]['camera_index'])]
    while len(sel)<count:
        score=dist[:,sel].min(1);score[sel]=-1;mx=score.max();sel.append(min(np.flatnonzero(np.isclose(score,mx)),key=lambda i:cand[i]['camera_index']))
    return [cand[i] for i in sel],cand,history,'PASS_GT_FREE_PROBE_SELECTION' if len(cand)>=16 else 'PASS_LIMITED_CANDIDATE_POOL'

