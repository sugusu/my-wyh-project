import numpy as np
def cluster_pixel(depths,scores,ratios,valid_counts,residuals,merge_tolerance=.0025):
 ids=np.flatnonzero(np.isfinite(depths)&(depths>0)&(valid_counts>=3));clusters=[]
 for i in ids[np.argsort(depths[ids],kind='stable')]:
  if clusters and abs(depths[i]-np.mean([depths[j] for j in clusters[-1]]))<=merge_tolerance:clusters[-1].append(int(i))
  else:clusters.append([int(i)])
 out=[]
 for members in clusters:
  rep=sorted(members,key=lambda i:(-scores[i],i))[0]
  out.append({'members':members,'depth':float(depths[rep]),'representative_tau':rep,'max_score':float(max(scores[members])),'mean_score':float(np.mean(scores[members])),'support_ratio':float(ratios[rep]),'residual':float(residuals[rep])})
 return out
