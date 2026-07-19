import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
def stats(x):
 x=np.asarray(x,float);return {'count':int(len(x)),'mean':float(np.mean(x)),'median':float(np.median(x)),'p75':float(np.quantile(x,.75)),'p90':float(np.quantile(x,.9)),'p95':float(np.quantile(x,.95)),'within_0.0025':float(np.mean(x<=.0025)),'within_0.005':float(np.mean(x<=.005)),'within_0.010':float(np.mean(x<=.01))}
def reliability(score,error):
 score=np.asarray(score);error=np.asarray(error);label=error>.005;rate=float(label.mean());ap=float(average_precision_score(label,score)) if label.any() else 0.;rho=float(spearmanr(score,error).statistic);order=np.argsort(score)[::-1];bins=[]
 for ids in np.array_split(np.argsort(score),10):bins.append({'count':len(ids),'score_mean':float(score[ids].mean()),'error_mean':float(error[ids].mean()),'error_rate':float(label[ids].mean())})
 def top(q):return float(label[order[:max(1,int(len(order)*q))]].mean())
 return {'positive_rate':rate,'random_auprc':rate,'dispersion_auprc':ap,'auprc_random_ratio':ap/(rate+1e-12),'spearman':rho,'top_5pct_error_rate':top(.05),'top_10pct_error_rate':top(.10),'top_20pct_error_rate':top(.20),'global_error_rate':rate,'calibration_10bin':bins}
