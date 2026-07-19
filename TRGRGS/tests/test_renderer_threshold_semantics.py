from trgr.threshold_semantics import threshold_window_reference,SEMANTICS_NAME
def test_max_weight_window_not_quantile_crossing():
 d=threshold_window_reference([1.,1.01,2.],[.1,.4,.3],0.,.8,.03)
 assert 1.<d<1.01 and SEMANTICS_NAME=='max_weight_threshold_window_depth'

