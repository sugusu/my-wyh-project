import numpy as np
from trgr.formal_depth_sweep import *
def test_names_thresholds_and_adjacent_validity():
 assert RENDERER_SEMANTICS_NAME=='max_weight_threshold_window_depth';assert DISPERSION_NAME=='threshold_conditioned_depth_dispersion';assert np.allclose(STARTS,[.025,.125,.275,.475,.675,.825]);r=compute_dispersion(np.array([[[1]],[[0]],[[2]],[[3]],[[0]],[[4]]],float));assert r['num_valid_hypotheses'][0,0]==4 and r[DISPERSION_NAME][0,0]>0

