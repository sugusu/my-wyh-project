import numpy as np
from trgr.threshold_semantics import alpha_compositing_reference
def test_controlled_alpha_crossings_are_monotonic():
 r=alpha_compositing_reference([1,2,3],[.2,.5,.8],[.05,.15,.30,.50,.70,.85])
 assert np.allclose(r['transmittance'],[1,.8,.4]);assert np.allclose(r['weights'],[.2,.4,.32]);assert np.allclose(r['cumulative_opacity'],[.2,.6,.92]);assert np.all(np.diff(r['crossing_depth'])>=0)

