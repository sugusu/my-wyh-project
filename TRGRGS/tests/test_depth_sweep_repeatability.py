import numpy as np
def test_repeatability_gate_definition():
 a=np.array([[0.,1.]],np.float32);b=a.copy();assert np.array_equal(a>0,b>0);assert np.max(np.abs(a-b))<=1e-6;assert np.isfinite(a).all()

