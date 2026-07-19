import numpy as np
from trgr.foreground_coordinate_audit import _numbers
def test_numeric_flatten_finite():assert np.isfinite(list(_numbers({'a':[1.,2.]}))).all()

