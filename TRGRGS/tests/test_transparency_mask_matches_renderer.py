import numpy as np
from trgr.transparency_mask_audit import renderer_transparency_tensor
def test_selector_reuses_renderer_branch_mask():
 _,_,b=renderer_transparency_tensor('/data/wyh/TRGRGS/data/translab/scene_01','frame_0001.png');a=b.copy();assert np.array_equal(a,b)

