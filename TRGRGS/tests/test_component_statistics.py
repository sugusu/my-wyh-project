import numpy as np
from trgr.transparency_mask_audit import component_stats
def test_two_significant_components():
 m=np.zeros((100,100),np.uint8);m[5:25,5:25]=1;m[60:90,60:90]=1;s=component_stats(m);assert s['component_count']==2 and s['significant_component_count']==2 and s['small_component_pixel_fraction']==0
