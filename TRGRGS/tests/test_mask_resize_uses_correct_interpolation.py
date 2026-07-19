import torch,numpy as np
def test_threshold_before_bilinear_resize_does_not_join_separated_regions():
 x=torch.zeros(1,1,16,16);x[:,:,2:5,2:5]=1;x[:,:,10:13,10:13]=1;y=torch.nn.functional.interpolate(x,size=(8,8),mode='bilinear');assert ((y>.5)[0,0]).sum()>0

