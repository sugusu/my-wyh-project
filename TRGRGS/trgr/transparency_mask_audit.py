from pathlib import Path
import numpy as np,cv2,torch
from PIL import Image

def renderer_transparency_tensor(scene,name,resolution=(800,800)):
    """Exact dataset_readers -> process_image transparency path, before CUDA rasterizer."""
    scene=Path(scene);p=scene/'transparent_masks'/Path(name).with_suffix('.png');raw=np.array(Image.open(p).convert('L'))
    binary=(raw.astype(np.float32)/255.0>.5).astype(np.float32)
    image=Image.open(scene/'images'/name);alpha=np.array(image.getchannel(3),np.float32)/255 if image.mode=='RGBA' else np.ones_like(binary)
    tensor=torch.from_numpy(binary*alpha)[None,None]
    resized=torch.nn.functional.interpolate(tensor,size=(resolution[1],resolution[0]),mode='bilinear')[0,0].numpy()
    return raw,resized,(resized>.5).astype(np.uint8)
def component_stats(mask,normal=None):
    m=mask.astype(np.uint8);total=int(m.sum());n,lab,stats,_=cv2.connectedComponentsWithStats(m,8);areas=stats[1:,cv2.CC_STAT_AREA].astype(int) if n>1 else np.array([],int);minimum=max(64,int(np.ceil(.001*total)));small=areas[areas<minimum];largest=int(areas.max()) if len(areas) else 0;border=int(m[0].sum()+m[-1].sum()+m[:,0].sum()+m[:,-1].sum())
    valid_inside=0
    if normal is not None and total:valid_inside=int(((np.linalg.norm(normal,axis=2)>.1)&(m>0)).sum())
    return {'mask_pixel_count':total,'mask_area_ratio':float(m.mean()),'component_count':len(areas),'largest_component_area':largest,'largest_component_ratio':float(largest/total) if total else 0.,'component_area_list':areas.tolist(),'minimum_small_component_area':minimum,'significant_component_count':int((areas>=minimum).sum()),'small_component_pixel_count':int(small.sum()),'small_component_pixel_fraction':float(small.sum()/total) if total else 0.,'border_touch_pixel_count':border,'border_touch_ratio':float(border/total) if total else 0.,'valid_normal_count_inside_mask':valid_inside,'valid_normal_ratio_inside_mask':float(valid_inside/total) if total else 0.}

