"""从可追溯源图生成指定分辨率；所有坐标都属于完整原图。"""
import sys
import hashlib
from pathlib import Path
from PIL import Image
from support import RUN, sha

sys.path.insert(0,str(RUN/'source_snapshot'))
from oven_food.augment import prepare_image


def render_row(row,size):
    kind=row['kind'];crop=None
    if kind=='reviewed_food_geometry':
        source=row['synthetic_source_path'];expected=row['synthetic_source_sha256']
    elif kind=='foreground_rotation':
        source=row['foreground_source_path'];expected=row['foreground_source_sha256']
    else:
        source=row['source'];expected=row['family_id']
        if kind=='annotation_roi':crop=row['crop_xyxy']
    if sha(source)!=expected:raise ValueError('缓存源文件指纹变化：'+str(source))
    with Image.open(source) as image:
        original_size=list(image.size)
        if crop is not None:
            l,t,r,b=crop
            if not (0<=l<r<=image.width and 0<=t<b<=image.height):raise ValueError('ROI超出完整源图')
            image=image.crop(crop)
        prepared=prepare_image(image,image_size=size,resize_mode='letterbox')
    return prepared,dict(cache_source=str(source),cache_source_sha256=expected,
          cache_source_size=original_size,cache_crop_xyxy=crop,
          source_already_low_resolution=max(original_size)<=224,
          prepared_pixel_sha256=hashlib.sha256(prepared.tobytes()).hexdigest())


def cache_row(row,size):
    prepared,meta=render_row(row,size)
    view_id=hashlib.sha256(row['actual_input'].encode('utf-8')).hexdigest()[:16]
    target=RUN/'data'/str(size)/'inputs'/row['split']/row['label']/(row['family_id']+'_'+view_id+'.png')
    target.parent.mkdir(parents=True,exist_ok=True);prepared.save(target)
    if size==224 and row['kind']!='foreground_rotation':
        assert meta['prepared_pixel_sha256']==row['prepared_pixel_sha256'],'重建224像素必须与冻结旧版完全一致'
    return {**row,**meta,'baseline_input':row['actual_input'],'actual_input':str(target),
            'prepared_sha256':sha(target),'prepared_resolution':size}
