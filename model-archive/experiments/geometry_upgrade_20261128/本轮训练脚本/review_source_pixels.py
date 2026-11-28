"""Spot-check preprocessing independently against full sources, after cache byte audit."""
from collections import defaultdict
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import hashlib, random, sys
from support import RUN, load, dump, sha
sys.path.insert(0,str(RUN/'source_snapshot'))
from oven_food.augment import prepare_image

def verify(row,size):
    p=row['cache_source']
    assert sha(p)==row['cache_source_sha256']
    with Image.open(p) as image:
        assert list(image.size)==row['cache_source_size']
        if row['kind']=='annotation_roi':image=image.crop(row['crop_xyxy'])
        out=prepare_image(image,image_size=size,resize_mode='letterbox')
        assert hashlib.sha256(out.tobytes()).hexdigest()==row['prepared_pixel_sha256']
    return row['kind'],row['environment'],row['label']

for name,size in [('A224',224),('B320',320)]:
    groups=defaultdict(list)
    for r in load(RUN/name/'training.json'):
        groups[(r['kind'],r['label'],r['environment'])].append(r)
    rng=random.Random(11028); selected=[r for g in groups.values() for r in rng.sample(g,min(2,len(g)))]
    with ThreadPoolExecutor(4) as pool:out=list(pool.map(partial(verify,size=size),selected))
    dump(RUN/(name+'_source_review.json'),dict(sampled_train_views=len(out),groups=len(groups),scope='Up to 2 per kind, label, domain; full source hash and size, ROI crop, resize, cached RGB identity',sample_inputs=[r['actual_input'] for r in selected]))
    print(name,len(out),len(groups))
