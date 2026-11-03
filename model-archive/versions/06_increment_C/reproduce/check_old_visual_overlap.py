"""以保留的旧输入副本作跨类别相似性筛查；候选须人工复核，不能只凭文件SHA。"""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageOps
from oven_food.manifest import read_manifest
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
rows=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
old=read_manifest(ROOT/'runs/B_wings_supplement_20261029_174246/manifest.csv')
old+=read_manifest(ROOT/'runs/B_wings_supplement_20261029_174246/frozen_new_chicken_test.csv')
old += [r for r in read_manifest(ROOT/'runs/increment_B_20261025_165256/manifest.csv') if r.split=='test']
cos=np.cos(np.pi/32*(np.arange(32)+.5)[None,:]*np.arange(8)[:,None])
def feat(path):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGBA');im=Image.alpha_composite(Image.new('RGBA',im.size,'white'),im).convert('RGB')
        im=ImageOps.contain(im,(256,256));p=np.asarray(im,dtype=np.int16)
        bg=np.median(np.array([p[0,0],p[-1,0],p[0,-1],p[-1,-1]]),axis=0)
        ys,xs=np.nonzero(np.max(np.abs(p-bg),axis=2)>35)
        if len(xs):im=im.crop((max(0,int(xs.min())-2),max(0,int(ys.min())-2),min(im.width,int(xs.max())+3),min(im.height,int(ys.max())+3)))
        im=ImageOps.pad(im,(32,32),color='white')
        def phash(x):
            low=(cos@np.asarray(x.convert('L'),dtype=np.float32)@cos.T).ravel()
            return sum(int(b)<<i for i,b in enumerate(low>np.median(low[1:])))
        return phash(im),phash(ImageOps.mirror(im)),np.asarray(im,dtype=np.float32)/255
old_features=[(r,feat(r.image_path)) for r in old]
pairs=[]
for row in rows:
    if not row['decoded']:continue
    a,_,p=feat(row['path'])
    for previous,(b,bm,q) in old_features:
        distance=min((a^b).bit_count(),(a^bm).bit_count())
        if distance<=3:
            mae=float(min(np.abs(p-q).mean(),np.abs(p-q[:,::-1]).mean()))
            if mae<.04:pairs.append(dict(new_id=row['id'],old_path=previous.image_path,old_label=previous.food_label,old_split=previous.split,distance=distance,pixel_mae=mae))
(OUT/'old_visual_overlap_candidates.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(previous_inputs=len(old),cross_old_candidates=pairs),ensure_ascii=False,indent=2))
