"""只读取四类原图，核对来源记录，生成逐张复核和同源候选预览。"""
from pathlib import Path
from collections import Counter,defaultdict
from urllib.parse import urlparse
import csv,hashlib,itertools,json
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
SOURCE=Path('C:/Users/lenovo/Desktop/项目/食物')
FOLDERS={'土豆':'土豆_无背景_200张','香肠':'香肠_没背景_200张','薯条':'薯条_没背景_200张','可颂':'PNG'}
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
cosine=np.cos(np.pi/32*(np.arange(32)+.5)[None,:]*np.arange(8)[:,None])


def rgb(path):
    with Image.open(path) as src:
        im=ImageOps.exif_transpose(src).convert('RGBA')
        return Image.alpha_composite(Image.new('RGBA',im.size,'white'),im).convert('RGB')


def canonical(image):
    im=ImageOps.contain(image,(256,256));p=np.asarray(im,dtype=np.int16)
    bg=np.median(np.array([p[0,0],p[-1,0],p[0,-1],p[-1,-1]]),axis=0)
    ys,xs=np.nonzero(np.max(np.abs(p-bg),axis=2)>35)
    if len(xs):im=im.crop((max(0,int(xs.min())-2),max(0,int(ys.min())-2),min(im.width,int(xs.max())+3),min(im.height,int(ys.max())+3)))
    return ImageOps.pad(im,(32,32),color='white')


def features(image):
    small=canonical(image)
    def phash(im):
        low=(cosine@np.asarray(im.convert('L'),dtype=np.float32)@cosine.T).ravel()
        return sum(int(bit)<<i for i,bit in enumerate(low>np.median(low[1:])))
    return phash(small),phash(ImageOps.mirror(small)),np.asarray(small,dtype=np.float32)/255


rows=[];feat={};seen=defaultdict(list)
for label,folder in FOLDERS.items():
    source=SOURCE/folder;metadata={}
    if (source/'来源记录.json').is_file():
        metadata={r['output_filename']:r for r in json.loads((source/'来源记录.json').read_text(encoding='utf-8-sig'))}
    elif label=='土豆':
        with (source/'索引与来源/图片来源.csv').open(encoding='utf-8-sig',newline='') as f:
            metadata={r['文件名']:r for r in csv.DictReader(f)}
    files=sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'})
    for index,path in enumerate(files,1):
        m=metadata.get(path.name,{})
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        page=m.get('page_url',m.get('来源页面',''));url=m.get('image_url',m.get('原图链接',''))
        row=dict(id=f'{label}_{index:03d}',index=index,food_class=label,path=str(path),filename=path.name,
                 sha256=digest,original_sha256=m.get('sha256',''),page_url=page,image_url=url,
                 description=m.get('description',''),metadata_present=bool(m),
                 metadata_sha256_matches=digest==m.get('output_sha256') if m.get('output_sha256') else None,
                 source_domain=(urlparse(page).hostname or '').lower().removeprefix('www.'),
                 processing=m.get('processing',''),source_dimensions=[m.get('width',m.get('宽')),m.get('height',m.get('高'))])
        try:
            with Image.open(path) as image:image.load();row.update(size=list(image.size),mode=image.mode,decoded=True)
            image=rgb(path);feat[row['id']]=features(image)
            row['content_sha256']=hashlib.sha256(ImageOps.pad(image,(224,224),method=Image.Resampling.BILINEAR,color='white').tobytes()).hexdigest()
            thumbnail=ImageOps.pad(image,(224,174),color='white')
            (OUT/'thumbnails').mkdir(exist_ok=True);thumbnail.save(OUT/'thumbnails'/(row['id']+'.jpg'),quality=93)
        except Exception as exc:row.update(decoded=False,error=repr(exc))
        rows.append(row)
    chosen=[r for r in rows if r['food_class']==label]
    for start in range(0,len(chosen),50):
        page=Image.new('RGB',(5*224,10*204),'#eeeeee');draw=ImageDraw.Draw(page)
        for i,row in enumerate(chosen[start:start+50]):
            x,y=(i%5)*224,(i//5)*204
            if row['decoded']:
                with Image.open(OUT/'thumbnails'/(row['id']+'.jpg')) as image:page.paste(image,(x,y))
            draw.text((x+3,y+178),row['id']+' '+row['filename'][:13],font=font,fill='black')
        page.save(OUT/f'contact_{label}_{start//50+1:02d}.jpg',quality=94)
internal=[]
for a,b in itertools.combinations(rows,2):
    if a['id'] not in feat or b['id'] not in feat:continue
    x,_,p=feat[a['id']];y,ym,q=feat[b['id']]
    distance=min((x^y).bit_count(),(x^ym).bit_count())
    if distance<=7:
        mae=float(min(np.abs(p-q).mean(),np.abs(p-q[:,::-1]).mean()))
        exact=a['sha256']==b['sha256'] or a.get('content_sha256')==b.get('content_sha256')
        if exact or (a['food_class']==b['food_class'] and (distance<=4 or mae<.075)) or (distance<=3 and mae<.04):
            internal.append(dict(left=a['id'],right=b['id'],distance=distance,pixel_mae=mae,exact=exact,cross_label=a['food_class']!=b['food_class']))
# 对照已有全部类别及本次补强数据的文件/原始SHA；只读历史记录，不读取缺失旧原图。
old_reviews=[]
for folder in ['formal_A_20261025_113459','increment_B_20261025_165256']:
    old_reviews+=json.loads((ROOT/'runs'/folder/'review_and_split.json').read_text(encoding='utf-8'))
old_reviews+=json.loads((ROOT/'runs/chicken_wings_review_20261029_171649/reviewed_inventory.json').read_text(encoding='utf-8'))
old_hashes=defaultdict(list)
for r in old_reviews:
    for digest in {r.get('sha256'),r.get('original_sha256')}-{None,''}:old_hashes[digest].append(r['path'])
for row in rows:
    row['old_hash_matches']=sorted({p for digest in {row['sha256'],row['original_sha256']}-{None,''} for p in old_hashes.get(digest,[])})
for filename,value in [('inventory.json',rows),('internal_similarity_candidates.json',internal)]:
    (OUT/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
summary=dict(counts=dict(Counter(r['food_class'] for r in rows)),decoded=sum(r['decoded'] for r in rows),
             metadata_present=dict(Counter(r['food_class'] for r in rows if r['metadata_present'])),
             metadata_hash_mismatches=[r['id'] for r in rows if r['metadata_sha256_matches'] is False],
             old_hash_matches=[r['id'] for r in rows if r['old_hash_matches']],similarity_pairs=len(internal),
             cross_label_pairs=[p for p in internal if p['cross_label']],
             source_domains={c:dict(Counter(r['source_domain'] or 'UNKNOWN' for r in rows if r['food_class']==c)) for c in FOLDERS})
(OUT/'inventory_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('Titles for manual checks:')
for r in rows:
    if r['description']:print(r['id']+' | '+r['description'].splitlines()[0])
