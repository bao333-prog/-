"""固定新增四类的筛选/来源组/划分；原A的所有旧划分只读复用。"""
from pathlib import Path
from collections import Counter,defaultdict
from urllib.parse import urlparse,unquote
import itertools,json,random,hashlib,re

import numpy as np
from PIL import Image,ImageOps
from oven_food.manifest import ManifestRecord,read_manifest,write_manifest,validate_manifest

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]/"runs/formal_A_20261025_113459"
ROWS=json.loads((OUT/"inventory.json").read_text(encoding="utf-8"))
CLASSES=["芦笋","培根","三文鱼","整鸡"]
EXCLUDED={"芦笋_027":"来源记录为游戏/三维资产讨论，真实照片身份疑点",
          "芦笋_168":"来源说明明确蒸熟后切段，食品状态疑点",
          "培根_057":"来源商品明确pre-cooked，虽描述自相矛盾仍先排除",
          "三文鱼_074":"来源商品明确grilled/cooked-to-order，食品状态疑点",
          "三文鱼_101":"图片带250g文字，避免文字捷径",
          "三文鱼_113":"图片带明显品牌标识",
          "三文鱼_182":"图片带大面积商品文字",
          "三文鱼_194":"图片带品牌标识"}
parent={r["id"]:r["id"] for r in ROWS};edges=[]
def find(key):
    while parent[key]!=key:
        parent[key]=parent[parent[key]];key=parent[key]
    return key
def union(a,b,reason):
    parent[find(b)]=find(a);edges.append({"left":a,"right":b,"reason":reason})
def rgb(path):
    with Image.open(path) as src:
        im=ImageOps.exif_transpose(src).convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA",im.size,"white"),im).convert("RGB")
def canonical(image):
    im=ImageOps.contain(image,(256,256));p=np.asarray(im,dtype=np.int16)
    bg=np.median(np.array([p[0,0],p[-1,0],p[0,-1],p[-1,-1]]),axis=0)
    ys,xs=np.nonzero(np.max(np.abs(p-bg),axis=2)>35)
    if len(xs):im=im.crop((max(0,int(xs.min())-2),max(0,int(ys.min())-2),min(im.width,int(xs.max())+3),min(im.height,int(ys.max())+3)))
    return ImageOps.pad(im,(32,32),color="white")
cosine=np.cos(np.pi/32*(np.arange(32)+.5)[None,:]*np.arange(8)[:,None])
def phash(im):
    low=(cosine@np.asarray(im.convert("L"),dtype=np.float32)@cosine.T).ravel()
    return sum(int(b)<<i for i,b in enumerate(low>np.median(low[1:])))
seen={};features={}
for row in ROWS:
    im=canonical(rgb(row["path"]));features[row["id"]]=(phash(im),phash(ImageOps.mirror(im)),np.asarray(im,dtype=np.float32)/255)
    for name,value in [("sha256",row["sha256"]),("original_sha256",row["original_sha256"])]:
        if value:
            key=(row["class"],name,value)
            if key in seen:union(seen[key],row["id"],name)
            seen[key]=row["id"]
    page=row["page_url"]
    if page.startswith("http"):key=(row["class"],"source_domain",urlparse(page).hostname)
    elif page:key=(row["class"],"legacy_source",page)
    else:continue
    if key in seen:union(seen[key],row["id"],str(key[1:]))
    seen[key]=row["id"]
for label in CLASSES:
    for left,right in itertools.combinations([r for r in ROWS if r["class"]==label],2):
        a,_,p=features[left["id"]];b,bm,q=features[right["id"]]
        distance=min((a^b).bit_count(),(a^bm).bit_count())
        if distance<=7:
            mae=float(min(np.abs(p-q).mean(),np.abs(p-q[:,::-1]).mean()))
            if distance<=4 or mae<.075:union(left["id"],right["id"],f"visual_similarity phash={distance},mae={mae:.4f}")
manual={"芦笋":[[6,54,55,69,112,116]],"培根":[[162,163]],
        "三文鱼":[[62,63,105],[35,83,111],[102,104]],"整鸡":[[34,35,36]]}
for label,groups in manual.items():
    for ids in groups:
        for i in ids[1:]:union(f"{label}_{ids[0]:03d}",f"{label}_{i:03d}","manual_conservative_visual_group")
grouped=defaultdict(list)
for row in ROWS:
    if row["id"] not in EXCLUDED:grouped[find(row["id"])].append(row)
rng=random.Random(42);assignments={}
def choose(pool,target,label):
    possible={0:[]}
    for i,group in enumerate(pool):
        for n,ids in list(possible.items()):possible.setdefault(n+len(group),ids+[i])
    choices=[(n,ids) for n,ids in possible.items() if n>0 and len(ids)<len(pool)]
    if not choices:raise ValueError(label+"同源组不足，停止划分")
    return min(choices,key=lambda item:abs(item[0]-target))[1]
for label in CLASSES:
    groups=[g for g in grouped.values() if g[0]["class"]==label];rng.shuffle(groups)
    total=sum(map(len,groups))
    for split in ["test","val"]:
        ids=choose(groups,round(total*.15),label)
        for i in ids:assignments[find(groups[i][0]["id"])]=split
        groups=[g for i,g in enumerate(groups) if i not in ids]
    for g in groups:assignments[find(g[0]["id"]) ]="train"
records=[];review=[]
for row in ROWS:
    if row["id"] in EXCLUDED:
        review.append({**row,"included":False,"reason":EXCLUDED[row["id"]]});continue
    group="Bnew_"+find(row["id"]);split=assignments[find(row["id"])]
    dest=OUT/"prepared_images"/row["class"]/(row["id"]+".png");dest.parent.mkdir(parents=True,exist_ok=True)
    ImageOps.pad(rgb(row["path"]),(224,224),method=Image.Resampling.BILINEAR,color="white").save(dest)
    records.append(ManifestRecord(str(dest),row["class"],"proxy",group,"unknown_web","unknown_web","none","none",split))
    review.append({**row,"included":True,"split":split,"group":group,"prepared_path":str(dest)})
old=read_manifest(BASE/"manifest.csv")
combined=old+records;validate_manifest(combined,check_files=True);write_manifest(OUT/"manifest.csv",combined)
old_review=json.loads((BASE/"review_and_split.json").read_text(encoding="utf-8"))
hash_split=defaultdict(set);hash_labels=defaultdict(set)
for row in old_review+review:
    if row["included"]:
        hash_split[row["sha256"]].add(row["split"]);hash_labels[row["sha256"]].add(row["class"])
assert all(len(v)==1 for v in hash_split.values()),"发现跨集合字节重复，需重新分组"
assert all(len(v)==1 for v in hash_labels.values()),"发现跨标签重复图，需重新复核"
summary={c:{s:sum(r.food_label==c and r.split==s for r in records) for s in ["train","val","test"]} for c in CLASSES}
for name,value in [("review_and_split.json",review),("grouping_edges.json",edges),("new_split_summary.json",summary)]:
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
config=json.loads((BASE/"train_config.json").read_text(encoding="utf-8"))
config.update(experiment_name="first_increment_B_replay20",manifest=str(OUT/"manifest.csv"),data_root=str(OUT),output_dir=str(OUT/"training"),
              checkpoint=str(BASE/"training/best.pt"),pretrained=False,augmentation="gentle",resize_mode="letterbox",class_balanced_loss=True,
              new_classes=CLASSES,replay_per_class=20,epochs=20,learning_rate=.0001,batch_size=16,num_workers=0,device="cpu")
(OUT/"train_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
(OUT/"baseline.json").write_text(json.dumps({"parent_dir":str(BASE),"parent_sha256":hashlib.sha256((BASE/"training/best.pt").read_bytes()).hexdigest(),
    "old_manifest_sha256":hashlib.sha256((BASE/"manifest.csv").read_bytes()).hexdigest(),"new_classes":CLASSES},ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"input":len(ROWS),"included":len(records),"excluded":len(EXCLUDED),"new_split":summary},ensure_ascii=False,indent=2))
