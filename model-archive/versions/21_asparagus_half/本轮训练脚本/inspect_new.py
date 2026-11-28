"""核对本次补充数据的文件、像素、采集会话；生成逐图人工核验索引。"""
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PREVIOUS = ROOT / 'runs/integrated_food_20261126_115154'
FOLDER = Path('C:/Users/lenovo/Desktop/项目/食物/第二批训练/芦笋半个有锡纸(1)/芦笋半个有锡纸')


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inspect(p):
    with Image.open(p) as im:
        rgb = im.convert('RGB')
        pixel = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        small = np.asarray(rgb.convert('L').resize((160, 90)), dtype=np.float32)
        thumb = np.asarray(rgb.convert('L').resize((9, 8)))
        dhash = (thumb[:, 1:] > thumb[:, :-1]).ravel()
        size = list(rgb.size)
    return {'source': str(p), 'sha256': sha(p), 'full_rgb_sha256': pixel,
            'image_size': size, '_small': small, '_dhash': dhash}


def main():
    files = sorted(p for p in FOLDER.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.png', '.jpeg', '.webp', '.bmp'})
    originals = {s: load(PREVIOUS/f'{s}_base.json') for s in ['training', 'validation', 'test']}
    file_lookup = {r['family_id']: {'split': s, 'source': r['source']} for s, rs in originals.items() for r in rs}
    held = originals['validation'] + originals['test']
    held_sessions = {r['source_session'] for r in held if r['environment']=='target'}
    with ThreadPoolExecutor(max_workers=4) as pool:
        new = list(pool.map(inspect, files))
        eval_rows = list(pool.map(inspect, [Path(r['source']) for r in held]))
    eval_pixels = {r['full_rgb_sha256'] for r in eval_rows}
    notes = []
    for i, r in enumerate(new):
        p = Path(r['source']); meta_path = p.with_suffix('.json'); meta = load(meta_path)
        ctx = meta['camera_context']
        r.update(index=i+1, metadata=str(meta_path), metadata_sha256=sha(meta_path),
                 source_session=ctx['session_id'], physical_sample_id=ctx.get('physical_sample_id'),
                 captured_at=ctx.get('captured_at_host_approx'), label='芦笋', split='train',
                 metadata_image_sha_matches=meta['sha256']==r['sha256'],
                 existing_file_match=file_lookup.get(r['sha256']),
                 held_out_rgb_overlap=r['full_rgb_sha256'] in eval_pixels,
                 held_out_session_overlap=ctx['session_id'] in held_sessions)
        candidates=[]
        for j, e in enumerate(eval_rows):
            # 近重复筛查仅作为人工查看线索，绝不把背景相似的分数当作独立性证明。
            mae=float(np.abs(r['_small']-e['_small']).mean())
            distance=int(np.count_nonzero(r['_dhash'] != e['_dhash']))
            candidates.append({'source': e['source'], 'label': held[j]['label'],
                               'mae': mae, 'dhash_distance': distance})
        r['nearest_eval'] = sorted(candidates, key=lambda x:x['mae'])[:3]
        r['visual_status']='pending'
        assert r['metadata_image_sha_matches'], p
        assert not r['held_out_rgb_overlap'] and not r['held_out_session_overlap'], p
        notes.append({k:v for k,v in r.items() if not k.startswith('_')})
    review=RUN/'review'; review.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for offset in range(0,len(notes),20):
        canvas=Image.new('RGB',(1280,1060),'white'); draw=ImageDraw.Draw(canvas)
        for k,r in enumerate(notes[offset:offset+20]):
            x=(k%4)*320; y=(k//4)*212
            with Image.open(r['source']) as im: canvas.paste(ImageOps.contain(im.convert('RGB'),(320,180)),(x,y))
            draw.text((x+3,y+181), f"{r['index']:03d} {Path(r['source']).stem}", font=font, fill='black')
        canvas.save(review/f'contact_{offset//20+1:02d}.jpg',quality=94)
    # 每张新图的最近评测图，按整体差异排序取前12对，供核验连续帧/重编码副本。
    pairs=sorted(notes,key=lambda r:r['nearest_eval'][0]['mae'])[:12]
    for page in range(2):
        canvas=Image.new('RGB',(960,756),'white'); draw=ImageDraw.Draw(canvas)
        for k,r in enumerate(pairs[page*6:page*6+6]):
            x=(k%2)*480; y=(k//2)*252; e=r['nearest_eval'][0]
            for dx,path in [(0,r['source']),(240,e['source'])]:
                with Image.open(path) as im: canvas.paste(ImageOps.contain(im.convert('RGB'),(240,180)),(x+dx,y))
            draw.text((x,y+182),f"新图{r['index']:03d} / {e['label']} MAE={e['mae']:.1f}",font=font,fill='black')
            draw.text((x,y+211),Path(e['source']).name,font=font,fill='black')
        canvas.save(review/f'nearest_eval_{page+1:02d}.jpg',quality=94)
    dump(RUN/'new_inventory_pending.json',notes)
    report={'images':len(notes),'metadata':len(list(FOLDER.rglob('*.json'))),
            'sessions':dict(Counter(r['source_session'] for r in notes)),
            'prefixes':dict(Counter(Path(r['source']).stem.rsplit('_',1)[0] for r in notes)),
            'existing_file_matches':sum(bool(r['existing_file_match']) for r in notes),
            'cross_eval_file_rgb_session_overlap':0,
            'unique_file_sha256':len({r['sha256'] for r in notes}),
            'unique_full_rgb_sha256':len({r['full_rgb_sha256'] for r in notes}),
            'physical_individual_independence_proven':False}
    dump(RUN/'new_data_initial_audit.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
