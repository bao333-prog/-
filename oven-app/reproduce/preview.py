"""训练图增强预览：确定性输入、旧增强与新温和增强，不修改原图。"""
from pathlib import Path
import os
import json
from PIL import Image,ImageDraw,ImageFont
from oven_food.augment import ProxyAugmenter,prepare_image
from oven_food.manifest import read_manifest
OUT=Path(os.environ.get("OVEN_RUN_DIR", Path(__file__).resolve().parents[1]/"records")).resolve()
records=read_manifest(OUT/'expected_training_manifest.csv')
font=ImageFont.truetype(os.environ.get('OVEN_FONT', 'C:/Windows/Fonts/msyh.ttc'),17)
labels=['鸡翅','玉米','培根','香肠','薯条','可颂']
canvas=Image.new('RGB',(5*224,36+6*248),'#eeeeee');draw=ImageDraw.Draw(canvas)
for i,title in enumerate(['确定性输入','原gentle_plus','新增强42','新增强43','新增强44']):draw.text((i*224+5,4),title,font=font,fill='black')
sources=[]
for row,label in enumerate(labels):
    record=next(r for r in records if r.split=='train' and r.food_label==label)
    with Image.open(record.image_path) as image:
        images=[prepare_image(image,image_size=224,resize_mode='letterbox'),ProxyAugmenter('gentle_plus',resize_mode='letterbox',seed=42)(image)]
        images += [ProxyAugmenter('gentle_retention',resize_mode='letterbox',seed=s)(image) for s in [42,43,44]]
    for col,image in enumerate(images):
        canvas.paste(image,(col*224,36+row*248));draw.text((col*224+5,36+row*248+225),label,fill='black',font=font)
    sources.append(dict(food_class=label,image_path=record.image_path))
canvas.save(OUT/'增强方案对照.jpg',quality=94)
(OUT/'preview_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT/'增强方案对照.jpg')
