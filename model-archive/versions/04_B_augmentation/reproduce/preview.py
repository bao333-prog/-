"""从训练集每类选一张，展示输入与四次随机增强；只保存预览，不改原图。"""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
from oven_food.augment import ProxyAugmenter, prepare_image
from oven_food.manifest import read_manifest

OUT = Path(__file__).resolve().parent
config = json.loads((OUT/'train_config.json').read_text(encoding='utf-8'))
records = read_manifest(config['manifest'])
names = ['南瓜', '吐司', '牛排', '玉米', '西兰花', '鸡翅', '芦笋', '培根', '三文鱼', '整鸡']
font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)
canvas = Image.new('RGB', (5*224, 10*250+36), '#e8e8e8')
draw = ImageDraw.Draw(canvas)
for col, title in enumerate(['确定性输入', '随机增强1', '随机增强2', '随机增强3', '随机增强4']):
    draw.text((col*224+10, 5), title, fill='black', font=font)
selected = []
for row, name in enumerate(names):
    record = next(r for r in records if r.food_label==name and r.split=='train')
    selected.append({'class': name, 'image_path': record.image_path, 'seeds': [42, 43, 44, 45]})
    with Image.open(record.image_path) as source:
        images = [prepare_image(source, image_size=224, resize_mode='letterbox')]
        images += [ProxyAugmenter('gentle_plus', seed=seed, resize_mode='letterbox')(source) for seed in [42, 43, 44, 45]]
    for col, image in enumerate(images):
        canvas.paste(image, (col*224, row*250+36))
        draw.text((col*224+6, row*250+260), name, fill='black', font=font)
canvas.save(OUT/'增强示例.jpg', quality=92)
(OUT/'preview_sources.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
print(OUT/'增强示例.jpg')
