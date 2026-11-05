"""从前端源码生成 GitHub Pages 静态资源，模型另用 export_web.py 导出。"""
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1]
destination=root.parent/'docs';destination.mkdir(exist_ok=True)
for name in ['inference.js','browser-model.js','model-contract.js']:
    shutil.copyfile(root/'frontend'/name,destination/name)
html=(root/'frontend/index.html').read_text(encoding='utf-8')
html=html.replace('<script src="/inference.js"></script>', '<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js"></script><script src="./model-contract.js"></script><script src="./browser-model.js"></script><script src="./inference.js"></script>')
html=html.replace('图片发送到本机模型处理，不保存上传图片。','图片仅在当前浏览器内识别，不上传到服务器。').replace('本地图片工作台','浏览器食材识别')
(destination/'index.html').write_text(html,encoding='utf-8')
(destination/'.nojekyll').touch()
