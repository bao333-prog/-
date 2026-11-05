"""导出浏览器模型，并在相同张量上核对 PyTorch/ONNX 输出。"""
import argparse, json, sys, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
import onnxruntime as ort
from predict import load_classifier

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--id',required=True);p.add_argument('--name',required=True);p.add_argument('--version',required=True);a=p.parse_args()
model,names,cp=load_classifier(a.model);torch.set_num_threads(4)
from oven_food.model import checkpoint_resize_mode
if checkpoint_resize_mode(cp) != 'letterbox': raise ValueError('Browser export currently supports letterbox only')
a.output.mkdir(parents=True,exist_ok=True)
size=cp['image_size'];torch.manual_seed(42);sample=torch.randn(1,3,size,size)
torch.onnx.export(model,sample,a.output/'model.onnx',input_names=['image'],output_names=['logits'],opset_version=17,dynamo=False)
session=ort.InferenceSession(str(a.output/'model.onnx'),providers=['CPUExecutionProvider'])
with torch.inference_mode(): expected=model(sample).numpy()
actual=session.run(None,{'image':sample.numpy()})[0]
np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-4)
metadata=cp.get('preprocessing',{})
(a.output/'model.json').write_text(json.dumps({'schema_version':1,'task':'classification','id':a.id,'name':a.name,'version':a.version,'file':'./model.onnx','resize_mode':'letterbox','color':'RGB','layout':'NCHW','input_name':'image','output_name':'logits','output_type':'logits','threshold':0.8,'sha256':hashlib.sha256((a.output/'model.onnx').read_bytes()).hexdigest(),'classes':names,'size':size,'mean':metadata.get('mean',[.485,.456,.406]),'std':metadata.get('std',[.229,.224,.225])},ensure_ascii=False),encoding='utf-8')
print('ONNX check passed; max logit error:',float(abs(actual-expected).max()))
