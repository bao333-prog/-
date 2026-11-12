from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image
repo=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(repo/'oven-app'),str(repo/'oven-app/src')]
from predict import image_tensor
cases=[]
for width,height,size,transparent in [(3,2,8,False),(257,129,17,False),(129,257,17,False),(2048,1366,224,False),(1,1,5,True),(31,23,11,True),(224,224,224,False),(10,6,7,False)]:
    y,x=np.indices((height,width));rgba=np.empty((height,width,4),dtype=np.uint8)
    for c in range(3):rgba[:,:,c]=(x*31+y*17+c*67)%256
    rgba[:,:,3]=(x*43+y*19)%256 if transparent else 255
    cp={'image_size':size,'preprocessing':{'resize_mode':'letterbox','mean':[0,0,0],'std':[1,1,1]}}
    tensor=image_tensor(Image.fromarray(rgba),cp).numpy()
    cases.append({'width':width,'height':height,'size':size,'transparent':transparent,'tensor_sha256':hashlib.sha256(tensor.tobytes()).hexdigest()})
out=repo/'oven-app/tests/fixtures/resize-reference.json';out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(cases,indent=2)+'\n',encoding='utf-8')
print('Generated',len(cases),'Pillow reference cases')
