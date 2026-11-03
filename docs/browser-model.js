// GitHub Pages: 图片只在浏览器中处理，加载公开的 ONNX 推理模型。
let sessionPromise;
const metadataPromise = fetch('./model.json').then(r => {if(!r.ok)throw Error('无法读取模型类别');return r.json();});
function getSession() {
  if (!sessionPromise) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    sessionPromise = ort.InferenceSession.create('./model.onnx', {executionProviders:['wasm']}).catch(error => {sessionPromise=null;throw error;});
  }
  return sessionPromise;
}
window.browserModel = {
  classes: async () => (await metadataPromise).classes,
  predict: async (file, signal) => {
    const [metadata, session] = await Promise.all([metadataPromise, getSession()]);
    if(signal.aborted)throw new DOMException('Aborted','AbortError');
    const bitmap = await createImageBitmap(file, {imageOrientation:'from-image'});
    if(bitmap.width * bitmap.height > 20000000){bitmap.close();throw Error('图片像素总数不能超过两千万');}
    const size=metadata.size, canvas=document.createElement('canvas');canvas.width=canvas.height=size;
    const context=canvas.getContext('2d',{willReadFrequently:true});
    context.fillStyle='#fff';context.fillRect(0,0,size,size);
    const scale=Math.min(size/bitmap.width,size/bitmap.height);
    const width=Math.round(bitmap.width*scale),height=Math.round(bitmap.height*scale);
    context.drawImage(bitmap,Math.floor((size-width)/2),Math.floor((size-height)/2),width,height);bitmap.close();
    const rgba=context.getImageData(0,0,size,size).data, count=size*size, values=new Float32Array(3*count);
    for(let i=0;i<count;i++)for(let c=0;c<3;c++)values[c*count+i]=(rgba[4*i+c]/255-metadata.mean[c])/metadata.std[c];
    if(signal.aborted)throw new DOMException('Aborted','AbortError');
    const output=await session.run({image:new ort.Tensor('float32',values,[1,3,size,size])});
    const logits=Array.from(output.logits.data);
    if(!logits.every(Number.isFinite))throw Error('模型输出异常，请重新选择图片');
    const max=Math.max(...logits), exps=logits.map(v=>Math.exp(v-max)), sum=exps.reduce((a,b)=>a+b,0);
    const candidates=exps.map((v,i)=>({name:metadata.classes[i],confidence:v/sum})).sort((a,b)=>b.confidence-a.confidence).slice(0,3);
    return {candidates,uncertain:candidates[0].confidence<.8,note:'预测置信度不是准确率；模型不能可靠判断未知食材，请人工确认。'};
  }
};
