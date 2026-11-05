// Images stay in this browser. Model and preprocessing are declared by model.json.
let metadataPromise, sessionPromise;
let inferenceQueue = Promise.resolve();
function getMetadata() {
  if (!metadataPromise) metadataPromise = fetch('./model.json', {cache:'no-cache'})
    .then(response => { if (!response.ok) throw Error('无法读取模型配置，请检查网络后重试'); return response.json(); })
    .then(window.modelContract.validate)
    .catch(error => { metadataPromise = null; throw error; });
  return metadataPromise;
}
function getSession(metadata) {
  if (!sessionPromise) sessionPromise = (async () => {
    if (!window.ort) throw Error('推理组件加载失败，请检查网络并刷新页面');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    const session = await ort.InferenceSession.create(metadata.file, {executionProviders:['wasm']});
    if (!session.inputNames.includes(metadata.input_name) || !session.outputNames.includes(metadata.output_name)) {
      await session.release();
      throw Error('模型输入输出名称与配置不匹配');
    }
    return session;
  })().catch(error => { sessionPromise = null; throw error; });
  return sessionPromise;
}
function checkAbort(signal) {
  if (signal.aborted) throw new DOMException('Aborted','AbortError');
}
window.browserModel = {
  info: getMetadata,
  classes: async () => (await getMetadata()).classes,
  predict(file, signal, onStage = () => {}) {
    // WASM runs cannot be interrupted, so serialize them and discard obsolete results.
    const task = inferenceQueue.catch(() => {}).then(async () => {
      checkAbort(signal);
      const started = performance.now();
      onStage('正在加载模型');
      const metadata = await getMetadata();
      const session = await getSession(metadata);
      checkAbort(signal);
      const loaded = performance.now();
      onStage('正在处理图片');
      const bitmap = await createImageBitmap(file, {imageOrientation:'from-image'});
      const size = metadata.size;
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = size;
      const context = canvas.getContext('2d', {willReadFrequently:true});
      try {
        checkAbort(signal);
        if (bitmap.width * bitmap.height > 20000000) throw Error('图片像素总数不能超过两千万');
        if (!context) throw Error('浏览器无法处理图片');
        context.fillStyle = '#fff'; context.fillRect(0,0,size,size);
        const scale = Math.min(size/bitmap.width, size/bitmap.height);
        const width = Math.round(bitmap.width*scale), height = Math.round(bitmap.height*scale);
        context.drawImage(bitmap, Math.floor((size-width)/2), Math.floor((size-height)/2), width, height);
      } finally { bitmap.close(); }
      const rgba = context.getImageData(0,0,size,size).data;
      const count = size*size, values = new Float32Array(3*count);
      for(let i=0;i<count;i++) for(let c=0;c<3;c++) values[c*count+i] = (rgba[4*i+c]/255-metadata.mean[c])/metadata.std[c];
      checkAbort(signal);
      onStage('正在识别食材');
      const beforeInference = performance.now();
      const output = await session.run({[metadata.input_name]:new ort.Tensor('float32', values, [1,3,size,size])});
      checkAbort(signal);
      const finished = performance.now();
      const tensor = output[metadata.output_name];
      if (!tensor || tensor.dims.length !== 2 || tensor.dims[0] !== 1 || tensor.dims[1] !== metadata.classes.length) throw Error('模型输出形状与分类配置不匹配');
      const candidates = window.modelContract.rank(tensor.data, metadata);
      return {candidates, uncertain:candidates[0].confidence < metadata.threshold,
        model:metadata.id, model_version:metadata.version,
        timing:{loading_ms:loaded-started, preprocessing_ms:beforeInference-loaded, inference_ms:finished-beforeInference, total_ms:finished-started},
        note:'预测置信度不是准确率；未知食材可能被误判，请人工确认。'};
    });
    inferenceQueue = task;
    return task;
  }
};
