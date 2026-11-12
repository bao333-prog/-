/* Classifier contract shared by the browser and dependency-free Node tests. */
(function(root) {
  'use strict';
  function validate(raw) {
    const fail = message => { throw new Error('模型配置错误：' + message); };
    if (!raw || raw.schema_version !== 1 || raw.task !== 'classification') fail('仅支持第 1 版单图分类配置');
    if (!raw.id || !raw.version || !raw.name) fail('缺少模型标识、名称或版本');
    if (typeof raw.file !== 'string' || !/^(?!\/)(?!.*(?:\.\.|:|\\))[\w./-]+\.onnx$/.test(raw.file)) fail('模型文件必须为相对 ONNX 路径');
    if (!Array.isArray(raw.classes) || !raw.classes.length || raw.classes.some(x => typeof x !== 'string' || !x.trim()) || new Set(raw.classes).size !== raw.classes.length) fail('类别不能为空或重复');
    if (!Number.isInteger(raw.size) || raw.size < 1 || raw.size > 2048) fail('图片尺寸无效');
    for (const key of ['mean','std']) if (!Array.isArray(raw[key]) || raw[key].length !== 3 || !raw[key].every(x => Number.isFinite(x) && (key !== 'std' || x > 0))) fail('归一化参数无效');
    if (raw.resize_mode !== 'letterbox' || raw.color !== 'RGB' || raw.layout !== 'NCHW') fail('当前仅实现 RGB / NCHW / letterbox 预处理');
    if (!raw.input_name || !raw.output_name || !['logits','probabilities'].includes(raw.output_type)) fail('输入输出约定无效');
    if (!Number.isFinite(raw.threshold) || raw.threshold < 0 || raw.threshold > 1) fail('阈值必须在 0 到 1 之间');
    return raw;
  }
  function rank(values, config) {
    if (values.length !== config.classes.length || !Array.from(values).every(Number.isFinite)) throw new Error('模型输出与类别配置不匹配，或包含异常数值');
    let probabilities;
    if (config.output_type === 'logits') {
      const max = Math.max(...values);
      const exps = Array.from(values, v => Math.exp(v - max));
      const sum = exps.reduce((a,b) => a+b, 0);
      probabilities = exps.map(v => v/sum);
    } else {
      const sum = Array.from(values).reduce((a,b) => a+b, 0);
      if (Array.from(values).some(v => v < 0 || v > 1) || Math.abs(sum-1) > .001) throw new Error('模型概率输出无效');
      probabilities = Array.from(values);
    }
    return probabilities.map((confidence,i) => ({name:config.classes[i], confidence})).sort((a,b) => b.confidence-a.confidence).slice(0,3);
  }

  // Match Python round() for the half-pixel positions used by Pillow ImageOps.pad.
  function roundHalfEven(value) {
    const lower=Math.floor(value), fraction=value-lower;
    return fraction===.5 ? lower+(lower%2) : Math.round(value);
  }
  function letterboxLayout(width,height,size) {
    if(![width,height,size].every(v=>Number.isInteger(v) && v>0)) throw Error('图片尺寸无效');
    const w=width>=height?size:roundHalfEven(width/height*size);
    const h=height>=width?size:roundHalfEven(height/width*size);
    if(!w || !h) throw Error('图片长宽比过大，请裁剪后重试');
    return {width:w,height:h,left:roundHalfEven((size-w)/2),top:roundHalfEven((size-h)/2)};
  }
  // Widen the triangular filter when downsampling, rather than sampling only
  // the nearest source pixels. Fixed-point weights and per-pass rounding follow
  // Pillow's 8-bit bilinear resampling convention.
  function coefficients(source,target) {
    const scale=source/target, filterScale=Math.max(1,scale), result=[];
    for(let out=0;out<target;out++) {
      const center=(out+.5)*scale;
      const start=Math.max(0,Math.floor(center-filterScale+.5));
      const end=Math.min(source,Math.floor(center+filterScale+.5));
      const weights=[];let sum=0;
      for(let i=start;i<end;i++){const weight=Math.max(0,1-Math.abs((i-center+.5)/filterScale));weights.push(weight);sum+=weight;}
      result.push({start,weights:weights.map(weight=>Math.floor(weight/sum*4194304+.5))});
    }
    return result;
  }
  function resizeRGB(rgb,width,height,targetWidth,targetHeight) {
    let horizontal=rgb;
    if(width!==targetWidth) {
      horizontal=new Uint8Array(targetWidth*height*3);const kernel=coefficients(width,targetWidth);
      for(let y=0;y<height;y++)for(let x=0;x<targetWidth;x++)for(let c=0;c<3;c++) {
        const {start,weights}=kernel[x];let sum=2097152;
        for(let i=0;i<weights.length;i++)sum+=rgb[(y*width+start+i)*3+c]*weights[i];
        horizontal[(y*targetWidth+x)*3+c]=Math.max(0,Math.min(255,Math.floor(sum/4194304)));
      }
    }
    if(height===targetHeight)return horizontal;
    const resized=new Uint8Array(targetWidth*targetHeight*3),kernel=coefficients(height,targetHeight);
    for(let y=0;y<targetHeight;y++)for(let x=0;x<targetWidth;x++)for(let c=0;c<3;c++) {
      const {start,weights}=kernel[y];let sum=2097152;
      for(let i=0;i<weights.length;i++)sum+=horizontal[((start+i)*targetWidth+x)*3+c]*weights[i];
      resized[(y*targetWidth+x)*3+c]=Math.max(0,Math.min(255,Math.floor(sum/4194304)));
    }
    return resized;
  }
  function tensorFromRGBA(rgba,width,height,metadata) {
    const layout=letterboxLayout(width,height,metadata.size);
    if(rgba.length!==width*height*4)throw Error('图片像素数据不完整');
    const rgb=new Uint8Array(width*height*3);
    for(let i=0;i<width*height;i++)for(let c=0;c<3;c++)rgb[i*3+c]=Math.floor((rgba[i*4+c]*rgba[i*4+3]+255*(255-rgba[i*4+3])+127)/255);
    const resized=resizeRGB(rgb,width,height,layout.width,layout.height);
    const count=metadata.size*metadata.size,values=new Float32Array(3*count);
    for(let c=0;c<3;c++)values.fill(Math.fround(Math.fround(1-metadata.mean[c])/metadata.std[c]),c*count,(c+1)*count);
    for(let y=0;y<layout.height;y++)for(let x=0;x<layout.width;x++)for(let c=0;c<3;c++) {
      const pixel=resized[(y*layout.width+x)*3+c];
      values[c*count+(y+layout.top)*metadata.size+x+layout.left]=Math.fround(Math.fround(Math.fround(pixel/255)-metadata.mean[c])/metadata.std[c]);
    }
    return values;
  }

  const api = {validate, rank, roundHalfEven, letterboxLayout, tensorFromRGBA};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.modelContract = api;
})(typeof window === 'undefined' ? this : window);
