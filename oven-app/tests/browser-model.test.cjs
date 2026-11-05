const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const contract = require('../frontend/model-contract.js');
const good = {schema_version:1,task:'classification',id:'test',name:'test',version:'1',file:'./model.onnx',classes:['a','b'],size:2,mean:[0,0,0],std:[1,1,1],resize_mode:'letterbox',color:'RGB',layout:'NCHW',input_name:'pixels',output_name:'scores',output_type:'logits',threshold:.8,sha256:'1'.repeat(64)};
function setup({fetchFail=false,sessionFail=false,run}={}) {
  let fetches=0,loads=0,closed=0;
  const window={modelContract:contract,location:{href:'http://localhost:4173/'}};
  const ort={env:{wasm:{}},Tensor:class {constructor(type,data,dims){Object.assign(this,{type,data,dims});}},InferenceSession:{create:async(url)=>{assert.equal(new URL(url).searchParams.get('sha256'),good.sha256);
    if(sessionFail && ++loads===1) throw Error('load failed');
    return {inputNames:['pixels'],outputNames:['scores'],run:run || (async feeds=>{assert.ok(feeds.pixels);return {scores:{dims:[1,2],data:[1,2]}}})};
  }}};
  window.ort=ort;
  const context=vm.createContext({window,ort,URL,DOMException,Float32Array,performance,
    fetch:async()=>{if(fetchFail && ++fetches===1) throw Error('offline');return {ok:true,json:async()=>good};},
    createImageBitmap:async()=>({width:2,height:2,close(){closed++;}}),
    document:{createElement:()=>({getContext:()=>({fillRect(){},drawImage(){},getImageData:()=>({data:new Uint8Array(16).fill(255)})})})}});
  vm.runInContext(fs.readFileSync(require.resolve('../frontend/browser-model.js'),'utf8'),context);
  return {api:window.browserModel,closed:()=>closed};
}
test('failed metadata and session loads can be retried',async()=>{
  const {api}=setup({fetchFail:true,sessionFail:true});
  await assert.rejects(api.info(),/offline/); assert.equal((await api.info()).id,'test');
  await assert.rejects(api.predict({},new AbortController().signal),/load failed/);
  const result=await api.predict({},new AbortController().signal);
  assert.equal(result.candidates[0].name,'b'); assert.ok(result.timing.inference_ms>=0);
});
test('cancellation discards in-flight output and serializes subsequent runs',async()=>{
  let release,first=true,concurrent=0,maxConcurrent=0;
  const {api}=setup({run:async()=>{concurrent++;maxConcurrent=Math.max(maxConcurrent,concurrent);if(first){first=false;await new Promise(r=>release=r);}concurrent--;return {scores:{dims:[1,2],data:[1,2]}};}});
  const controller=new AbortController();
  const old=api.predict({},controller.signal);
  const discarded=assert.rejects(old,{name:'AbortError'});
  while(!release) await new Promise(r=>setImmediate(r));
  controller.abort();const next=api.predict({},new AbortController().signal);release();
  await discarded;assert.equal((await next).model,'test');assert.equal(maxConcurrent,1);
});
test('incompatible output shapes are rejected and image bitmaps are closed',async()=>{
  const {api,closed}=setup({run:async()=>({scores:{dims:[1,3],data:[1,2,3]}})});
  await assert.rejects(api.predict({},new AbortController().signal),/形状/);assert.equal(closed(),1);
});
