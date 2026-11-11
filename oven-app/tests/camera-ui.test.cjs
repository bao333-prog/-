const {test}=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
const {CameraSession}=require('../frontend/camera-session.js');
function setup(){
  const elements=new Map(),timers=new Map(),loads=[];let timerId=0,stops=0,ended,frameResolve;
  function element(id){if(!elements.has(id))elements.set(id,{id,hidden:false,disabled:false,checked:true,value:'',textContent:'',readyState:2,videoWidth:1280,videoHeight:720,events:{},addEventListener(name,fn){(this.events[name]||=[]).push(fn);},setAttribute(){},replaceChildren(){},add(){},play:async()=>{}});return elements.get(id);}
  const track={readyState:'live',stop(){stops++;},getSettings:()=>({deviceId:'usb'}),addEventListener(name,fn){if(name==='ended')ended=fn;}};
  const stream={getTracks:()=>[track],getVideoTracks:()=>[track]};
  const document={hidden:false,events:{},addEventListener(name,fn){this.events[name]=fn;}};
  const media={enumerateDevices:async()=>[{kind:'videoinput',deviceId:'usb',label:'USB camera'}],getUserMedia:async()=>stream,events:{},addEventListener(name,fn){this.events[name]=fn;}};
  const window={isSecureContext:true,ovenCamera:{CameraSession,captureFrame:async()=>({type:'image/jpeg'})},addEventListener(){}};
  const context=vm.createContext({window,document,navigator:{mediaDevices:media},localStorage:{getItem:()=>'',setItem(){}},Option:class{},Date,$:element,busy:false,classesReady:true,loadVersion:0,
    retryButton:{hidden:true},restartButton:element('restart'),enable(){},clearDraft(){},clearPrediction(){context.busy=false;},reset(){context.loadVersion++;context.busy=false;},
    load(file){loads.push(file);context.busy=true;context.enable();return new Promise(resolve=>{loads.at(-1).finish=()=>{context.busy=false;context.enable();resolve();};});},
    setTimeout(fn){const id=++timerId;timers.set(id,fn);return id;},clearTimeout(id){timers.delete(id);}});
  vm.runInContext(fs.readFileSync(require.resolve('../frontend/camera.js'),'utf8'),context);
  const settle=async()=>{for(let i=0;i<6;i++)await new Promise(r=>setImmediate(r));};
  return {context,element,document,loads,timers,settle,stops:()=>stops,ended:()=>ended(),delayFrame(){window.ovenCamera.captureFrame=()=>new Promise(r=>frameResolve=r);return ()=>frameResolve({type:'image/jpeg'});}};
}
test('continuous mode waits for recognition before scheduling next frame and stops on manual selection',async()=>{
  const s=setup();s.element('input-mode-camera').onclick();await s.element('camera-connect').onclick();await s.settle();
  s.element('camera-continuous').onclick();await s.settle();assert.equal(s.loads.length,1);assert.equal(s.timers.size,0);
  s.loads[0].finish();await s.settle();assert.equal(s.timers.size,1);
  s.element('ingredient').events.change[0]();assert.equal(s.timers.size,0);assert.equal(s.element('camera-continuous').textContent,'开始连续识别');
});
test('switching to upload releases the camera and discards a pending screenshot',async()=>{
  const s=setup();s.element('input-mode-camera').onclick();await s.element('camera-connect').onclick();await s.settle();
  const release=s.delayFrame();const pending=s.element('camera-snapshot').onclick();await s.settle();
  s.element('input-mode-file').onclick();release();await pending;
  assert.equal(s.loads.length,0);assert.ok(s.stops()>0);assert.equal(s.element('camera-pane').hidden,true);
});
test('USB loss cancels continuous scheduling and clears the attached stream',async()=>{
  const s=setup();s.element('input-mode-camera').onclick();await s.element('camera-connect').onclick();await s.settle();
  s.element('camera-continuous').onclick();await s.settle();s.loads[0].finish();await s.settle();
  s.ended();assert.equal(s.timers.size,0);assert.equal(s.element('camera-video').srcObject,null);assert.equal(s.element('camera-snapshot').disabled,true);
});
