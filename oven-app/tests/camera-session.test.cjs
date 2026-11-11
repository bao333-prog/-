const {test}=require('node:test');
const assert=require('node:assert/strict');
const {CameraSession,captureFrame}=require('../frontend/camera-session.js');
function fakeStream(){let stopped=0,ended;const track={readyState:'live',stop(){stopped++;},addEventListener(name,fn){if(name==='ended')ended=fn;}};return {getTracks:()=>[track],getVideoTracks:()=>[track],stopped:()=>stopped,end:()=>ended()};}
test('requests video only and selects the exact USB device; stop releases tracks',async()=>{
  const stream=fakeStream();let constraints;
  const session=new CameraSession({getUserMedia:async value=>{constraints=value;return stream;}},()=>{});
  await session.start('oven-usb');assert.equal(constraints.audio,false);assert.deepEqual(constraints.video.deviceId,{exact:'oven-usb'});
  session.stop();assert.equal(stream.stopped(),1);assert.equal(session.stream,null);
});
test('late permission result after closing is released without reconnecting',async()=>{
  let resolve;const stream=fakeStream(),events=[];
  const session=new CameraSession({getUserMedia:()=>new Promise(r=>resolve=r)},state=>events.push(state));
  const pending=session.start();session.stop();resolve(stream);await pending;
  assert.equal(session.stream,null);assert.equal(stream.stopped(),1);assert.ok(!events.includes('connected'));
});
test('a newer connection wins over a stale connection',async()=>{
  const resolves=[],first=fakeStream(),second=fakeStream();
  const session=new CameraSession({getUserMedia:()=>new Promise(r=>resolves.push(r))},()=>{});
  const a=session.start('a'),b=session.start('b');resolves[1](second);await b;resolves[0](first);await a;
  assert.equal(session.stream,second);assert.equal(first.stopped(),1);session.stop();
});
test('permission denied and occupied devices produce actionable errors',async()=>{
  for(const [name,expected] of [['NotAllowedError',/权限被拒绝/],['NotReadableError',/占用/],['NotFoundError',/USB/],['OverconstrainedError',/所选摄像头/]]) {
    let message;const session=new CameraSession({getUserMedia:async()=>{throw Object.assign(Error(),{name});}},(state,text)=>{if(state==='error')message=text;});
    assert.equal(await session.start(),null);assert.match(message,expected);
  }
});
test('connection timeout releases a stream that arrives afterwards',async()=>{
  let resolve,message;const stream=fakeStream();
  const session=new CameraSession({getUserMedia:()=>new Promise(r=>resolve=r)},(state,text)=>{if(state==='error')message=text;},5);
  await session.start();assert.match(message,/超时/);resolve(stream);await new Promise(r=>setImmediate(r));assert.equal(stream.stopped(),1);
});
test('USB ended event closes the stream; old ended events cannot close a new stream',async()=>{
  const first=fakeStream(),second=fakeStream();let stream=first;const events=[];
  const session=new CameraSession({getUserMedia:async()=>stream},state=>events.push(state));
  await session.start();stream=second;await session.start();first.end();assert.equal(session.stream,second);
  second.end();assert.equal(session.stream,null);assert.equal(events.at(-1),'disconnected');
});
test('capture waits for a real frame, limits size and creates a JPEG for existing upload pipeline',async()=>{
  await assert.rejects(captureFrame({readyState:0}),/尚未就绪/);
  const canvas={getContext:()=>({drawImage(){}}),toBlob:callback=>callback(new Blob(['frame'],{type:'image/jpeg'}))};
  const result=await captureFrame({readyState:2,videoWidth:3840,videoHeight:2160},()=>canvas,File);
  assert.equal(canvas.width,1280);assert.equal(canvas.height,720);assert.equal(result.type,'image/jpeg');
  assert.ok(result.name.endsWith('.jpg'));
});
