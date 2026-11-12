const {test}=require('node:test'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {roundHalfEven,letterboxLayout,tensorFromRGBA}=require('../frontend/model-contract.js');
const reference=require('./fixtures/resize-reference.json');
test('letterbox centers odd padding using Python half-to-even rounding',()=>{
  assert.equal(roundHalfEven(37.5),38);assert.equal(roundHalfEven(36.5),36);
  assert.deepEqual(letterboxLayout(2048,1366,224),{width:224,height:149,left:0,top:38});
  assert.deepEqual(letterboxLayout(1366,2048,224),{width:149,height:224,left:38,top:0});
  assert.throws(()=>letterboxLayout(0,10,224));
});
for(const fixture of reference)test(`matches Pillow bilinear RGB and alpha tensor ${fixture.width}x${fixture.height}->${fixture.size}`,()=>{
  const {width,height,size,transparent}=fixture,rgba=new Uint8Array(width*height*4);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const i=(y*width+x)*4;for(let c=0;c<3;c++)rgba[i+c]=(x*31+y*17+c*67)%256;
    rgba[i+3]=transparent?(x*43+y*19)%256:255;
  }
  const tensor=tensorFromRGBA(rgba,width,height,{size,mean:[0,0,0],std:[1,1,1]});
  assert.equal(crypto.createHash('sha256').update(Buffer.from(tensor.buffer)).digest('hex'),fixture.tensor_sha256);
});
