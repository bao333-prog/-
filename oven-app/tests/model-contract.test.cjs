const {test} = require('node:test');
const assert = require('node:assert/strict');
const contract = require('../frontend/model-contract.js');
const good = {schema_version:1,task:'classification',id:'demo',name:'Demo',version:'v1',file:'./model.onnx',classes:['a','b'],size:224,mean:[0,0,0],std:[1,1,1],resize_mode:'letterbox',color:'RGB',layout:'NCHW',input_name:'image',output_name:'logits',output_type:'logits',threshold:.8};
test('rejects detection models, unsupported preprocessing and ambiguous labels', () => {
  for (const patch of [{task:'detection'},{resize_mode:'center_crop'},{classes:['a','a']},{std:[1,0,1]},{file:'https://example.com/model.onnx'},{input_name:''},{threshold:2}]) assert.throws(() => contract.validate({...good,...patch}));
  assert.equal(contract.validate(good), good);
});
test('stable softmax preserves class order without overflow', () => {
  const result = contract.rank([10000,10001], good);
  assert.equal(result[0].name,'b'); assert.ok(Math.abs(result[0].confidence-.73105858)<1e-7);
});
test('fails closed for incompatible model output and invalid probabilities', () => {
  for(const values of [[1],[NaN,1],[Infinity,0]]) assert.throws(() => contract.rank(values, good));
  assert.throws(() => contract.rank([.9,.9], {...good,output_type:'probabilities'}));
  assert.equal(contract.rank([.1,.9], {...good,output_type:'probabilities'})[0].name,'b');
});
