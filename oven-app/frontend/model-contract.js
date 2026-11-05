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
  const api = {validate, rank};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.modelContract = api;
})(typeof window === 'undefined' ? this : window);
