let predictionController;
const resultPanel = document.createElement('div');
resultPanel.setAttribute('aria-live', 'polite');
$('form').prepend(resultPanel);
const restartButton = document.createElement('button');
restartButton.id = 'restart';
restartButton.type = 'button';
restartButton.className = 'primary full';
restartButton.style.marginTop = '12px';
restartButton.textContent = '识别下一张';
restartButton.hidden = true;
$('draft').after(restartButton);
restartButton.onclick = () => {
  reset();
  $('pick').focus();
};
const unknown = '以上都不是';
const uncertainChoice = '不确定，需要人工确认';
const clearPrediction = () => {
  if (predictionController) predictionController.abort();
  resultPanel.textContent = '';
};
const originalReset = reset;
reset = function () { clearPrediction(); originalReset(); restartButton.hidden = true; };
$('remove').onclick = reset;
const originalLoad = load;
load = async function(file) {
  if (!file) return;
  clearPrediction();
  reset();
  const pending = originalLoad(file);
  const expectedVersion = loadVersion;
  await pending;
  if (loadVersion !== expectedVersion) return;
  if (!objectUrl) return;
  restartButton.hidden = false;
  const version = loadVersion;
  const controller = new AbortController();
  predictionController = controller;
  resultPanel.textContent = window.browserModel ? '正在加载模型并识别，首次使用请稍候……' : '正在识别，请稍候……';
  const timeout = setTimeout(() => controller.abort(), window.browserModel ? 120000 : 30000);
  try {
    let data;
    if (window.browserModel) { data = await window.browserModel.predict(file, controller.signal); } else {
    const body = new FormData(); body.append('image', file);
    const response = await fetch('/api/predict', {method:'POST', body, signal:controller.signal});
    data = await response.json();
    if (version !== loadVersion) return;
    if (!response.ok) throw new Error(data.error || '识别失败');
    }
    if (version !== loadVersion || controller.signal.aborted) return;
    resultPanel.replaceChildren();
    const message = document.createElement('p');
    message.textContent = data.uncertain ? '结果不确定，请人工确认。' : '识别完成，请核对后手动选择食材。';
    resultPanel.append(message);
    for (const candidate of data.candidates) {
      const row = document.createElement('p');
      row.textContent = `${candidate.name} · 预测置信度 ${(candidate.confidence*100).toFixed(1)}%`;
      resultPanel.append(row);
    }
    const note = document.createElement('p'); note.textContent = data.note; resultPanel.append(note);
  } catch (error) {
    if (version === loadVersion) resultPanel.textContent = error.name === 'AbortError' ? '识别超时，请重新上传或手动确认。' : error.message;
  } finally { clearTimeout(timeout); }
};
$('file').onchange = e => load(e.target.files[0]);
(window.browserModel ? window.browserModel.classes().then(classes => ({ok:true,json:async()=>({classes})})) : fetch('/api/classes')).then(async response => {
  if (!response.ok) throw new Error('无法读取模型类别');
  const data = await response.json();
  $('ingredient').replaceChildren(new Option('请选择并确认食材', ''));
  for (const name of [...data.classes, uncertainChoice, unknown]) $('ingredient').add(new Option(name,name));
}).catch(() => { $('error').textContent = '模型加载失败，请检查网络连接并刷新页面。'; });
const originalNext = $('next').onclick;
$('next').onclick = () => {
  if ([unknown, uncertainChoice].includes($('ingredient').value)) {
    $('draft').textContent = '已标记：'+$('ingredient').value+'。请重新拍摄或由人工确认；不生成烹饪设置。';
    $('draft').style.display = 'block';
    $('step3').classList.remove('active');
    return;
  }
  originalNext();
};
