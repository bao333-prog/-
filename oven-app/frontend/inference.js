let predictionController, currentFile, classesReady = false, busy = false;
const unknown = '以上都不是', uncertainChoice = '不确定，需要人工确认';
const resultPanel = document.createElement('section');
resultPanel.id = 'prediction-result'; resultPanel.setAttribute('aria-live','polite');
$('form').prepend(resultPanel);
const actionBar = document.createElement('div'); actionBar.className = 'prediction-actions';
const retryButton = document.createElement('button'); retryButton.type='button'; retryButton.id='retry'; retryButton.className='text-button'; retryButton.textContent='重试识别'; retryButton.hidden=true;
const restartButton = document.createElement('button'); restartButton.type='button'; restartButton.id='restart'; restartButton.className='text-button'; restartButton.textContent='识别下一张'; restartButton.hidden=true;
actionBar.append(retryButton,restartButton); $('draft').after(actionBar);
const originalEnable = enable;
enable = function() { originalEnable(); $('next').disabled ||= !classesReady || busy; };
function status(text) { $('model-status').textContent=text; }
function clearPrediction() {
  if (predictionController) predictionController.abort();
  predictionController=null; busy=false; resultPanel.replaceChildren(); retryButton.hidden=true;
}
const originalReset = reset;
reset = function() {
  clearPrediction(); currentFile=null; originalReset(); restartButton.hidden=true;
  if(classesReady) status('等待上传图片');
};
$('remove').onclick=reset;
restartButton.onclick=()=>{ reset(); $('pick').focus(); };
function textElement(tag,text,className) { const element=document.createElement(tag); element.textContent=text; if(className) element.className=className; return element; }
async function loadModelInfo() {
  classesReady=false; enable(); $('reload-model').disabled=true;
  status('正在读取模型信息');
  try {
    let data;
    if(window.browserModel) data=await window.browserModel.info();
    else {
      const response=await fetch('/api/classes',{cache:'no-cache'});
      if(!response.ok) throw Error('无法读取模型信息');
      data=await response.json();
    }
    if(!Array.isArray(data.classes) || !data.classes.length) throw Error('模型类别列表为空');
    const selected=$('ingredient').value;
    $('ingredient').replaceChildren(new Option('请选择并确认食材',''));
    for(const name of [...data.classes,uncertainChoice,unknown]) $('ingredient').add(new Option(name,name));
    if([...data.classes,uncertainChoice,unknown].includes(selected)) $('ingredient').value=selected;
    $('model-name').textContent=data.name || data.model || '本机分类模型';
    $('model-version').textContent=data.version || data.model || '未声明版本';
    $('model-count').textContent=data.classes.length+' 类';
    $('model-classes').textContent=data.classes.join(' · ');
    $('model-mode').textContent=window.browserModel?'浏览器本地推理':'本机服务推理';
    classesReady=true;
    status('配置已就绪 · 上传后加载模型');
    $('reload-model').hidden=true;
    if($('error').dataset.modelError) { $('error').textContent=''; delete $('error').dataset.modelError; }
  } catch(error) {
    status('模型信息读取失败');
    $('error').textContent=error.message+'，请重试。'; $('error').dataset.modelError='true';
    $('reload-model').hidden=false;
  } finally { $('reload-model').disabled=false; enable(); }
}
$('reload-model').onclick=()=>loadModelInfo().then(()=>{ if(classesReady && currentFile && !busy) runPrediction(currentFile); });
async function runPrediction(file) {
  clearPrediction(); clearDraft();
  const controller=new AbortController(); predictionController=controller;
  const version=loadVersion, started=performance.now(); busy=true; enable();
  retryButton.hidden=true; resultPanel.textContent='正在准备识别…';
  const active=()=>version===loadVersion && predictionController===controller && !controller.signal.aborted;
  const showStage=message=>{ if(active()) { status(message); resultPanel.textContent=message+'，首次使用请稍候…'; } };
  const timer=setTimeout(()=>{
    if(!active()) return;
    controller.abort(); busy=false; enable();
    resultPanel.textContent='识别超时，请重试或手动确认。'; status('识别超时'); retryButton.hidden=false;
  },window.browserModel?120000:30000);
  try {
    let data;
    if(window.browserModel) data=await window.browserModel.predict(file,controller.signal,showStage);
    else {
      showStage('正在识别食材');
      const body=new FormData(); body.append('image',file);
      const response=await fetch('/api/predict',{method:'POST',body,signal:controller.signal});
      data=await response.json();
      if(!response.ok) throw Error(data.error || '识别失败');
    }
    if(!active()) return;
    if(!Array.isArray(data.candidates) || !data.candidates.length || data.candidates.some(c=>typeof c.name!=='string' || !Number.isFinite(c.confidence) || c.confidence<0 || c.confidence>1)) throw Error('识别结果格式无效');
    const candidateNames=Array.from($('ingredient').options,o=>o.value);
    if(data.candidates.some(c=>!candidateNames.includes(c.name))) throw Error('识别类别与模型配置不一致，请刷新模型信息');
    resultPanel.replaceChildren(textElement('h3',data.uncertain?'结果不确定，请人工确认':'识别完成，请核对后选择食材'));
    const list=document.createElement('ol'); list.className='candidate-list';
    for(const candidate of data.candidates) {
      const row=document.createElement('li');
      row.append(textElement('span',candidate.name),textElement('strong',(candidate.confidence*100).toFixed(1)+'%'));
      const meter=document.createElement('meter'); meter.min=0; meter.max=1; meter.value=candidate.confidence; meter.setAttribute('aria-label',candidate.name+' 预测置信度'); row.append(meter); list.append(row);
    }
    resultPanel.append(list);
    const timing=data.timing || {}, total=performance.now()-started;
    const stats=document.createElement('div'); stats.className='timing';
    stats.append(textElement('span','本次总耗时 '+(total/1000).toFixed(2)+' 秒'));
    if(Number.isFinite(timing.inference_ms)) stats.append(textElement('span','纯推理 '+timing.inference_ms.toFixed(0)+' ms'));
    resultPanel.append(stats,textElement('p','总耗时含加载、排队和图片处理；纯推理为本次设备测量，不代表 RK3506 性能。','result-note'),textElement('p',data.note || '预测置信度不是准确率，请人工确认。','result-note'));
    status('识别完成 · 等待人工确认');
  } catch(error) {
    if(active()) {
      resultPanel.textContent=error.message || '识别失败，请重试';
      status('识别失败 · 可重试或手动确认'); retryButton.hidden=false;
    }
  } finally {
    clearTimeout(timer);
    if(version===loadVersion && predictionController===controller) { busy=false; enable(); }
  }
}
const originalLoad=load;
load=async function(file) {
  if(!file) return;
  reset();
  const pending=originalLoad(file), version=loadVersion;
  await pending;
  if(version!==loadVersion || !objectUrl) return;
  currentFile=file; restartButton.hidden=false;
  if(classesReady) await runPrediction(file);
  else { resultPanel.textContent='请先重试读取模型信息，再进行识别。'; $('reload-model').hidden=false; }
};
$('file').onchange=e=>load(e.target.files[0]);
retryButton.onclick=()=>{if(currentFile) runPrediction(currentFile);};
const originalNext=$('next').onclick;
$('next').onclick=()=>{
  if($('next').disabled) return;
  if([unknown,uncertainChoice].includes($('ingredient').value)) {
    $('draft').textContent='已标记：'+$('ingredient').value+'。请重新上传或由人工确认；不生成烹饪设置。'; $('draft').style.display='block'; $('step3').classList.remove('active'); return;
  }
  originalNext();
};
loadModelInfo();
