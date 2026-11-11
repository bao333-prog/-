(() => {
  'use strict';
  const video=$('camera-video'),media=navigator.mediaDevices;
  let cameraMode=false,connected=false,connecting=false,capturing=false,continuous=false,loopTimer,loopVersion=0,captureVersion=0,wantsReconnect=false,listVersion=0;
  let remembered=''; try { remembered=localStorage.getItem('oven-camera-device')||''; } catch {}
  const select=$('camera-device');
  const say=message=>{$('camera-status').textContent=message;};
  const usable=()=>cameraMode && connected && video.readyState>=2 && video.videoWidth>0;
  const controls=()=>{
    $('camera-connect').disabled=connecting || connected;
    $('camera-stop').disabled=!connecting && !connected;
    select.disabled=connecting;
    $('camera-snapshot').disabled=!usable() || capturing || continuous || busy;
    $('camera-continuous').disabled=!usable() || (capturing && !continuous) || (!continuous && (busy || !classesReady));
    $('camera-continuous').textContent=continuous?'停止连续识别':'开始连续识别';
  };
  function stopLoop() {continuous=false; ++loopVersion; clearTimeout(loopTimer);controls();}
  function cancelFrame() {++captureVersion;++loadVersion;clearPrediction();clearDraft();enable();if(cameraMode && classesReady) $('model-status').textContent='等待摄像头截图';}
  function disconnected() {stopLoop();cancelFrame();connected=false;connecting=false;video.srcObject=null;controls();}
  const session=new window.ovenCamera.CameraSession(media,(state,message,stream)=>{
    if(state==='connecting') {connecting=true;say(message);controls();return;}
    if(state==='connected') {
      connecting=false;connected=true;video.srcObject=stream;
      video.play().catch(()=>{if(session.stream===stream) session.stop('预览无法播放，请重试连接。');});
      const id=stream.getVideoTracks()[0].getSettings().deviceId;
      if(id) {remembered=id;try{localStorage.setItem('oven-camera-device',id);}catch{}}
      say('摄像头已连接，正在等待画面…');refreshDevices();controls();return;
    }
    if(state==='error') wantsReconnect=false;
    disconnected();say(message);
  });
  async function refreshDevices() {
    if(!media?.enumerateDevices) return;
    const version=++listVersion;
    try {
      const devices=(await media.enumerateDevices()).filter(device=>device.kind==='videoinput');
      if(version!==listVersion) return;
      const selected=select.value || remembered;
      select.replaceChildren(new Option('浏览器默认摄像头（首次授权后可切换）',''));
      devices.forEach((device,index)=>select.add(new Option(device.label||'摄像头 '+(index+1),device.deviceId)));
      if(selected && !devices.some(d=>d.deviceId===selected)) select.add(new Option('上次选择的摄像头（当前未检测到）',selected));
      select.value=selected;
      $('camera-devices-note').textContent=devices.length?'检测到 '+devices.length+' 个视频设备；请确认选择的是烤箱摄像头。':'未检测到视频设备；请插入 USB，或先连接以授权设备访问。';
      if(!connected && !connecting && cameraMode && wantsReconnect && $('camera-reconnect').checked && selected && devices.some(d=>d.deviceId===selected) && !document.hidden) connect();
    } catch {say('无法刷新设备列表，请检查浏览器权限。');}
  }
  async function connect() {
    if(connecting || !cameraMode) return;
    if(!window.isSecureContext || !media?.getUserMedia) {say('此页面无法调用摄像头，请用 localhost 或 HTTPS 打开，并使用支持摄像头的浏览器。');return;}
    wantsReconnect=true;await session.start(select.value);controls();
  }
  function setMode(camera) {
    cameraMode=camera;wantsReconnect=false;stopLoop();session.stop();reset();
    $('upload-pane').hidden=camera;$('camera-pane').hidden=!camera;
    $('input-mode-file').setAttribute('aria-pressed',String(!camera));$('input-mode-camera').setAttribute('aria-pressed',String(camera));
    $('input-heading').textContent=camera?'摄像头识别':'上传食材';
    if(camera) {say('先连接并选择烤箱摄像头；首次使用需允许访问。');refreshDevices();} controls();
  }
  async function snapshot() {
    if(!usable() || capturing || busy) return;
    capturing=true;const version=captureVersion;controls();
    try {
      const file=await window.ovenCamera.captureFrame(video);
      if(version!==captureVersion || !usable()) return;
      $('camera-capture-note').textContent='最近截图：'+new Date().toLocaleTimeString()+' · 结果对应截图时刻，不代表当前实时画面。';
      await load(file);
    } catch(error) {if(version===captureVersion) {say(error.message);stopLoop();}}
    finally {capturing=false;controls();}
  }
  async function nextFrame(version) {
    if(!continuous || version!==loopVersion || !usable() || document.hidden) return;
    await snapshot();
    if(continuous && version===loopVersion) {
      if(!classesReady || !retryButton.hidden) {stopLoop();say('识别未完成或失败，连续识别已暂停，请重试。');return;}
      loopTimer=setTimeout(()=>nextFrame(version),2000);
    }
  }
  $('input-mode-file').onclick=()=>setMode(false);$('input-mode-camera').onclick=()=>setMode(true);
  $('camera-connect').onclick=connect;
  $('camera-stop').onclick=()=>{wantsReconnect=false;session.stop();};
  $('camera-refresh').onclick=refreshDevices;
  select.onchange=()=>{wantsReconnect=false;session.stop('设备已切换，请点击连接摄像头。');};
  $('camera-snapshot').onclick=snapshot;
  $('camera-continuous').onclick=()=>{if(continuous) {stopLoop();cancelFrame();say('连续识别已停止，预览保持开启。');} else if(usable() && classesReady && !busy){continuous=true;controls();nextFrame(++loopVersion);}};
  // A manual selection must remain stable while the user confirms it.
  $('ingredient').addEventListener('change',stopLoop);
  $('next').addEventListener('click',stopLoop);
  restartButton.addEventListener('click',()=>{stopLoop();cancelFrame();controls();});
  video.addEventListener('loadeddata',()=>{if(connected){say('实时预览已就绪，请点击识别当前画面。');controls();}});
  video.addEventListener('error',()=>session.stop('视频预览异常，请重新连接。'));
  media?.addEventListener?.('devicechange',refreshDevices);
  document.addEventListener('visibilitychange',()=>{if(document.hidden){wantsReconnect=false;session.stop('页面进入后台，摄像头已关闭；返回后请重新连接。');}});
  window.addEventListener('pagehide',()=>{wantsReconnect=false;session.stop();});
  const baseEnable=enable;enable=function(){baseEnable();controls();};
  controls();
})();
