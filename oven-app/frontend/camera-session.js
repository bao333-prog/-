(function(root) {
  'use strict';
  class CameraSession {
    constructor(media, onState, timeoutMs=20000) {
      this.media=media; this.onState=onState; this.timeoutMs=timeoutMs; this.generation=0; this.stream=null;
    }
    stop(message='摄像头已关闭') {
      ++this.generation;
      const stream=this.stream; this.stream=null;
      if(stream) stream.getTracks().forEach(track=>track.stop());
      this.onState('stopped',message);
    }
    async start(deviceId='') {
      this.stop(); const generation=this.generation;
      this.onState('connecting','正在连接，请在浏览器提示中允许摄像头访问…');
      let timer;
      try {
        const request=this.media.getUserMedia({audio:false,video:{...(deviceId?{deviceId:{exact:deviceId}}:{}),width:{ideal:1280},height:{ideal:720}}});
        const guarded=request.then(stream=>{
          if(generation!==this.generation) { stream.getTracks().forEach(track=>track.stop()); return null; }
          return stream;
        });
        const stream=await Promise.race([guarded,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Object.assign(new Error('连接超时，请处理权限提示后重试'),{name:'CameraTimeout'})),this.timeoutMs);})]);
        if(generation!==this.generation || !stream) return null;
        const tracks=stream.getVideoTracks();
        if(!tracks.length || tracks[0].readyState==='ended') { stream.getTracks().forEach(track=>track.stop()); throw Error('摄像头没有可用的视频轨道'); }
        this.stream=stream;
        tracks[0].addEventListener('ended',()=>{
          if(this.stream!==stream) return;
          this.stop('摄像头连接已断开，请检查 USB');
          this.onState('disconnected','摄像头连接已断开，请检查 USB');
        },{once:true});
        this.onState('connected','摄像头已连接',stream);
        return stream;
      } catch(error) {
        if(generation!==this.generation) return null;
        this.stop();
        const messages={NotAllowedError:'摄像头权限被拒绝，请在浏览器网站权限中允许后重试。',NotFoundError:'没有找到摄像头，请插入烤箱 USB 并刷新设备列表。',NotReadableError:'摄像头无法读取，可能正被“相机”或其他软件占用。',OverconstrainedError:'所选摄像头不可用，请检查 USB 或重新选择设备。',SecurityError:'浏览器禁止摄像头访问，请使用 HTTPS 或 localhost。'};
        this.onState('error',messages[error.name] || error.message || '摄像头连接失败');
        return null;
      } finally { clearTimeout(timer); }
    }
  }
  async function captureFrame(video, createCanvas=()=>document.createElement('canvas'), FileType=File) {
    if(video.readyState<2 || !video.videoWidth || !video.videoHeight) throw Error('画面尚未就绪，请稍候再识别');
    const scale=Math.min(1,1280/Math.max(video.videoWidth,video.videoHeight));
    const canvas=createCanvas(); canvas.width=Math.max(1,Math.round(video.videoWidth*scale)); canvas.height=Math.max(1,Math.round(video.videoHeight*scale));
    const context=canvas.getContext('2d'); if(!context) throw Error('浏览器无法截取画面');
    context.imageSmoothingEnabled=true;context.imageSmoothingQuality='high';
    context.drawImage(video,0,0,canvas.width,canvas.height);
    const blob=await new Promise((resolve,reject)=>canvas.toBlob(value=>value?resolve(value):reject(Error('截图失败，请重试')),'image/jpeg',.9));
    return new FileType([blob],'摄像头-'+Date.now()+'.jpg',{type:'image/jpeg'});
  }
  const api={CameraSession,captureFrame};
  if(typeof module!=='undefined' && module.exports) module.exports=api; else root.ovenCamera=api;
})(typeof window==='undefined'?this:window);
