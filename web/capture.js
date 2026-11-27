import { Board, byId, notice, request, sessionURL } from './common.js';

const corners = ['左上', '右上', '右下', '左下'];
const targetFps = 30;
const frameInterval = 1000 / targetFps;

function cameraError(error) {
  const messages = {
    NotAllowedError: '摄像头权限未获允许。请在浏览器的地址栏和系统隐私设置中允许摄像头访问，再重试。',
    NotFoundError: '没有找到摄像头。请接入设备后重试。',
    NotReadableError: '无法读取摄像头。请检查设备连接，或关闭正在占用摄像头的应用后重试。',
    OverconstrainedError: '所选摄像头已不可用。请选择其他设备后重试。',
    AbortError: '摄像头连接中断，请检查设备后重试。',
    SecurityError: '浏览器阻止了摄像头访问。请使用本机地址打开工作台并检查权限。',
  };
  return messages[error.name] || '摄像头连接失败：' + error.message;
}

function validQuad(points) {
  if (points.length !== 4) return false;
  const [tl, tr, br, bl] = points;
  return tl[1] + tr[1] < bl[1] + br[1] && tl[0] + bl[0] < tr[0] + br[0] && points.every(([x, y], i) => {
    const [bx, by] = points[(i + 1) % 4];
    const [cx, cy] = points[(i + 2) % 4];
    return (bx - x) * (cy - by) - (by - y) * (cx - bx) > 0;
  });
}

export function setupCapture() {
  const video = byId('video');
  const plane = byId('video-plane');
  const finder = byId('viewfinder');
  const device = byId('device');
  const record = byId('record');
  const connect = byId('connect');
  const alert = message => notice(byId('live-notice'), message);
  const board = new Board(byId('live-board'));
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d', { alpha: false });
  let stream = null;
  let points = [];
  let connecting = false;
  let videoWaiting = false;
  let cameraVersion = 0;
  let devicesVersion = 0;
  let pendingHint = '';
  let phase = 'idle';
  let sessionId = null;
  let creation = null;
  let frame = null;
  let stopSampling = null;
  let inputWidth = 0;
  let inputHeight = 0;
  let completed = false;

  function drawPoints() {
    const overlay = byId('calibration-overlay');
    overlay.replaceChildren();
    if (points.length >= 2) {
      const shape = document.createElementNS('http://www.w3.org/2000/svg', points.length === 4 ? 'polygon' : 'polyline');
      shape.setAttribute('points', points.map(([x, y]) => [x * 1000, y * 1000].join(',')).join(' '));
      overlay.append(shape);
    }
    const marks = points.map(([x, y], index) => {
      const mark = document.createElement('span');
      mark.className = 'point-marker';
      mark.textContent = index + 1;
      mark.style.left = x * 100 + '%';
      mark.style.top = y * 100 + '%';
      return mark;
    });
    byId('point-marks').replaceChildren(...marks);
  }

  function sizePlane() {
    if (!video.videoWidth || !video.videoHeight) return;
    const scale = Math.min(finder.clientWidth / video.videoWidth, finder.clientHeight / video.videoHeight);
    plane.style.width = video.videoWidth * scale + 'px';
    plane.style.height = video.videoHeight * scale + 'px';
  }

  function render() {
    const ready = Boolean(stream && !connecting && video.videoWidth);
    const muted = stream?.getVideoTracks()[0]?.muted || videoWaiting;
    const idle = phase === 'idle';
    const monitoring = ['recording', 'stopping', 'retry'].includes(phase);
    const actions = byId(monitoring ? 'session-actions' : 'control-body');
    if (record.parentElement !== actions) actions.append(record);
    byId('capture-view').classList.toggle('monitoring', monitoring);
    byId('nav-live').classList.toggle('is-recording', phase === 'recording');
    byId('live-description').textContent = monitoring ? '对照原始画面与数字板书。暂缓更新的区域保留上一次确认的状态。' : '连接摄像头，校准白板四角，持续记录板书变化。';
    const valid = validQuad(points);
    const edit = ready && idle && !muted;
    plane.hidden = !ready;
    plane.classList.toggle('editable', edit && points.length < 4);
    byId('camera-idle').hidden = ready;
    connect.disabled = connecting || !idle;
    connect.querySelector('span').textContent = connecting ? '正在连接…' : '开启摄像头';
    byId('camera-hint').textContent = connecting ? '请在浏览器提示中选择允许' : pendingHint || '首次连接时，请允许浏览器使用摄像头';
    byId('input-label').hidden = Boolean(stream || connecting);
    byId('release').hidden = (!idle && phase !== 'starting') || (!stream && !connecting);
    byId('release').textContent = connecting ? '取消连接' : phase === 'starting' ? '取消并释放' : '释放摄像头';
    byId('input-status').textContent = muted ? '输入暂时中断' : ready ? '实时输入' : connecting ? '连接中' : '无输入信号';
    finder.querySelector('.hud').classList.toggle('signal', ready && !muted);
    byId('resolution').textContent = ready ? video.videoWidth + ' × ' + video.videoHeight + ' px' : '等待视频输入';
    byId('point-count').textContent = points.length + ' / 4';
    byId('calibration-status').textContent = valid ? '已校准' : points.length ? '校准中' : '未校准';
    let guide = '开启摄像头后，在预览中依次点选四角。';
    if (ready) guide = valid ? '四角已就绪。请保持摄像头位置固定。' : points.length === 4 ? '角点顺序有误，请撤销或重置后重选。' : '请在预览中点选白板的' + corners[points.length] + '角。';
    byId('point-instruction').textContent = guide;
    document.querySelectorAll('.corner-steps li').forEach((step, index) => {
      step.classList.toggle('done', index < points.length);
      step.classList.toggle('current', edit && index === points.length);
    });
    byId('reset').disabled = byId('undo').disabled = !idle || !points.length;
    device.disabled = !idle || connecting;
    const action = { idle: '开始采集', starting: '正在建立会话…', recording: '停止采集', stopping: '正在结束会话…', retry: '重试结束会话' };
    record.querySelector('span').textContent = action[phase];
    record.querySelector('use').setAttribute('href', idle ? '#play' : '#stop');
    record.classList.toggle('running', phase === 'recording' || phase === 'retry');
    record.disabled = idle ? !edit || !valid : phase === 'stopping' || phase === 'starting';
    const status = byId('session-status');
    status.classList.toggle('active', phase === 'recording');
    status.querySelector('span').textContent = { idle: completed ? '会话已结束' : '会话未开始', starting: '会话建立中', recording: '正在采集', stopping: '正在结束', retry: '结束待重试' }[phase];
    drawPoints();
    if (ready) sizePlane();
  }

  async function refreshDevices() {
    const version = ++devicesVersion;
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const list = await navigator.mediaDevices.enumerateDevices();
      if (version !== devicesVersion) return;
      const selected = stream?.getVideoTracks()[0]?.getSettings().deviceId || device.value;
      const inputs = list.filter(item => item.kind === 'videoinput' && item.deviceId);
      const options = inputs.map((item, i) => new Option(item.label || '摄像头 ' + (i + 1), item.deviceId));
      device.replaceChildren(new Option('系统默认摄像头', ''), ...options);
      if (inputs.some(item => item.deviceId === selected)) device.value = selected;
    } catch {
      // Some browsers withhold device names or inventory until permission is granted.
    }
  }

  function releaseCamera() {
    stopSampling?.();
    stopSampling = null;
    cameraVersion++;
    connecting = false;
    videoWaiting = false;
    const previous = stream;
    stream = null;
    for (const track of previous?.getTracks() || []) track.stop();
    video.srcObject = null;
    points = [];
    inputWidth = inputHeight = 0;
    render();
  }

  async function endSession(reason = '') {
    if (phase === 'stopping') return;
    if (phase === 'idle') {
      releaseCamera();
      alert(reason);
      return;
    }
    phase = 'stopping';
    releaseCamera();
    alert(reason);
    await Promise.allSettled([creation, frame]);
    try {
      if (sessionId) await request(sessionURL(sessionId) + '/stop', { method: 'POST' });
      completed = Boolean(sessionId);
      sessionId = null;
      phase = 'idle';
    } catch (error) {
      phase = 'retry';
      alert([reason, '摄像头已释放，结束会话失败：' + error.message].filter(Boolean).join(' '));
    }
    render();
  }

  async function openCamera() {
    releaseCamera();
    pendingHint = '';
    alert('');
    if (!navigator.mediaDevices?.getUserMedia) {
      pendingHint = '浏览器未提供摄像头访问';
      alert('请使用支持摄像头的浏览器，通过 http://127.0.0.1:8765 或 localhost 打开本机工作台。');
      render();
      return;
    }
    const version = cameraVersion;
    const selected = device.value;
    connecting = true;
    render();
    try {
      const constraints = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: targetFps } };
      if (selected) constraints.deviceId = { exact: selected };
      const candidate = await navigator.mediaDevices.getUserMedia({ video: constraints, audio: false });
      if (version !== cameraVersion) {
        candidate.getTracks().forEach(track => track.stop());
        return;
      }
      stream = candidate;
      const track = stream.getVideoTracks()[0];
      if (!track || track.readyState === 'ended') throw new DOMException('设备无视频输入', 'NotFoundError');
      track.addEventListener('ended', () => {
        if (version === cameraVersion) {
          pendingHint = '设备已断开，可重新连接';
          void endSession('摄像头连接中断。请检查设备并重新连接、校准。');
        }
      });
      track.addEventListener('mute', () => {
        if (version !== cameraVersion) return;
        alert('视频输入暂时中断，正在等待摄像头恢复。');
        render();
      });
      track.addEventListener('unmute', () => {
        if (version !== cameraVersion) return;
        alert('');
        render();
      });
      video.srcObject = candidate;
      await video.play();
      if (version !== cameraVersion) return;
      connecting = false;
      inputWidth = video.videoWidth;
      inputHeight = video.videoHeight;
      render();
      void refreshDevices();
    } catch (error) {
      if (version !== cameraVersion) return;
      releaseCamera();
      pendingHint = '连接失败，可重试或更换输入设备';
      alert(cameraError(error));
      render();
      void refreshDevices();
    }
  }

  function startSampling(id) {
    const inputRate = byId('sampling-fps');
    const rate = byId('processing-fps');
    const glare = byId('glare-notice');
    const startedAt = performance.now();
    const windowMs = 3000;
    let samples = [];
    let completions = [];
    let lastInput = null;
    let pendingStamp = null;
    let drawnSeq = null;
    let nextSampleAt = null;
    let timer = null;
    let videoCallback = null;
    let reader = null;
    const track = stream.getVideoTracks()[0];
    const negotiatedFps = track.getSettings().frameRate;
    const sourcePaced = negotiatedFps > 0 && negotiatedFps <= targetFps;
    const hasPresentationCallback = typeof video.requestVideoFrameCallback === 'function';
    const active = () => phase === 'recording' && sessionId === id;

    byId('processing-rate').hidden = false;
    rate.textContent = inputRate.textContent = '—';
    const rateTimer = setInterval(() => {
      const now = performance.now();
      const seconds = Math.min(windowMs, now - startedAt) / 1000;
      samples = samples.filter(time => time > now - windowMs);
      completions = completions.filter(time => time > now - windowMs);
      inputRate.textContent = (samples.length / seconds).toFixed(1);
      rate.textContent = (completions.length / seconds).toFixed(1);
    }, 1000);

    async function processFrame(t) {
      // toBlob snapshots the selected canvas bitmap before serializing it asynchronously.
      const jpeg = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.88));
      if (!active()) return;
      if (!jpeg || jpeg.type !== 'image/jpeg') throw new Error('浏览器无法生成 JPEG 画面');
      const seen = drawnSeq === null ? '' : '&seen_seq=' + drawnSeq;
      const state = await request(sessionURL(id) + '/frames?t=' + t + seen, {
        method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: jpeg,
      });
      if (!active()) return;
      completions.push(performance.now());
      glare.hidden = !(Number.isFinite(state.glare) && state.glare > 0);
      if (state.seq === drawnSeq) {
        board.setHeld(state.held);
        return;
      }
      if (state.board === undefined && state.unknown === undefined) {
        // Ask for a full state if a compact response references an unseen board.
        drawnSeq = null;
        board.setHeld(state.held);
        return;
      }
      await board.update(state);
      drawnSeq = state.seq;
    }

    function sendLatest() {
      if (!active() || frame || pendingStamp === null) return;
      const stamp = pendingStamp;
      pendingStamp = null;
      const job = processFrame(stamp);
      frame = job;
      job.then(() => {
        if (frame === job) frame = null;
        sendLatest();
      }, error => {
        if (frame === job) frame = null;
        if (active()) void endSession('采集已中断：' + error.message);
      });
    }

    function sample(image, token, mediaMs) {
      if (!active() || token === lastInput || track.muted) return;
      if (image === video && (videoWaiting || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA)) return;
      if (lastInput !== null && token < lastInput) nextSampleAt = null;
      lastInput = token;

      // A negotiated 30 Hz (or slower) source already provides the cadence.
      if (!sourcePaced) {
        if (nextSampleAt !== null && mediaMs < nextSampleAt - 3) return;
        const due = nextSampleAt ?? mediaMs;
        nextSampleAt = due + Math.max(1, Math.floor((mediaMs - due) / frameInterval) + 1) * frameInterval;
      }
      try {
        const stamp = Date.now();
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        pendingStamp = stamp;
        samples.push(performance.now());
        sendLatest();
      } catch (error) {
        void endSession('采集已中断：' + error.message);
      }
    }

    function presented(_now, metadata) {
      if (!active()) return;
      videoCallback = video.requestVideoFrameCallback(presented);
      sample(video, metadata.mediaTime, metadata.mediaTime * 1000);
    }

    function pollFrameCount() {
      if (!active()) return;
      const quality = video.getVideoPlaybackQuality?.();
      const count = quality ? quality.totalVideoFrames - quality.droppedVideoFrames : video.webkitDecodedFrameCount;
      if (count > 0) sample(video, count, performance.now());
      timer = setTimeout(pollFrameCount, frameInterval / 3);
    }

    async function readCameraFrames() {
      try {
        while (active()) {
          const { done, value } = await reader.read();
          if (done) {
            if (active()) void endSession('摄像头视频流已结束，请重新连接。');
            break;
          }
          try {
            sample(value, value.timestamp, value.timestamp / 1000);
          } finally {
            value.close();
          }
        }
      } catch (error) {
        if (active()) void endSession('视频帧读取中断：' + error.message);
      } finally {
        reader.releaseLock();
      }
    }

    stopSampling = () => {
      clearTimeout(timer);
      clearInterval(rateTimer);
      if (reader) void reader.cancel().catch(() => {});
      if (videoCallback !== null) video.cancelVideoFrameCallback(videoCallback);
      byId('processing-rate').hidden = true;
      pendingStamp = null;
      rate.textContent = inputRate.textContent = '—';
      glare.hidden = true;
    };
    if (typeof window.MediaStreamTrackProcessor === 'function') {
      try {
        reader = new MediaStreamTrackProcessor({ track, maxBufferSize: 1 }).readable.getReader();
      } catch {
        // Some browsers expose the processor only for certain track types.
      }
    }
    if (reader) void readCameraFrames();
    else if (hasPresentationCallback) videoCallback = video.requestVideoFrameCallback(presented);
    else if (video.getVideoPlaybackQuality || typeof video.webkitDecodedFrameCount === 'number') pollFrameCount();
    else void endSession('当前浏览器无法识别独立视频帧，请使用支持视频帧读取的浏览器版本。');
  }

  async function startSession() {
    if (phase !== 'idle' || !stream || !validQuad(points) || !video.videoWidth) return;
    phase = 'starting';
    alert('');
    const scale = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    render();
    creation = request('/api/sessions', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ corners: points, width: canvas.width, height: canvas.height }),
    }).then(result => {
      if (!result.id) throw new Error('服务未返回会话编号');
      sessionId = result.id;
    });
    try {
      await creation;
      if (phase !== 'starting') return;
      phase = 'recording';
      completed = false;
      board.clear('等待第一帧', '正在接收本次会话的板书。');
      startSampling(sessionId);
    } catch (error) {
      if (phase === 'starting') {
        phase = 'idle';
        alert('会话建立失败：' + error.message);
      }
    } finally {
      creation = null;
      render();
    }
  }

  connect.addEventListener('click', openCamera);
  byId('release').addEventListener('click', () => { pendingHint = ''; void endSession(); });
  device.addEventListener('change', () => { if (stream) void openCamera(); });
  record.addEventListener('click', () => { if (phase === 'idle') void startSession(); else void endSession(); });
  plane.addEventListener('click', event => {
    if (!stream || connecting || videoWaiting || stream.getVideoTracks()[0].muted || phase !== 'idle' || points.length >= 4) return;
    const rect = plane.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    points.push([x, y]);
    alert('');
    render();
  });
  byId('undo').addEventListener('click', () => { points.pop(); alert(''); render(); });
  byId('reset').addEventListener('click', () => { points = []; alert(''); render(); });
  video.addEventListener('waiting', () => {
    if (!stream || connecting) return;
    videoWaiting = true;
    render();
  });
  video.addEventListener('playing', () => {
    if (!videoWaiting) return;
    videoWaiting = false;
    render();
  });
  video.addEventListener('error', () => {
    if (stream) void endSession('视频输入发生错误。请重新连接设备并校准。');
  });
  video.addEventListener('resize', () => {
    if (!stream || !video.videoWidth || connecting) return;
    if (inputWidth && (inputWidth !== video.videoWidth || inputHeight !== video.videoHeight)) {
      const reason = '摄像头画面尺寸已改变，请重新校准白板四角。';
      if (phase !== 'idle') { void endSession(reason); return; }
      points = [];
      alert(reason);
    }
    inputWidth = video.videoWidth;
    inputHeight = video.videoHeight;
    render();
  });
  new ResizeObserver(sizePlane).observe(finder);
  navigator.mediaDevices?.addEventListener?.('devicechange', refreshDevices);
  window.addEventListener('pagehide', () => {
    if (sessionId) navigator.sendBeacon(sessionURL(sessionId) + '/stop');
    // A pending create can still resolve if the page is kept in the back-forward cache.
    if (phase === 'starting') void endSession();
    else {
      if (sessionId) completed = true;
      sessionId = null;
      phase = 'idle';
      releaseCamera();
    }
  });
  void refreshDevices();
  render();
}
