import { byId, notice, request, sessionURL } from './common.js';

const MAX_SESSIONS = 194;
const svgNS = 'http://www.w3.org/2000/svg';

function element(name, attrs = {}) {
  const node = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

function safeColor(color) {
  return typeof color === 'string' && /^#[0-9a-f]{6}$/i.test(color) ? color : '#202824';
}

function pathData(points, closed = false) {
  if (!Array.isArray(points) || points.length < (closed ? 3 : 2)) return '';
  if (!points.every(p => Array.isArray(p) && Number.isFinite(p[0]) && Number.isFinite(p[1]))) return '';
  return points.map(([x, y], i) => (i ? 'L' : 'M') + x + ' ' + y).join(' ') + (closed ? ' Z' : '');
}

function contours(rings, closed) {
  return Array.isArray(rings) ? rings.map(points => pathData(points, closed)).filter(Boolean).join(' ') : '';
}

function visibleStrokes(events, seq) {
  const strokes = new Map();
  for (const event of events) {
    if (event.seq > seq) break;
    for (const change of event.changes || []) {
      if (change.op === 'remove') strokes.delete(change.id);
      else if (change.stroke) strokes.set(change.id, { ...change.stroke, id: change.id,
        identity_ambiguous: change.identity_ambiguous ?? change.stroke.identity_ambiguous });
    }
  }
  return strokes;
}

function drawStroke(stroke, hits, ink) {
  // Every coordinate, including a hole or a branch, is already in board pixels.
  const outline = contours(stroke.outline, true);
  const branches = (stroke.centerlines || []).map(branch =>
    pathData(branch.points ?? branch, branch.closed === true)).filter(Boolean).join(' ');
  const primary = pathData(stroke.points, stroke.closed);
  const centerline = branches || primary;
  const layers = (stroke.layers || []).map(layer => ({
    d: contours(layer.outline, true), color: safeColor(layer.color),
  })).filter(layer => layer.d);
  const hasSurface = layers.length > 0 || Boolean(outline);
  if (!centerline && !hasSurface) return;
  const color = safeColor(stroke.color);
  const width = stroke.width > 0 ? stroke.width : 1;
  const group = element('g', { id: stroke.id, 'data-stroke-id': stroke.id, 'data-width': width, 'data-color': color });
  const appearance = stroke.appearance || {};
  if (Number.isFinite(appearance.opacity) && appearance.opacity >= 0 && appearance.opacity <= 1) {
    group.setAttribute('opacity', appearance.opacity);
  }
  const cap = ['butt', 'square', 'round'].includes(appearance.cap) ? appearance.cap : 'round';
  const title = element('title');
  title.textContent = stroke.id;
  group.append(title);
  if (layers.length) {
    for (const layer of layers) group.append(element('path', {
      d: layer.d, fill: layer.color, 'fill-rule': 'evenodd', 'data-geometry': 'outline',
    }));
    if (outline) group.append(element('path', { d: outline, display: 'none', 'data-geometry': 'footprint' }));
  } else if (outline) {
    group.append(element('path', { d: outline, fill: color, 'fill-rule': 'evenodd', 'data-geometry': 'outline' }));
  }
  if (centerline) {
    const path = element('path', { d: centerline, 'stroke-linecap': cap, 'stroke-linejoin': 'round',
      stroke: color, 'stroke-width': width, 'data-geometry': 'line' });
    path.setAttribute('fill', 'none');
    if (hasSurface) path.setAttribute('display', 'none');
    group.append(path);
    if (branches && primary) group.append(element('path', { d: primary, display: 'none', 'data-geometry': 'primary' }));
    hits.append(element('path', { d: centerline, 'data-hit-id': stroke.id, fill: 'none', stroke: 'transparent',
      'stroke-width': 10, 'vector-effect': 'non-scaling-stroke', 'pointer-events': 'stroke' }));
  }
  ink.append(group);
}

function timeLabel(value) {
  return Number.isFinite(value) ? new Date(value).toLocaleString('zh-CN', { hour12: false }) : '—';
}

function download(name, type, content) {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([content], { type }));
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

async function readJSON(url, signal) {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal.addEventListener('abort', cancel, { once: true });
  const timer = setTimeout(cancel, 20000);
  try {
    if (signal.aborted) cancel();
    const response = await fetch(url, { signal: controller.signal, cache: 'no-store' });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || '服务返回 HTTP ' + response.status);
    return { status: response.status, body };
  } catch (error) {
    if (controller.signal.aborted && !signal.aborted) throw new Error('本机服务响应超时');
    if (error instanceof TypeError) throw new Error('无法连接本机服务');
    throw error;
  } finally {
    clearTimeout(timer);
    signal.removeEventListener('abort', cancel);
  }
}

export function setupInk() {
  const selector = byId('ink-session');
  const methodSelector = byId('ink-method');
  const yoloOption = methodSelector.querySelector('option[value="yolo-experiment"]');
  const slider = byId('ink-seq');
  const svg = byId('ink-svg');
  const sheet = byId('ink-sheet');
  const detail = byId('ink-detail');
  const modeButtons = [...byId('ink-modes').querySelectorAll('button')];
  const zoom = byId('ink-zoom');
  const exportSvg = byId('ink-export-svg');
  const exportJson = byId('ink-export-json');
  let board = byId('ink-board');
  let sessions = [];
  let payload = null;
  let selected = '';
  let seq = 0;
  let picked = null;
  let strokes = new Map();
  let mode = 'ink';
  let recognitionMode = methodSelector.value === 'neural' ? 'neural' : 'classic';
  let loadedMode = 'classic';
  let loadedModelVersion = '';
  let directoryVersion = 0;
  let loadController = null;
  let referenceController = null;
  let referenceTimer = 0;
  let referenceKey = '';

  async function updateExperimentalMethod(id, signal) {
    yoloOption.hidden = true;
    yoloOption.disabled = true;
    try {
      const { body } = await readJSON(sessionURL(id) + '/ink-packages', signal);
      const packageInfo = (body.packages || []).find(item => item.mode === 'yolo-experiment');
      if (packageInfo) {
        yoloOption.hidden = false;
        yoloOption.disabled = false;
        yoloOption.textContent = 'YOLO 分割（实验） · ' + (packageInfo.model_version || '本机预计算');
      } else if (recognitionMode === 'yolo-experiment') {
        recognitionMode = 'classic'; methodSelector.value = 'classic';
      }
    } catch (error) {
      if (!signal.aborted && recognitionMode === 'yolo-experiment') {
        recognitionMode = 'classic'; methodSelector.value = 'classic';
      }
    }
  }

  function show(message = '') { notice(byId('ink-notice'), message); }

  function cancelReference() {
    clearTimeout(referenceTimer);
    referenceController?.abort();
    referenceController = null;
    byId('ink-reference-status').hidden = true;
  }

  function updateLayer() {
    cancelReference();
    sheet.dataset.mode = mode;
    byId('ink-overlay-key').hidden = mode !== 'overlay';
    for (const button of modeButtons) {
      button.disabled = !payload;
      button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
    }
    board.hidden = true;
    if (mode === 'ink' || !payload) return;
    const key = selected + '/' + seq;
    if (referenceKey === key) { board.hidden = false; return; }
    const controller = new AbortController();
    referenceController = controller;
    byId('ink-reference-status').hidden = false;
    // Coalesce slider movements. Never show a reference from a previous sequence.
    referenceTimer = setTimeout(async () => {
      try {
        const { body: state } = await readJSON(sessionURL(selected) + '/state?seq=' + seq, controller.signal);
        const image = new Image();
        image.src = 'data:image/png;base64,' + state.board;
        await image.decode();
        if (controller.signal.aborted) return;
        if (state.seq !== seq || image.naturalWidth !== payload.width || image.naturalHeight !== payload.height) {
          throw new Error('原板序号或尺寸与墨迹不一致');
        }
        referenceKey = key;
        image.id = board.id;
        image.alt = board.alt;
        board.replaceWith(image);
        board = image;
        byId('ink-reference-status').hidden = true;
      } catch (error) {
        if (controller.signal.aborted) return;
        mode = 'ink'; updateLayer();
        show('无法对照此序号的原板。' + error.message);
      }
    }, 80);
  }

  function inspect(id) {
    svg.querySelector('[data-selection]')?.remove();
    svg.querySelector('.selected')?.classList.remove('selected');
    detail.replaceChildren();
    picked = strokes.has(id) ? id : null;
    if (!picked) return;
    const group = [...svg.querySelectorAll('[data-stroke-id]')].find(node => node.dataset.strokeId === picked);
    if (!group) return;
    const stroke = strokes.get(picked);
    group.classList.add('selected');
    const box = group.getBBox();
    const margin = 4 + (group.querySelector('[data-geometry="outline"]') ? 0 : Number(stroke.width || 1) / 2);
    svg.append(element('rect', { x: box.x - margin, y: box.y - margin, width: box.width + margin * 2,
      height: box.height + margin * 2, fill: 'none', stroke: '#a54a20', 'stroke-dasharray': '4 3',
      'vector-effect': 'non-scaling-stroke', 'pointer-events': 'none', 'data-ui': '', 'data-selection': '' }));
    const first = payload.events.find(event => event.seq === stroke.first_seq);
    const direction = stroke.direction === 'forward' ? '正向' : stroke.direction === 'reverse' ? '反向' : '未知';
    const rows = [['墨迹 ID', stroke.id], ['代表墨色', safeColor(stroke.color)], ['代表线宽', stroke.width + ' px'],
      ['首次确认', 'seq ' + stroke.first_seq], ['确认时间', timeLabel(first?.confirmed_at)],
      ['中心线点数', (stroke.centerlines || [stroke.points || []]).reduce((n, line) => n + (line.points ?? line).length, 0)],
      ['方向', direction + (stroke.direction_source === 'inferred' ? '（推断）' : '')]];
    if (stroke.identity_ambiguous || first?.identity_ambiguous) rows.push(['身份', '存在多个可能的前身']);
    for (const [name, value] of rows) {
      const row = document.createElement('div');
      const term = document.createElement('dt');
      const desc = document.createElement('dd');
      term.textContent = name; desc.textContent = value;
      row.append(term, desc); detail.append(row);
    }
  }

  function clear(title, description = '') {
    payload = null; picked = null; strokes.clear(); referenceKey = ''; loadedMode = 'classic'; loadedModelVersion = '';
    recognitionMode = 'classic'; methodSelector.value = 'classic'; yoloOption.hidden = true; yoloOption.disabled = true;
    board.removeAttribute('src');
    mode = 'ink'; updateLayer();
    svg.replaceChildren(); detail.replaceChildren();
    byId('ink-empty-title').textContent = title;
    byId('ink-empty-description').textContent = description;
    byId('ink-empty').hidden = false;
    byId('ink-canvas').hidden = true;
    byId('ink-status').textContent = '未加载';
    byId('ink-seq-output').textContent = '—';
    slider.value = '0'; slider.disabled = true;
    zoom.disabled = exportSvg.disabled = exportJson.disabled = true;
  }

  function render() {
    const { width, height } = payload;
    // The SVG viewport and the raster reference share a single responsive sheet.
    svg.setAttribute('width', width); svg.setAttribute('height', height);
    svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
    sheet.style.setProperty('--ink-native-width', width + 'px');
    strokes = visibleStrokes(payload.events, seq);
    const hits = element('g', { 'data-ui': '' });
    const ink = element('g');
    for (const stroke of strokes.values()) drawStroke(stroke, hits, ink);
    svg.replaceChildren(element('rect', { width, height, fill: '#ffffff', 'data-paper': '' }), hits, ink);
    byId('ink-empty').hidden = true;
    byId('ink-canvas').hidden = false;
    slider.value = String(seq);
    byId('ink-seq-output').textContent = seq + ' / ' + payload.through_seq;
    const modelLabel = loadedModelVersion ? ' · ' + loadedModelVersion : '';
    byId('ink-status').textContent = width + ' × ' + height + ' px · ' + (ink.childElementCount ? ink.childElementCount + ' 条墨迹' : '此序号暂无已确认墨迹') + modelLabel;
    zoom.disabled = exportSvg.disabled = exportJson.disabled = false;
    inspect(picked);
    updateLayer();
  }

  async function loadSession(id, method, controller) {
    while (!controller.signal.aborted) {
      const { status, body: result } = await readJSON(sessionURL(id) + '/ink?mode=' + encodeURIComponent(method), controller.signal);
      if (controller.signal.aborted) return null;
      if (status === 202 || result.ready === false || result.status === 'pending') {
        await new Promise(resolve => setTimeout(resolve, 500)); continue;
      }
      if (result.version !== 1 || !Array.isArray(result.events) || !(result.width > 0 && result.height > 0)) {
        throw new Error('电子墨迹格式无效');
      }
      return result;
    }
    return null;
  }

  async function selectSession(id) {
    loadController?.abort();
    const controller = new AbortController();
    loadController = controller;
    selected = id; selector.value = id;
    clear('正在读取电子墨迹', '首次打开时会从这次会话生成笔迹。'); show();
    try {
      await updateExperimentalMethod(id, controller.signal);
      if (controller.signal.aborted) return;
      const result = await loadSession(id, recognitionMode, controller);
      if (!result || controller.signal.aborted) return;
      payload = result; loadedMode = recognitionMode; loadedModelVersion = typeof result.model_version === 'string' ? result.model_version : ''; seq = Number(result.through_seq);
      slider.max = String(seq); slider.disabled = seq === 0;
      render();
    } catch (error) {
      if (controller.signal.aborted) return;
      clear('无法读取电子墨迹', '刷新会话可重试。'); show(error.message);
    } finally { if (loadController === controller) loadController = null; }
  }

  async function rederive() {
    if (!selected || !payload) return;
    loadController?.abort();
    const controller = new AbortController();
    loadController = controller;
    const previous = payload;
    const previousSeq = seq;
    const method = recognitionMode;
    const label = method === 'neural' ? '淡墨增强（实验）' : method === 'yolo-experiment' ? 'YOLO 分割（实验）' : '标准 · OpenCV';
    byId('ink-status').textContent = '正在按 ' + label + ' 重新派生…';
    show('正在重新派生电子墨迹，当前结果仍可查看…');
    try {
      const result = await loadSession(selected, method, controller);
      if (!result || controller.signal.aborted) return;
      payload = result; loadedMode = method; loadedModelVersion = typeof result.model_version === 'string' ? result.model_version : '';
      seq = Math.min(previousSeq, Number(result.through_seq));
      slider.max = String(Number(result.through_seq)); slider.disabled = seq === 0;
      render(); show('');
    } catch (error) {
      if (!controller.signal.aborted) { payload = previous; seq = previousSeq; render(); show('重新派生失败，已保留原结果。' + error.message); }
    } finally { if (loadController === controller) loadController = null; }
  }

  svg.addEventListener('click', event => {
    const hit = event.target.closest('[data-hit-id], [data-stroke-id]');
    inspect(hit?.dataset.hitId || hit?.dataset.strokeId);
  });
  slider.addEventListener('input', () => { if (payload) { seq = Number(slider.value); render(); } });
  selector.addEventListener('change', () => void selectSession(selector.value));
  methodSelector.addEventListener('change', () => { recognitionMode = methodSelector.value; void rederive(); });
  for (const button of modeButtons) button.addEventListener('click', () => {
    mode = button.dataset.mode; show(); updateLayer();
  });
  zoom.addEventListener('click', () => {
    const enlarged = sheet.classList.toggle('enlarged');
    zoom.setAttribute('aria-pressed', String(enlarged)); zoom.textContent = enlarged ? '适应宽度' : '放大';
  });
  exportJson.addEventListener('click', () => {
    if (payload) download('ink-' + selected + '-seq-' + seq + '.json', 'application/json', JSON.stringify({ ...payload, view_seq: seq, recognition_mode: loadedMode, model_version: loadedModelVersion || undefined }, null, 2));
  });
  exportSvg.addEventListener('click', () => {
    if (!payload) return;
    const copy = svg.cloneNode(true);
    copy.querySelectorAll('[data-ui]').forEach(node => node.remove());
    copy.querySelectorAll('.selected').forEach(node => node.classList.remove('selected'));
    copy.removeAttribute('id');
    copy.setAttribute('aria-label', '电子墨迹 ' + selected + ' · seq ' + seq);
    const metadata = document.createElementNS(svgNS, 'metadata');
    metadata.textContent = JSON.stringify({ session_id: selected, seq, recognition_mode: loadedMode, model_version: loadedModelVersion || undefined });
    copy.insertBefore(metadata, copy.firstChild);
    download('ink-' + selected + '-seq-' + seq + '.svg', 'image/svg+xml', new XMLSerializer().serializeToString(copy));
  });
  byId('ink-refresh').addEventListener('click', () => void open(true));

  async function open(refresh = false) {
    const version = ++directoryVersion;
    try {
      const result = await request('/api/sessions');
      if (version !== directoryVersion) return;
      sessions = (result.sessions || []).slice().sort((a, b) => b.created_at - a.created_at).slice(0, MAX_SESSIONS);
      selector.replaceChildren(...sessions.map(s => new Option(timeLabel(s.created_at) + ' · seq ' + s.seq, s.id)));
      selector.disabled = !sessions.length;
      if (!sessions.length) { clear('暂无采集记录', '完成一次白板采集后即可查看墨迹。'); return; }
      const session = sessions.find(s => s.id === selected) || sessions[0];
      selector.value = session.id;
      if (refresh || !payload || session.id !== selected || payload.through_seq !== session.seq) await selectSession(session.id);
      else updateLayer();
    } catch (error) {
      if (version !== directoryVersion) return;
      clear('无法读取会话目录', '请确认本机服务正在运行。'); show(error.message);
    }
  }
  function close() {
    directoryVersion++; loadController?.abort(); cancelReference(); show();
  }
  clear('正在读取会话');
  return { open, close };
}
