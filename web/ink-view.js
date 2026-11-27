import { byId, notice, request, sessionURL } from './common.js';

const MAX_SESSIONS = 194;

function safeColor(value) {
  return typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value) ? value : '#202824';
}

function pathData(points, closed) {
  if (!Array.isArray(points) || !points.length) return '';
  const first = points[0];
  if (!Array.isArray(first) || first.length < 2) return '';
  const commands = ['M', Number(first[0]), Number(first[1])];
  for (const point of points.slice(1)) {
    if (Array.isArray(point) && point.length >= 2 && Number.isFinite(Number(point[0])) && Number.isFinite(Number(point[1]))) {
      commands.push('L', Number(point[0]), Number(point[1]));
    }
  }
  if (closed) commands.push('Z');
  return commands.join(' ');
}

function eventAt(events, seq) {
  return events.filter(event => Number(event.seq) <= seq);
}

function timeLabel(value) {
  return Number.isFinite(Number(value)) ? new Date(Number(value)).toLocaleString('zh-CN', { hour12: false }) : '—';
}

function download(name, type, content) {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([content], { type }));
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

export function setupInk() {
  const root = byId('ink-view');
  const sessionList = byId('ink-session-list');
  const selector = byId('ink-session');
  const slider = byId('ink-seq');
  const sliderOutput = byId('ink-seq-output');
  const svg = byId('ink-svg');
  const status = byId('ink-status');
  const detail = byId('ink-detail');
  const noticeNode = byId('ink-notice');
  const exportSvg = byId('ink-export-svg');
  const exportJson = byId('ink-export-json');
  let sessions = [];
  let payload = null;
  let selected = null;
  let seq = 0;
  let selectedId = null;
  let loadController = null;
  let loadVersion = 0;

  function show(message = '') { notice(noticeNode, message); }

  function clear(title = '选择会话以查看电子墨迹', description = '从左侧选择一次采集记录。') {
    svg.replaceChildren();
    svg.setAttribute('viewBox', '0 0 1000 700');
    byId('ink-empty-title').textContent = title;
    byId('ink-empty-description').textContent = description;
    byId('ink-empty').hidden = false;
    byId('ink-canvas').hidden = true;
    detail.textContent = '尚未选择墨迹';
    slider.disabled = true;
    exportSvg.disabled = true;
    exportJson.disabled = true;
  }

  function render() {
    if (!payload) return clear();
    const width = Number(payload.width) || 1000;
    const height = Number(payload.height) || 700;
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.replaceChildren();
    const strokes = new Map();
    for (const event of eventAt(payload.events || [], seq)) {
      for (const change of event.changes || []) {
        if (!change?.id) continue;
        if (change.op === 'remove') strokes.delete(change.id);
        else if (change.stroke) strokes.set(change.id, { ...change.stroke, id: change.id, identity_ambiguous: change.identity_ambiguous ?? change.stroke.identity_ambiguous });
      }
    }
    for (const stroke of strokes.values()) {
      const d = pathData(stroke.points, stroke.closed);
      if (!d) continue;
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', d);
      // Ink is a centerline representation. A closed path remains a closed
      // outline, while its interior stays transparent in the workbench and
      // in the exported SVG.
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', safeColor(stroke.color));
      path.setAttribute('stroke-width', String(Math.max(0.5, Number(stroke.width) || 2)));
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('stroke-linejoin', 'round');
      path.dataset.strokeId = stroke.id;
      path.addEventListener('click', event => { event.stopPropagation(); selectStroke(stroke); });
      svg.append(path);
    }
    byId('ink-empty').hidden = strokes.size > 0;
    byId('ink-canvas').hidden = false;
    byId('ink-empty-title').textContent = strokes.size ? '' : '此序号暂无已确认墨迹';
    byId('ink-empty-description').textContent = strokes.size ? '' : '拖动时间线或选择其他会话。';
    slider.value = String(seq);
    sliderOutput.textContent = `seq ${seq} / ${payload.through_seq ?? seq}`;
    status.textContent = `seq ${seq} · ${strokes.size} 条墨迹`;
    exportSvg.disabled = strokes.size === 0;
    exportJson.disabled = !payload;
  }

  function selectStroke(stroke) {
    selectedId = stroke.id;
    for (const path of svg.querySelectorAll('path')) path.classList.toggle('selected', path.dataset.strokeId === selectedId);
    detail.innerHTML = '';
    const firstSeq = Number(stroke.first_seq);
    const firstEvent = (payload?.events || []).find(event => Number(event.seq) === firstSeq);
    const direction = stroke.direction === 'forward' ? '正向' : stroke.direction === 'reverse' ? '反向' : '方向不确定';
    const directionSource = stroke.direction_source === 'inferred' ? '推断' : '未知';
    const lines = [
      ['ID', stroke.id], ['颜色', safeColor(stroke.color)], ['宽度', String(stroke.width ?? '—')],
      ['首次出现', `seq ${stroke.first_seq ?? '—'}`], ['确认时间', timeLabel(firstEvent?.confirmed_at)],
      ['点数', String(stroke.points?.length ?? 0)], ['方向', `${direction}（依据：${directionSource}）`],
    ];
    if (stroke.identity_ambiguous === true || payload?.events?.some(event => Number(event.seq) === firstSeq && event.identity_ambiguous === true)) {
      lines.push(['身份', '不确定：存在多个合理前身']);
    }
    for (const [label, value] of lines) {
      const row = document.createElement('div'); const term = document.createElement('dt'); const desc = document.createElement('dd');
      term.textContent = label; desc.textContent = value; row.append(term, desc); detail.append(row);
    }
  }

  function renderSessions() {
    sessionList.replaceChildren();
    if (!sessions.length) { const empty = document.createElement('p'); empty.className = 'list-message'; empty.textContent = '尚无可用会话'; sessionList.append(empty); return; }
    for (const session of sessions) {
      const button = document.createElement('button'); button.className = 'session-row'; button.textContent = `${new Date(session.created_at).toLocaleString('zh-CN')} · seq ${session.seq}`;
      button.setAttribute('aria-pressed', String(session.id === selected));
      button.addEventListener('click', () => void selectSession(session.id)); sessionList.append(button);
    }
  }

  async function loadInk(id, version) {
    show('正在读取电子墨迹…'); status.textContent = '正在加载';
    loadController = new AbortController();
    const signal = loadController.signal;
    for (let attempt = 0; attempt < 120; attempt++) {
      if (version !== loadVersion || signal.aborted) throw new Error('电子墨迹读取已取消');
      const timer = setTimeout(() => loadController?.abort(), 20000);
      let result;
      try {
        const response = await fetch(sessionURL(id) + '/ink', { cache: 'no-store', signal });
        result = await response.json().catch(() => null);
        if (!response.ok && response.status !== 202) throw new Error(result?.error || '服务返回 HTTP ' + response.status);
        if (!result) throw new Error('服务返回了无效的 JSON');
      } catch (error) {
        if (signal.aborted) throw new Error('电子墨迹读取已取消');
        if (error instanceof TypeError) throw new Error('无法连接本机服务，请检查服务是否正在运行');
        throw error;
      } finally { clearTimeout(timer); }
      if (result?.status === 202 || result?.status === 'pending' || result?.ready === false) { await new Promise(resolve => setTimeout(resolve, 500)); continue; }
      if (result?.version !== 1 || !Array.isArray(result.events)) throw new Error('服务返回的电子墨迹格式不受支持');
      return result;
    }
    throw new Error('电子墨迹仍在生成，请稍后重试');
  }

  async function selectSession(id) {
    loadVersion++;
    loadController?.abort();
    const version = loadVersion;
    selected = id; payload = null; seq = 0; selectedId = null; renderSessions(); clear('正在读取电子墨迹', '首次加载可能需要几秒钟。');
    try {
      payload = await loadInk(id, version);
      if (version !== loadVersion) return;
      seq = Number(payload.through_seq) || 0;
      slider.max = String(seq); slider.disabled = seq === 0; slider.value = String(seq);
      render(); show('');
    } catch (error) {
      if (version !== loadVersion) return;
      clear(error.message === '电子墨迹读取已取消' ? '读取已取消' : '无法读取电子墨迹', '可点击刷新会话后重试。'); show(error.message);
    } finally { if (version === loadVersion) loadController = null; }
  }

  function currentStrokes() {
    const strokes = new Map();
    for (const event of eventAt(payload?.events || [], seq)) for (const change of event.changes || []) {
      if (change.op === 'remove') strokes.delete(change.id); else if (change.stroke) strokes.set(change.id, { ...change.stroke, id: change.id, identity_ambiguous: change.identity_ambiguous ?? change.stroke.identity_ambiguous });
    }
    return strokes;
  }

  slider.addEventListener('input', () => { seq = Number(slider.value); render(); });
  exportJson.addEventListener('click', () => {
    if (!payload) return;
    const exportPayload = { ...payload, events: payload.events, view_seq: seq };
    download(`ink-${selected}-seq-${seq}.json`, 'application/json', JSON.stringify(exportPayload, null, 2));
  });
  exportSvg.addEventListener('click', () => {
    if (!payload) return;
    const copy = svg.cloneNode(true); copy.querySelectorAll('.selected').forEach(node => node.classList.remove('selected'));
    copy.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    download(`ink-${selected}-seq-${seq}.svg`, 'image/svg+xml', new XMLSerializer().serializeToString(copy));
  });
  byId('ink-refresh').addEventListener('click', () => void open());

  async function open() {
    show('正在读取会话目录…');
    try { const result = await request('/api/sessions'); sessions = (result.sessions || []).slice().sort((a, b) => b.created_at - a.created_at).slice(0, MAX_SESSIONS); renderSessions(); if (!sessions.length) clear(); else if (!selected || !sessions.some(s => s.id === selected)) await selectSession(sessions[0].id); show(''); }
    catch (error) { renderSessions(); clear('无法读取会话目录', '请确认本机服务正在运行。'); show(error.message); }
  }
  function close() { loadVersion++; loadController?.abort(); loadController = null; show(''); }
  clear();
  return { open, close };
}
