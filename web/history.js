import { Board, byId, notice, request, sessionURL } from './common.js';

const eventNames = { first_visible: '首次确认可见区域', add: '新增板书', erase: '擦除板书', update: '板书更新' };
const dateTime = new Intl.DateTimeFormat('zh-CN', {
  year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
});

function timeLabel(ms) {
  return Number.isFinite(ms) ? dateTime.format(ms) : '—';
}

function eventTime(ms) {
  return Number.isFinite(ms) ? timeLabel(ms) + '.' + String(new Date(ms).getMilliseconds()).padStart(3, '0') : '—';
}

function bitmap(value) {
  return fetch('data:image/png;base64,' + value).then(response => response.blob()).then(createImageBitmap);
}

class ReplayBoard {
  constructor(board) {
    this.board = board;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'board-image';
    this.mask = document.createElement('canvas');
    this.mask.className = 'board-image';
    this.ctx = this.canvas.getContext('2d', { alpha: false });
    this.maskCtx = this.mask.getContext('2d', { alpha: false });
    this.mask.style.filter = 'invert(1)';
    this.current = null;
    board.paint = () => this.paint();
  }

  reset() {
    this.current = null;
  }

  hasFrame(seq) {
    return this.current?.seq === seq;
  }

  async decode(result) {
    if (result.start % 32 || result.end < result.start || !result.snapshot) throw new Error('服务返回了无效的回放分段');
    const snapshot = { board: await bitmap(result.snapshot.board), known: await bitmap(result.snapshot.known) };
    const frames = [];
    for (const frame of result.frames || []) {
      const patches = [];
      for (const patch of frame.patches || []) {
        patches.push({ x: patch.x, y: patch.y, board: await bitmap(patch.board), known: await bitmap(patch.known) });
      }
      frames.push({ seq: frame.seq, patches });
    }
    return { start: result.start, end: result.end, snapshot, frames };
  }

  async render(segment, seq) {
    const width = segment.snapshot.board.width;
    const height = segment.snapshot.board.height;
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width; this.canvas.height = height;
      this.mask.width = width; this.mask.height = height;
    }
    this.ctx.clearRect(0, 0, width, height);
    this.maskCtx.clearRect(0, 0, width, height);
    this.ctx.drawImage(segment.snapshot.board, 0, 0);
    this.maskCtx.drawImage(segment.snapshot.known, 0, 0);
    for (const frame of segment.frames) {
      if (frame.seq > seq) break;
      for (const patch of frame.patches) {
        this.ctx.drawImage(patch.board, patch.x, patch.y);
        this.maskCtx.drawImage(patch.known, patch.x, patch.y);
      }
    }
    this.current = { seq, segment };
    this.board.find('.seq').textContent = '序号 ' + seq;
    this.paint();
  }

  paint() {
    const root = this.board.find('.board-images');
    const empty = this.board.find('.board-empty');
    const mask = this.board.layer === 'unknown';
    for (const button of this.board.buttons) button.setAttribute('aria-pressed', String(button.dataset.layer === this.board.layer));
    this.board.find('.legend').classList.toggle('mask', mask);
    this.board.find('.legend span').textContent = mask ? '白色为未知，黑色为已知' : '尚未确认的区域';
    if (!this.current) {
      root.hidden = true; empty.hidden = false;
      return;
    }
    root.hidden = false; empty.hidden = true;
    const image = mask ? this.mask : this.canvas;
    const current = root.querySelector('.board-image');
    if (current !== image) current.replaceWith(image);
    this.board.find('.unknown-overlay').hidden = true;
  }
}

export function setupHistory() {
  const board = new Board(byId('history-board'), true);
  const replay = new ReplayBoard(board);
  const list = byId('session-list');
  const slider = byId('scrubber');
  const alert = message => notice(byId('history-notice'), message);
  let sessions = [];
  let selected = null;
  let events = [];
  let position = 0;
  let listVersion = 0;
  let sessionVersion = 0;
  let stateVersion = 0;
  let playing = false;
  let playTimer = null;
  let seekTimer = null;
  const segments = new Map();
  const maxSegments = 3;

  function listMessage(message) {
    const p = document.createElement('p');
    p.className = 'list-message';
    p.textContent = message;
    list.replaceChildren(p);
  }

  function rows() {
    if (!sessions.length) {
      listMessage('尚无采集记录');
      return;
    }
    const nodes = sessions.map(session => {
      const button = document.createElement('button');
      button.className = 'session-row';
      if (session.id === selected) button.setAttribute('aria-current', 'true');
      const info = document.createElement('span');
      const title = document.createElement('strong');
      title.textContent = timeLabel(session.created_at);
      const detail = document.createElement('small');
      detail.textContent = session.id.slice(0, 8) + ' · ' + session.seq + ' 次变化';
      button.title = session.id;
      info.append(title, detail);
      const arrow = document.createElement('span');
      arrow.textContent = '›';
      arrow.setAttribute('aria-hidden', 'true');
      button.append(info, arrow);
      button.addEventListener('click', () => { if (selected !== session.id) void select(session.id); });
      return button;
    });
    list.replaceChildren(...nodes);
  }

  function transport() {
    slider.max = Math.max(0, events.length - 1);
    slider.value = position;
    slider.disabled = events.length < 2;
    slider.setAttribute('aria-valuetext', events[position] ? '序号 ' + events[position].seq : '无事件');
    byId('prev').disabled = position <= 0;
    byId('next').disabled = position >= events.length - 1;
    byId('playback').disabled = events.length < 2;
    byId('playback').textContent = playing ? '暂停' : '播放';
    byId('event-counter').textContent = events.length ? position + ' / ' + (events.length - 1) : '—';
  }

  function pause() {
    clearTimeout(playTimer);
    playing = false;
    transport();
  }

  function eventDetails(description, event) {
    byId('event-summary').textContent = description;
    byId('observed').textContent = eventTime(event?.observed_at);
    byId('confirmed').textContent = eventTime(event?.confirmed_at);
  }

  function resetDetail(title, description) {
    ++sessionVersion;
    ++stateVersion;
    clearTimeout(seekTimer);
    events = [];
    position = 0;
    pause();
    board.clear(title, description);
    replay.reset();
    segments.clear();
    eventDetails('等待选择会话');
  }

  function seek(index, debounce = false) {
    clearTimeout(seekTimer);
    clearTimeout(playTimer);
    const event = events[index];
    if (!event || !selected) return;
    position = index;
    const version = ++stateVersion;
    transport();
    if (!replay.hasFrame(event.seq)) board.loading('正在读取序号 ' + event.seq);
    eventDetails('正在读取序号 ' + event.seq + '…');
    alert('');
    async function load() {
      try {
        const segment = await loadSegment(event.seq);
        if (version !== stateVersion) return;
        await replay.render(segment, event.seq);
        if (version !== stateVersion) return;
        eventDetails(event.seq === 0 ? '初始状态 · 全板未知' : '序号 ' + event.seq + ' · ' + (eventNames[event.kind] || event.kind || '板书变化'), event);
        if (playing && position < events.length - 1) playTimer = setTimeout(() => seek(position + 1), 1000);
        else if (playing) pause();
      } catch (error) {
        if (version !== stateVersion) return;
        pause();
        board.clear('无法读取此状态', '可选择其他事件，或刷新记录重试。');
        eventDetails('读取失败');
        alert(error.message);
      }
    }
    if (debounce) seekTimer = setTimeout(load, 80);
    else void load();
  }

  async function loadSegment(seq) {
    const start = Math.floor(seq / 32) * 32;
    if (segments.has(start)) return segments.get(start);
    const promise = request(sessionURL(selected) + '/replay?seq=' + seq)
      .then(result => replay.decode(result));
    segments.set(start, promise);
    try {
      const segment = await promise;
      while (segments.size > maxSegments) segments.delete(segments.keys().next().value);
      const next = segment.end + 1;
      if (next <= events[events.length - 1]?.seq) void loadSegment(next).catch(() => {});
      return segment;
    } catch (error) {
      segments.delete(start);
      throw error;
    }
  }

  async function select(id, preferred = 0) {
    selected = id;
    resetDetail('正在读取会话');
    const version = sessionVersion;
    rows();
    alert('');
    try {
      const result = await request(sessionURL(id) + '/events');
      if (version !== sessionVersion) return;
      events = [{ seq: 0 }, ...result.events.filter(event => event.seq > 0).sort((a, b) => a.seq - b.seq)];
      seek(Math.min(preferred, events.length - 1));
    } catch (error) {
      if (version !== sessionVersion) return;
      board.clear('无法读取会话', '点击刷新记录后重试。');
      eventDetails('读取失败');
      alert(error.message);
    }
  }

  async function refresh() {
    const version = ++listVersion;
    const previous = selected;
    const previousPosition = position;
    selected = null;
    resetDetail('正在读取会话目录');
    listMessage('正在读取记录…');
    byId('session-count').textContent = '—';
    alert('');
    try {
      const result = await request('/api/sessions');
      if (version !== listVersion) return;
      sessions = result.sessions.slice().sort((a, b) => b.created_at - a.created_at);
      byId('session-count').textContent = sessions.length;
      if (sessions.length) {
        const found = sessions.some(session => session.id === previous);
        await select(found ? previous : sessions[0].id, found ? previousPosition : 0);
      } else {
        rows();
        board.clear('尚无会话可回放', '在实时采集工作台完成一次采集后，即可在这里查看。');
      }
    } catch (error) {
      if (version !== listVersion) return;
      listMessage('会话目录读取失败');
      board.clear('无法读取会话目录', '请检查本机服务后刷新记录。');
      alert(error.message);
    }
  }

  byId('refresh-history').addEventListener('click', refresh);
  slider.addEventListener('input', () => {
    const index = Number(slider.value);
    pause();
    seek(index, true);
  });
  byId('prev').addEventListener('click', () => { pause(); seek(position - 1); });
  byId('next').addEventListener('click', () => { pause(); seek(position + 1); });
  byId('playback').addEventListener('click', () => {
    if (playing) { pause(); return; }
    playing = true;
    seek(position >= events.length - 1 ? 0 : position + 1);
  });
  return {
    open: refresh,
    close() {
      ++listVersion;
      ++sessionVersion;
      ++stateVersion;
      clearTimeout(seekTimer);
      pause();
    },
  };
}
