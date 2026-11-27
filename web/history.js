import { byId, notice, request, sessionURL } from './common.js';
import { ReplayBoard, decodeSegment, disposeSegment } from './replay-board.js';

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

export function setupHistory() {
  const replay = new ReplayBoard(byId('history-board'));
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
  const segmentStart = seq => seq - seq % 32;

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

  function release(start, entry) {
    segments.delete(start);
    entry.controller.abort();
    if (entry.segment) disposeSegment(entry.segment);
  }

  function resetDetail(title, description) {
    ++sessionVersion;
    ++stateVersion;
    clearTimeout(seekTimer);
    events = [];
    position = 0;
    pause();
    replay.clear(title, description);
    for (const [start, entry] of segments) release(start, entry);
    eventDetails('等待选择会话');
  }

  function seek(index, debounce = false) {
    clearTimeout(seekTimer);
    clearTimeout(playTimer);
    const event = events[index];
    if (!event || !selected) return;
    position = index;
    const version = ++stateVersion;
    const targetStart = segmentStart(event.seq);
    const nextStart = targetStart + 32;
    for (const [start, entry] of segments) {
      if (!entry.segment && start !== targetStart && start !== nextStart) release(start, entry);
    }
    transport();
    alert('');
    async function load() {
      try {
        const segment = await loadSegment(event.seq);
        if (version !== stateVersion) return;
        await replay.render(segment, event.seq, () => version === stateVersion);
        if (version !== stateVersion) return;
        eventDetails(event.seq === 0 ? '初始状态 · 全板未知' : '序号 ' + event.seq + ' · ' + (eventNames[event.kind] || event.kind || '板书变化'), event);
        prefetch(segment.end + 1, version);
        if (playing && position < events.length - 1) playTimer = setTimeout(() => seek(position + 1), 1000);
        else if (playing) pause();
      } catch (error) {
        if (version !== stateVersion) return;
        pause();
        if (!replay.hasFrame()) {
          replay.clear('无法读取此状态', '可选择其他事件，或刷新记录重试。');
          eventDetails('读取失败');
        }
        alert(error.message);
      }
    }
    if (debounce) seekTimer = setTimeout(load, 80);
    else void load();
  }

  function loadSegment(seq) {
    const start = segmentStart(seq);
    if (segments.has(start)) {
      const cached = segments.get(start);
      segments.delete(start);
      segments.set(start, cached);
      return cached.promise;
    }
    const controller = new AbortController();
    const entry = { controller, segment: null };
    segments.set(start, entry);
    if (segments.size > maxSegments) release(...segments.entries().next().value);
    const timeout = setTimeout(() => controller.abort(new Error('读取回放分段超时')), 20000);
    entry.promise = (async () => {
      try {
        const response = await fetch(sessionURL(selected) + '/replay?seq=' + seq,
          { cache: 'no-store', signal: controller.signal });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload?.error || '服务返回 HTTP ' + response.status);
        if (payload.start !== start) throw new Error('服务返回的分段起点与请求不一致');
        const segment = await decodeSegment(payload, controller.signal);
        if (segments.get(start) !== entry) {
          disposeSegment(segment);
          throw new Error('回放请求已过期');
        }
        entry.segment = segment;
        return segment;
      } catch (error) {
        if (segments.get(start) === entry) segments.delete(start);
        throw error;
      } finally {
        clearTimeout(timeout);
      }
    })();
    return entry.promise;
  }

  function prefetch(seq, version) {
    if (version !== stateVersion || !selected || seq > events[events.length - 1]?.seq) return;
    void loadSegment(seq).catch(() => {});
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
      replay.clear('无法读取会话', '点击刷新记录后重试。');
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
        replay.clear('尚无会话可回放', '在实时采集工作台完成一次采集后，即可在这里查看。');
      }
    } catch (error) {
      if (version !== listVersion) return;
      listMessage('会话目录读取失败');
      replay.clear('无法读取会话目录', '请检查本机服务后刷新记录。');
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
      for (const [start, entry] of segments) release(start, entry);
    },
  };
}
