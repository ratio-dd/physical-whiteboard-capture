import { byId, notice, request, sessionURL } from './common.js';
import { PlaybackClock, eventAt, eventOffsets } from './playback-clock.js';
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
  let offsets = [];
  let clock = new PlaybackClock(0);
  let displayed = -1;
  let pending = false;
  let playTimer = null;
  const speed = byId('playback-speed');
  let seekTimer = null;
  const segments = new Map();
  const maxSegments = 3;
  const segmentStart = seq => seq - seq % 32;

  function durationLabel(ms) {
    const tenths = Math.floor(ms / 100);
    const minutes = Math.floor(tenths / 600);
    const seconds = Math.floor(tenths / 10) % 60;
    return String(minutes).padStart(2, '0') + ':' + String(seconds).padStart(2, '0') + '.' + tenths % 10;
  }

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

  function progress(ms) {
    slider.value = Math.floor(ms);
    const label = durationLabel(ms);
    byId('replay-elapsed').textContent = label;
    slider.setAttribute('aria-valuetext', label);
  }

  function transport() {
    slider.max = clock.duration;
    slider.disabled = clock.duration === 0;
    progress(clock.read());
    byId('prev').disabled = position <= 0;
    byId('next').disabled = position >= events.length - 1;
    byId('playback').disabled = events.length < 2;
    byId('playback').textContent = clock.playing ? '暂停' : '播放';
    byId('event-counter').textContent = displayed >= 0 ? displayed + ' / ' + (events.length - 1) : '—';
    byId('replay-total').textContent = durationLabel(clock.duration);
  }

  function stopClock() {
    clearTimeout(playTimer);
    clock.pause();
  }

  function tick() {
    clearTimeout(playTimer);
    if (!clock.playing) return;
    const ms = clock.read();
    const due = eventAt(offsets, ms);
    progress(ms);
    if (!pending && due !== displayed) show(due, { timed: true });
    if (ms >= clock.duration) {
      // The final frame may still be decoding. Its completion calls tick again.
      if (!pending && displayed === events.length - 1) {
        stopClock();
        transport();
      }
      return;
    }
    // Update the playhead during idle gaps and wake up exactly when the next
    // event is due. Rendering never changes the playback clock's anchors.
    const wait = Math.min(100, clock.until(offsets[due + 1]));
    playTimer = setTimeout(tick, Math.max(1, Math.ceil(wait)));
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
    stopClock();
    pending = false;
    events = [];
    offsets = [];
    position = 0;
    displayed = -1;
    clock = new PlaybackClock(0);
    transport();
    replay.clear(title, description);
    for (const [start, entry] of segments) release(start, entry);
    eventDetails('等待选择会话');
  }

  function show(index, { debounce = false, timed = false } = {}) {
    if (!events[index]) return;
    clearTimeout(seekTimer);
    const version = ++stateVersion;
    const current = () => version === stateVersion;
    position = index;
    pending = true;
    const targetStart = segmentStart(events[index].seq);
    for (const [start, entry] of segments) {
      if (!entry.segment && start !== targetStart && start !== targetStart + 32) release(start, entry);
    }
    transport();
    alert('');
    async function load() {
      try {
        const segment = await loadSegment(events[index].seq);
        if (!current()) return;
        if (timed && clock.playing) {
          // A late network response may span several events or a whole segment.
          index = eventAt(offsets, clock.read());
          if (events[index].seq > segment.end) return;
          position = index;
        }
        const event = events[index];
        await replay.render(segment, event.seq, current);
        if (!current()) return;
        displayed = index;
        eventDetails(event.seq === 0 ? '初始状态 · 全板未知' :
          '序号 ' + event.seq + ' · ' + (eventNames[event.kind] || event.kind || '板书变化'), event);
        prefetch(segment.end + 1, version);
      } catch (error) {
        if (!current()) return;
        stopClock();
        if (!replay.hasFrame()) {
          replay.clear('无法读取此状态', '可选择其他事件，或刷新记录重试。');
          eventDetails('读取失败');
        }
        alert(error.message);
      } finally {
        if (current()) {
          pending = false;
          transport();
          if (clock.playing) tick();
        }
      }
    }
    if (debounce) seekTimer = setTimeout(load, 80);
    else void load();
  }

  function step(index) {
    if (!events[index]) return;
    stopClock();
    clock.seek(offsets[index]);
    show(index);
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

  async function select(id, preferredTime = null) {
    selected = id;
    resetDetail('正在读取会话');
    const version = sessionVersion;
    rows();
    alert('');
    try {
      const result = await request(sessionURL(id) + '/events');
      if (version !== sessionVersion) return;
      events = [{ seq: 0 }, ...result.events.filter(event => event.seq > 0).sort((a, b) => a.seq - b.seq)];
      offsets = eventOffsets(events, sessions.find(item => item.id === id)?.created_at);
      clock = new PlaybackClock(offsets.at(-1));
      clock.setRate(Number(speed.value));
      clock.seek(preferredTime ?? 0);
      show(preferredTime === null ? 0 : eventAt(offsets, clock.read()));
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
    const previousTime = clock.read();
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
        await select(found ? previous : sessions[0].id, found ? previousTime : null);
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
    const target = Number(slider.value);
    stopClock();
    clock.seek(target);
    show(eventAt(offsets, target), { debounce: true });
  });
  byId('prev').addEventListener('click', () => step(position - 1));
  byId('next').addEventListener('click', () => step(position + 1));
  byId('playback').addEventListener('click', () => {
    if (clock.playing) {
      stopClock();
      const due = eventAt(offsets, clock.read());
      if (pending || displayed !== due) show(due);
      else transport();
      return;
    }
    if (!events.length) return;
    if (displayed === events.length - 1 && clock.read() >= clock.duration) clock.seek(0);
    clock.play();
    transport();
    tick();
  });
  speed.addEventListener('change', () => {
    clock.setRate(Number(speed.value));
    transport();
    if (clock.playing) tick();
  });
  return {
    open: refresh,
    close() {
      ++listVersion;
      ++sessionVersion;
      ++stateVersion;
      clearTimeout(seekTimer);
      pending = false;
      stopClock();
      for (const [start, entry] of segments) release(start, entry);
    },
  };
}
