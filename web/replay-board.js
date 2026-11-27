import { byId } from './common.js';

// Keep compressed PNGs for backward seeks, but decode only a small window of patches.
const DECODED_FRAME_LIMIT = 6;
const DECODED_BYTE_LIMIT = 24 * 1024 * 1024;

async function bitmap(encoded) {
  const bytes = Uint8Array.fromBase64
    ? Uint8Array.fromBase64(encoded)
    : Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
  return createImageBitmap(new Blob([bytes], { type: 'image/png' }));
}

function closePatches(patches) {
  for (const patch of patches) {
    patch.board.close();
    patch.known.close();
  }
}

async function decodePatches(patches, signal) {
  const decoded = [];
  try {
    for (const patch of patches) {
      signal.throwIfAborted();
      const board = await bitmap(patch.board);
      let known;
      try {
        known = await bitmap(patch.known);
      } catch (error) {
        board.close();
        throw error;
      }
      decoded.push({ x: patch.x ?? 0, y: patch.y ?? 0, board, known });
      if (board.width !== known.width || board.height !== known.height) {
        throw new Error('板书和掩码的尺寸不一致');
      }
    }
    signal.throwIfAborted();
    return decoded;
  } catch (error) {
    closePatches(decoded);
    throw error;
  }
}

class Segment {
  constructor(payload, snapshot) {
    this.start = payload.start;
    this.end = payload.end;
    this.snapshot = snapshot;
    this.frames = payload.frames;
    this.cache = new Map();
    this.abort = new AbortController();
    this.bytes = 0;
  }

  async frame(seq) {
    const entries = this.cache;
    if (entries.has(seq)) {
      const hit = entries.get(seq);
      entries.delete(seq);
      entries.set(seq, hit);
      return hit.task;
    }
    const entry = {};
    entry.task = decodePatches(this.frames[seq - this.start - 1].patches, this.abort.signal)
      .then(patches => {
        if (this.abort.signal.aborted) {
          closePatches(patches);
          this.abort.signal.throwIfAborted();
        }
        entry.patches = patches;
        entry.bytes = patches.reduce((sum, p) => sum + 8 * p.board.width * p.board.height, 0);
        this.bytes += entry.bytes;
        for (const [oldSeq, old] of entries) {
          if (entries.size <= DECODED_FRAME_LIMIT && this.bytes <= DECODED_BYTE_LIMIT) break;
          if (oldSeq === seq || !old.patches) continue;
          closePatches(old.patches);
          entries.delete(oldSeq);
          this.bytes -= old.bytes;
        }
        return patches;
      }).catch(error => {
        entries.delete(seq);
        throw error;
      });
    entries.set(seq, entry);
    return entry.task;
  }

  warm(seq) {
    if (seq <= this.end) void this.frame(seq).catch(() => {});
  }
}

export async function decodeSegment(payload, signal) {
  const { start, end, snapshot, frames } = payload;
  if (!Number.isInteger(start) || !Number.isInteger(end) || start % 32 !== 0 ||
      end < start || end >= start + 32 || !snapshot || !Array.isArray(frames) ||
      frames.length !== end - start || frames.some((f, i) => f.seq !== start + i + 1 || !Array.isArray(f.patches))) {
    throw new Error('服务返回了无效的回放分段');
  }
  const [base] = await decodePatches([snapshot], signal);
  const segment = new Segment(payload, base);
  segment.warm(start + 1);
  return segment;
}

export function disposeSegment(segment) {
  segment.abort.abort();
  closePatches([segment.snapshot]);
  for (const entry of segment.cache.values()) if (entry.patches) closePatches(entry.patches);
  segment.cache.clear();
  segment.frames = [];
}

function surface() {
  const board = document.createElement('canvas').getContext('2d', { alpha: false });
  const unknown = document.createElement('canvas').getContext('2d', { alpha: false, willReadFrequently: true });
  return { board, unknown };
}

function resize(surface, width, height) {
  for (const { canvas } of [surface.board, surface.unknown]) {
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
  }
}

function draw(surface, patch) {
  const { board, known, x, y } = patch;
  surface.board.drawImage(board, x, y);
  surface.unknown.drawImage(known, x, y);
  // Stored masks use white for known. The layer shown to the user uses white for unknown.
  surface.unknown.globalCompositeOperation = 'difference';
  surface.unknown.fillStyle = 'white';
  surface.unknown.fillRect(x, y, known.width, known.height);
  surface.unknown.globalCompositeOperation = 'source-over';
}

export class ReplayBoard {
  constructor(root) {
    root.append(byId('board-template').content.cloneNode(true));
    this.find = selector => root.querySelector(selector);
    this.front = surface();
    this.buttons = [...root.querySelectorAll('[data-layer]')];
    this.layer = 'board';
    this.find('.board-title').textContent = '会话回放';
    this.find('.held').hidden = true;
    const images = this.find('.board-images');
    for (const [name, layer] of Object.entries(this.front)) {
      layer.canvas.className = 'board-image';
      layer.canvas.dataset.image = name;
      layer.canvas.setAttribute('role', 'img');
      layer.canvas.setAttribute('aria-label', name === 'board' ? '透视校正后的板书' : '未知区域掩码');
    }
    this.overlay = document.createElement('canvas').getContext('2d');
    this.overlay.canvas.className = 'unknown-overlay';
    this.overlay.canvas.style.cssText = 'width:100%;height:100%;background:none;pointer-events:none';
    this.overlay.canvas.setAttribute('aria-hidden', 'true');
    images.replaceChildren(this.front.board.canvas, this.front.unknown.canvas, this.overlay.canvas);
    for (const button of this.buttons) button.addEventListener('click', () => {
      this.layer = button.dataset.layer;
      this.paint();
    });
    this.clear('选择会话以查看板书');
  }

  paint() {
    const mask = this.layer === 'unknown';
    for (const b of this.buttons) b.setAttribute('aria-pressed', String(b.dataset.layer === this.layer));
    this.front.board.canvas.hidden = mask;
    this.front.unknown.canvas.hidden = !mask;
    this.overlay.canvas.hidden = mask;
    this.find('.legend').classList.toggle('mask', mask);
    this.find('.legend span').textContent = mask ? '白色为未知，黑色为已知' : '尚未确认的区域';
    this.find('.board-images').hidden = !this.hasFrame();
    this.find('.board-empty').hidden = this.hasFrame();
  }

  clear(title, description = '') {
    this.seq = null;
    this.segment = null;
    this.find('.board-empty strong').textContent = title;
    this.find('.board-empty p').textContent = description;
    this.find('.seq').textContent = '序号 —';
    this.paint();
  }

  hasFrame() {
    return this.seq !== null;
  }

  // Convert the changed mask area into the same translucent hatch as the live board.
  hatch(x, y, width, height) {
    const pixels = this.front.unknown.getImageData(x, y, width, height);
    const data = pixels.data;
    for (let row = 0, i = 0; row < height; row++) {
      for (let col = 0; col < width; col++, i += 4) {
        if (!data[i]) { data[i + 3] = 0; continue; }
        const stripe = (x + col + y + row) % 8 < 2;
        data[i] = stripe ? 121 : 237;
        data[i + 1] = stripe ? 135 : 240;
        data[i + 2] = stripe ? 119 : 230;
        data[i + 3] = stripe ? 140 : 166;
      }
    }
    this.overlay.putImageData(pixels, x, y);
  }

  async render(segment, seq, current) {
    if (seq < segment.start || seq > segment.end) throw new Error('回放分段未包含所选序号');
    if (!current()) return;
    if (this.segment === segment && this.seq === seq) return;
    const { width, height } = segment.snapshot.board;
    if (this.segment === segment && seq === this.seq + 1) {
      // Consecutive playback touches only this event's rectangles.
      const patches = await segment.frame(seq);
      if (!current()) return;
      for (const patch of patches) draw(this.front, patch);
      for (const p of patches) this.hatch(p.x, p.y, p.board.width, p.board.height);
    } else {
      // Keep both displayed layers intact while a backwards or distant seek decodes.
      const work = surface();
      try {
        resize(work, width, height);
        draw(work, segment.snapshot);
        for (let i = segment.start + 1; i <= seq; i++) {
          const patches = await segment.frame(i);
          if (!current()) return;
          for (const patch of patches) draw(work, patch);
        }
        resize(this.front, width, height);
        this.front.board.drawImage(work.board.canvas, 0, 0);
        this.front.unknown.drawImage(work.unknown.canvas, 0, 0);
        this.overlay.canvas.width = width;
        this.overlay.canvas.height = height;
        this.hatch(0, 0, width, height);
      } finally {
        resize(work, 0, 0);
      }
    }
    this.seq = seq;
    this.segment = segment;
    this.find('.seq').textContent = '序号 ' + seq;
    this.paint();
    segment.warm(seq + 1);
  }
}
