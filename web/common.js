export const byId = id => document.getElementById(id);
export const sessionURL = id => '/api/sessions/' + encodeURIComponent(id);

export async function request(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(path, { cache: 'no-store', ...options, signal: controller.signal });
    const result = await response.json().catch(() => null);
    if (response.status === 202 && result) return { ...result, status: 202 };
    if (!response.ok) throw new Error(result?.error || '服务返回 HTTP ' + response.status);
    if (!result) throw new Error('服务返回了无效的 JSON');
    return result;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('本机服务响应超时');
    if (error instanceof TypeError) throw new Error('无法连接本机服务，请检查服务是否正在运行');
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

export function notice(node, message = '') {
  node.textContent = message;
  node.hidden = !message;
}

function png(data) {
  if (typeof data !== 'string' || !data) throw new Error('服务返回的 PNG 数据缺失');
  const image = new Image();
  image.src = 'data:image/png;base64,' + data;
  return image;
}

export class Board {
  constructor(root, history = false) {
    root.append(byId('board-template').content.cloneNode(true));
    this.find = selector => root.querySelector(selector);
    this.buttons = [...root.querySelectorAll('[data-layer]')];
    this.layer = 'board';
    this.revision = 0;
    for (const button of this.buttons) {
      button.addEventListener('click', () => {
        this.layer = button.dataset.layer;
        this.paint();
      });
    }
    if (history) {
      this.find('.board-title').textContent = '会话回放';
      this.find('.held').hidden = true;
    }
    this.clear(history ? '选择会话以查看板书' : '等待第一帧', history ? '从会话目录中选择一条记录。' : '完成校准并开始采集后，数字白板将在这里更新。');
  }

  clear(title, description = '') {
    this.revision++;
    this.images = null;
    this.find('.board-empty strong').textContent = title;
    this.find('.board-empty p').textContent = description;
    this.find('.seq').textContent = '序号 —';
    this.find('.held b').textContent = '—';
    this.paint();
  }

  loading(title) {
    this.revision++;
    if (!this.images) {
      this.find('.board-empty strong').textContent = title;
      this.find('.board-empty p').textContent = '';
    }
  }

  setHeld(fraction) {
    this.find('.held b').textContent = Number.isFinite(fraction) ? (fraction * 100).toFixed(1) + '%' : '—';
  }

  async update(state) {
    const revision = ++this.revision;
    const images = { board: png(state.board), unknown: png(state.unknown) };
    try {
      await Promise.all([images.board.decode(), images.unknown.decode()]);
    } catch {
      throw new Error('无法读取服务返回的白板或未知区域图像');
    }
    if (revision !== this.revision) return;
    if (images.board.naturalWidth !== images.unknown.naturalWidth || images.board.naturalHeight !== images.unknown.naturalHeight) {
      throw new Error('白板和未知区域图像的尺寸不一致');
    }
    this.images = images;
    this.find('.seq').textContent = '序号 ' + state.seq;
    this.setHeld(state.held);
    this.paint();
  }

  paint() {
    const mask = this.layer === 'unknown';
    for (const button of this.buttons) button.setAttribute('aria-pressed', String(button.dataset.layer === this.layer));
    this.find('.legend').classList.toggle('mask', mask);
    this.find('.legend span').textContent = mask ? '白色为未知，黑色为已知' : '尚未确认的区域';
    this.find('.board-images').hidden = !this.images;
    this.find('.board-empty').hidden = Boolean(this.images);
    const overlay = this.find('.unknown-overlay');
    overlay.hidden = mask || !this.images;
    if (!this.images) {
      this.find('.board-image').removeAttribute('src');
      overlay.style.maskImage = '';
      return;
    }
    const image = this.images[this.layer];
    image.className = 'board-image';
    image.alt = mask ? '未知区域掩膜，白色未知、黑色已知' : '透视校正后的板书';
    const current = this.find('.board-image');
    if (current !== image) current.replaceWith(image);
    overlay.style.maskImage = mask ? '' : 'url("' + this.images.unknown.src + '")';
  }
}
