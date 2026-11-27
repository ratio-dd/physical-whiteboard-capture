// Playback time is measured from session creation to the last confirmed state.
export function eventOffsets(events, createdAt) {
  const origin = Number.isFinite(createdAt) ? createdAt : (events[1]?.confirmed_at ?? 0);
  let previous = 0;
  return events.map((event, i) => {
    if (i && Number.isFinite(event.confirmed_at)) {
      previous = Math.max(previous, event.confirmed_at - origin);
    }
    return previous;
  });
}

// The most recent state due at this time; equal timestamps resolve in sequence order.
export function eventAt(offsets, ms) {
  let low = 0, high = offsets.length;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (offsets[middle] <= ms) low = middle + 1;
    else high = middle;
  }
  return low - 1;
}

export class PlaybackClock {
  constructor(duration, now = () => performance.now()) {
    this.duration = duration;
    this.now = now;
    this.mediaAnchor = 0;
    this.wallAnchor = now();
    this.rate = 1;
    this.playing = false;
  }

  read() {
    const elapsed = this.playing ? (this.now() - this.wallAnchor) * this.rate : 0;
    return Math.min(this.duration, this.mediaAnchor + elapsed);
  }

  seek(ms) {
    this.mediaAnchor = Math.max(0, Math.min(this.duration, ms));
    this.wallAnchor = this.now();
  }

  play() {
    if (this.playing) return;
    this.wallAnchor = this.now();
    this.playing = true;
  }

  pause() {
    this.seek(this.read());
    this.playing = false;
  }

  setRate(rate) {
    if (!Number.isFinite(rate) || rate <= 0) throw new RangeError('invalid playback speed');
    this.seek(this.read());
    this.rate = rate;
  }

  until(ms) {
    return Math.max(0, this.wallAnchor + (ms - this.mediaAnchor) / this.rate - this.now());
  }
}
