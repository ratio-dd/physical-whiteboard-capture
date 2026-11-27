import test from 'node:test';
import assert from 'node:assert/strict';
import { eventAt, eventOffsets, PlaybackClock } from '../web/playback-clock.js';

test('confirmation times become a monotone timeline relative to session creation', () => {
  const events = [
    { seq: 0, confirmed_at: 1000 }, { seq: 1, confirmed_at: 1200 },
    { seq: 2, confirmed_at: 1200 }, { seq: 3, confirmed_at: 1000 },
    { seq: 4, confirmed_at: 2200 },
  ];
  const offsets = eventOffsets(events, 1000);
  assert.deepEqual(offsets, [0, 200, 200, 200, 1200]);
  assert.deepEqual([0, 199, 200, 201, 1200].map(ms => eventAt(offsets, ms)), [0, 0, 3, 3, 4]);
});

test('clock preserves pause position and changing rate keeps the same media time', () => {
  let now = 0;
  const clock = new PlaybackClock(1200, () => now);
  clock.play(); now = 200;
  assert.equal(clock.read(), 200);
  clock.pause(); now += 200;
  assert.equal(clock.read(), 200);
  clock.setRate(2); clock.play(); now += 200;
  assert.equal(clock.read(), 600);
  clock.setRate(0.5); assert.equal(clock.read(), 600);
  now += 200;
  assert.equal(clock.read(), 700);
});

test('clock clamps seek, reaches the exact tail, and reports playing state', () => {
  let now = 100;
  const clock = new PlaybackClock(1200, () => now);
  clock.seek(-5); assert.equal(clock.read(), 0);
  clock.play(); now += 1200;
  assert.equal(clock.read(), 1200);
  assert.equal(clock.playing, true, 'clock only reports media position at the tail');
  clock.pause();
  assert.equal(clock.playing, false, 'controller pauses after the tail frame is drawn');
  clock.seek(9999); assert.equal(clock.read(), 1200);
});

test('until keeps the original media deadline across asynchronous decode delay', () => {
  let now = 0;
  const clock = new PlaybackClock(5000, () => now);
  clock.play();
  now = 100;
  const wait = clock.until(1000);
  assert.equal(wait, 900, 'deadline is 900 wall milliseconds away at 1x');
  now = 2100; // decode completes 2 seconds later, after the original deadline
  assert.equal(clock.read(), 2100);
  assert.equal(clock.until(1000), 0, 'an overdue deadline is immediately due');
});

test('each supported rate maps the same wall interval to exact media duration', () => {
  for (const [rate, expected] of [[0.5, 100], [1, 200], [2, 400], [4, 800]]) {
    let now = 0;
    const clock = new PlaybackClock(10_000, () => now);
    clock.setRate(rate); clock.play(); now = 200;
    assert.equal(clock.read(), expected, `${rate}x media duration`);
  }
});

test('all supported rates stay finite and monotonic on sparse intervals', () => {
  for (const rate of [0.5, 1, 2, 4]) {
    let now = 0;
    const clock = new PlaybackClock(60_000, () => now);
    clock.setRate(rate); clock.play();
    let previous = 0;
    for (let i = 0; i < 100; i += 1) {
      now += 137;
      const current = clock.read();
      assert.ok(Number.isFinite(current));
      assert.ok(current >= previous, `rate ${rate} moved backwards`);
      previous = current;
    }
    assert.ok(previous > 0);
  }
});
