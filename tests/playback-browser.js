async page => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  const uiStates = [];
  page.on('request', r => { if (r.url().includes('/state?')) uiStates.push(r.url()); });

  // Keep the second segment unavailable while its events become due.
  let release, intercepted = false;
  const gate = new Promise(resolve => { release = resolve; });
  await page.route('**/replay?seq=32', async route => {
    intercepted = true;
    await gate;
    await route.continue();
  });
  const rows = page.locator('#session-list .session-row');
  await rows.filter({ hasText: '40 次变化' }).click();
  const slider = page.locator('#scrubber');
  const button = page.locator('#playback');
  const speed = page.locator('#playback-speed');
  const seq = page.locator('#history-board .seq');
  const waitSeq = n => page.waitForFunction(n => document.querySelector('#history-board .seq').textContent  ===  '序号 ' + n, n);
  async function seek(ms, n) {
    await slider.evaluate((slider, ms) => {
      slider.value = ms;
      slider.dispatchEvent(new Event('input', { bubbles: true }));
    }, ms);
    await waitSeq(n);
  }
  await page.waitForFunction(() => Number(document.getElementById('scrubber').max) === 40000);
  check(await speed.inputValue() === '1', 'initial playback rate');
  await seek(7200, 7);
  check((await slider.inputValue()) === '7200', 'seek lost 200ms within the interval');
  const transitions = await page.evaluate(async () => {
    const button = document.getElementById('playback'), slider = document.getElementById('scrubber');
    const speed = document.getElementById('playback-speed');
    const read = () => Number(slider.value);
    const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
    function rate(value) {
      speed.value = value;
      speed.dispatchEvent(new Event('change', { bubbles: true }));
    }
    rate('0.5');
    const start = performance.now();
    button.click();
    await wait(260);
    const pause = performance.now();
    button.click();
    const frozen = read();
    await wait(150);
    if (read() !== frozen || button.textContent !== '播放') throw new Error('paused media advanced');
    const resumed = performance.now();
    button.click();
    await wait(120);
    const switchAt = performance.now();
    rate('2');
    const switched = read();
    await wait(180);
    const end = performance.now();
    button.click();
    const expectedSwitch = frozen + (switchAt - resumed) * 0.5;
    return {
      frozen, expectedFrozen: 7200 + (pause - start) * 0.5,
      switched, expectedSwitch,
      end: read(), expectedEnd: expectedSwitch + (end - switchAt) * 2,
    };
  });
  for (const [actual, expected] of [['frozen', 'expectedFrozen'],['switched', 'expectedSwitch'],['end', 'expectedEnd']]) {
    check(Math.abs(transitions[actual] - transitions[expected]) < 30, 'media clock jump at ' + actual + ' ' + JSON.stringify(transitions));
  }

  await seek(31800, 31);
  await speed.selectOption('4');
  check(intercepted, 'expected a pending prefetch for seq 32');
  await page.evaluate(() => {
    const label = document.querySelector('#history-board .seq');
    window.renderedAfter31 = [];
    new MutationObserver(() => window.renderedAfter31.push(Number(label.textContent.split(' ').at(-1))))
      .observe(label, { childList: true });
  });
  await button.click();
  await page.waitForFunction(() => Number(document.getElementById('scrubber').value) >= 34200);
  check(await seq.innerText() === '序号 31', 'lost current frame while waiting for segment');
  release();
  await page.waitForFunction(() => window.renderedAfter31.length > 0);
  const caughtUp = await page.evaluate(() => window.renderedAfter31[0]);
  check(caughtUp >= 34, 'late segment displayed stale events one by one: ' + caughtUp);
  await button.click();
  await seek(39600, 39);
  await button.click();
  await waitSeq(40);
  await page.waitForFunction(() => document.getElementById('playback').textContent === '播放');
  check(await slider.inputValue() === '40000', 'player did not stop at final timestamp');

  await button.click();
  await rows.first().click();  // newest fixture has no events
  await waitSeq(0);
  await page.waitForTimeout(350);
  check(await seq.innerText() === '序号 0' && await button.isDisabled(), 'old session clock updated new session');
  check(!uiStates.length, 'unexpected per-frame state requests');
  return { passed: true, caughtUp, transitions, uiStateRequests: uiStates.length };
}
