import { byId } from './common.js';
import { setupCapture } from './capture.js';
import { setupHistory } from './history.js';
import { setupInk } from './ink-view.js';

setupCapture();
const historyView = setupHistory();
const inkView = setupInk();
function navigate() {
  const history = location.hash === '#history';
  const ink = location.hash === '#ink';
  byId('capture-view').hidden = history || ink;
  byId('history-view').hidden = !history;
  byId('ink-view').hidden = !ink;
  for (const [nav, current] of [['nav-live', !history && !ink], ['nav-history', history], ['nav-ink', ink]]) {
    if (current) byId(nav).setAttribute('aria-current', 'page');
    else byId(nav).removeAttribute('aria-current');
  }
  if (history) void historyView.open();
  else historyView.close();
  if (ink) void inkView.open();
  else inkView.close();
  window.scrollTo(0, 0);
}
window.addEventListener('hashchange', navigate);
navigate();
