import { byId } from './common.js';
import { setupCapture } from './capture.js';
import { setupHistory } from './history.js';

setupCapture();
const historyView = setupHistory();
function navigate() {
  const history = location.hash === '#history';
  byId('capture-view').hidden = history;
  byId('history-view').hidden = !history;
  for (const [nav, current] of [['nav-live', !history], ['nav-history', history]]) {
    if (current) byId(nav).setAttribute('aria-current', 'page');
    else byId(nav).removeAttribute('aria-current');
  }
  if (history) void historyView.open();
  else historyView.close();
  window.scrollTo(0, 0);
}
window.addEventListener('hashchange', navigate);
navigate();
