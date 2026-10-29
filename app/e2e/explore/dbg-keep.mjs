import { launchApp, capture, closeApp, waitForShell } from '../helpers.mjs';
const { app, win } = await launchApp({ dir: process.argv[2], tutorialSeen: true }, { width: 1280, height: 720, lang: 'fr' });
win.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log('[console]', m.text().slice(0, 200)); });
win.on('response', async (r) => { if (/derush\/toggle/.test(r.url())) console.log('[resp]', r.status(), r.request().postData(), await r.text().catch(() => '')); });
try {
  await waitForShell(win);
  await win.getByRole('button', { name: 'Dérushage', exact: true }).first().click();
  await win.getByText('Black Clover', { exact: false }).first().click();
  const keep = win.getByRole('button', { name: /Garder la scène/ });
  await keep.waitFor();
  console.log('state1', await win.evaluate(() => { const v = document.querySelector('video'); return v ? { rs: v.readyState, src: v.src.slice(-50), t: v.currentTime, err: v.error && v.error.code } : 'novideo'; }));
  await keep.click();
  await win.keyboard.press('ArrowRight');
  console.log('scene label', await win.getByText(/^Scène \d+\/\d+/).first().innerText());
  await keep.click();
  await win.waitForTimeout(2000);
  console.log('scene label2', await win.getByText(/^Scène \d+\/\d+/).first().innerText());
  console.log('after keep:', (await win.locator('body').innerText()).match(/\d+ gardées dans cet épisode/)?.[0]);
  await win.waitForTimeout(8000);
  console.log('state2', await win.evaluate(() => { const v = document.querySelector('video'); return v ? { rs: v.readyState, t: v.currentTime, err: v.error && v.error.code, net: v.networkState } : 'novideo'; }));
  await capture(win, 'dbg-keep');
} finally { await closeApp(app); }
