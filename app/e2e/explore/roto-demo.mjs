// Screen recording of the roto editor in use (demo for the v06 report), in
// our own Chromium under Xvfb, against a sidecar on 127.0.0.1:8731 (a GPU
// machine through an SSH tunnel) and the front served by `vite preview` on
// :5173 (built with VITE_API_PORT=8731). Writes <out>/demo.webm and
// <out>/demo-events.json (wait segments, to speed them up when editing).
// usage: node e2e/explore/roto-demo.mjs <out> <plan.json>
//   plan.json: {"query": "...", "card": 0, "target": "all",
//               "steps": [{"frame": 12, "do": "brush-add"|"brush-remove"|"click+"|"click-",
//                          "pts": [[fx, fy], ...], "size": 14}, {"do": "propagate"}, ...]}
import { chromium } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

const [out, planPath] = process.argv.slice(2);
const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));
fs.mkdirSync(out, { recursive: true });
const exe = process.env.AMV_CHROME || (process.env.HOME + '/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome');
const W = 1600; const H = 900;
const browser = await chromium.launch({ headless: false, executablePath: exe, args: ['--window-size=1600,900'] });
const context = await browser.newContext({ viewport: { width: W, height: H }, recordVideo: { dir: out, size: { width: W, height: H } } });
const page = await context.newPage();
const t0 = Date.now();
const events = [];
const mark = (kind, label) => events.push({ t: (Date.now() - t0) / 1000, kind, label });
page.on('pageerror', (e) => console.log('pageerror', e.message));
await page.addInitScript(() => {
  localStorage.setItem('amv-tutorial-shown', '1');
  localStorage.setItem('amv-lang', 'fr');
  // a visible cursor (Playwright's mouse is invisible in recordings)
  window.addEventListener('DOMContentLoaded', () => {
    const c = document.createElement('div');
    c.id = 'demo-cursor';
    Object.assign(c.style, { position: 'fixed', left: '0px', top: '0px', width: '18px', height: '18px', marginLeft: '-9px', marginTop: '-9px',
      borderRadius: '50%', border: '2px solid white', background: 'rgba(236,72,153,0.55)', zIndex: 2147483647, pointerEvents: 'none',
      boxShadow: '0 0 6px rgba(0,0,0,0.8)', transition: 'transform 80ms' });
    document.body.appendChild(c);
    window.addEventListener('mousemove', (e) => { c.style.left = `${e.clientX}px`; c.style.top = `${e.clientY}px`; }, true);
    window.addEventListener('mousedown', () => { c.style.transform = 'scale(0.7)'; }, true);
    window.addEventListener('mouseup', () => { c.style.transform = 'scale(1)'; }, true);
  });
});
const pause = (ms) => page.waitForTimeout(ms);
async function glide(x, y, steps = 18) { await page.mouse.move(x, y, { steps }); }
async function clickEl(loc) {
  const b = await loc.boundingBox();
  await glide(b.x + b.width / 2, b.y + b.height / 2);
  await pause(250);
  await loc.click();
}
async function caption(text, ms = 2200) {
  await page.evaluate((t) => {
    let c = document.getElementById('demo-caption');
    if (!c) {
      c = document.createElement('div');
      c.id = 'demo-caption';
      Object.assign(c.style, { position: 'fixed', left: '50%', bottom: '120px', transform: 'translateX(-50%)', padding: '10px 18px',
        background: 'rgba(0,0,0,0.82)', color: 'white', font: '600 20px system-ui, sans-serif', borderRadius: '10px', zIndex: 2147483646,
        pointerEvents: 'none', border: '1px solid rgba(236,72,153,0.7)' });
      document.body.appendChild(c);
    }
    c.textContent = t;
    c.style.display = t ? 'block' : 'none';
  }, text);
  if (ms) await pause(ms);
}
async function waitJob(label) {
  mark('wait-start', label);
  await page.getByTestId('roto-progress').waitFor({ timeout: 60_000 }).catch(() => {});
  await page.waitForFunction(() => !document.querySelector('[data-testid="roto-progress"]'), null, { timeout: 30 * 60_000 });
  mark('wait-end', label);
}
async function xform() {
  const b = await page.getByTestId('roto-canvas').boundingBox();
  const s = Math.min(b.width / plan.w, b.height / plan.h) * 0.96;
  return { b, s, tx: (b.width - plan.w * s) / 2, ty: (b.height - plan.h * s) / 2 };
}
const toScreen = (xf, fx, fy) => [xf.b.x + xf.tx + fx * xf.s, xf.b.y + xf.ty + fy * xf.s];
async function gotoFrame(f) {
  await page.keyboard.press('Home');
  for (let k = 0; k < f; k += 1) await page.keyboard.press('ArrowRight');
  await pause(700);
}

// 1. find the shot, open it, "Détourer ce plan"
await page.goto('http://127.0.0.1:5173/');
await pause(1500);
await clickEl(page.getByRole('button', { name: /^Recherche/ }).first());
await page.locator('input[type="range"]').first().fill('0');
const q = page.locator('input[type="text"], input:not([type])').first();
await clickEl(q);
await q.type(plan.query, { delay: 45 });
await q.press('Enter');
const cards = page.locator('div.group.cursor-pointer');
await cards.first().waitFor({ timeout: 180_000 });
await pause(1200);
await clickEl(cards.nth(plan.card || 0));
await pause(1500);
await caption('Un plan ouvert : « Détourer ce plan »', 0);
await clickEl(page.getByTestId('roto-open'));
await page.getByTestId('roto-editor').waitFor({ timeout: 120_000 });
await pause(1500);
await caption('L’IA fait le premier jet sur tout le plan', 0);
if (plan.target && plan.target !== 'all') await clickEl(page.getByTestId(`roto-target-${plan.target}`));
await pause(800);
await clickEl(page.getByTestId('roto-detect'));
await waitJob('detect');
await caption('', 0);
await pause(1500);

// 2. views and suspects
await caption('Les images à vérifier sont en rouge sous la vidéo (N = suivante)', 2600);
for (const k of ['2', '3', '4', '1']) { await page.keyboard.press(k); await pause(900); }
await caption('', 0);

// 3. the corrections of the plan
for (const st of plan.steps) {
  if (st.caption) await caption(st.caption, 1600);
  if (st.frame !== undefined) await gotoFrame(st.frame);
  if (st.do === 'suspect') { await clickEl(page.getByTestId('roto-next-suspect')); await pause(1200); continue; }
  if (st.do === 'zoom') {
    const xf = await xform();
    const [x, y] = toScreen(xf, st.pts[0][0], st.pts[0][1]);
    await glide(x, y);
    for (let k = 0; k < (st.n || 4); k += 1) { await page.mouse.wheel(0, -240); await pause(120); }
    await pause(1000);
    continue;
  }
  if (st.do === 'fit') { await page.keyboard.press('0'); await pause(800); continue; }
  if (st.do === 'propagate') {
    await clickEl(page.getByTestId(`roto-propagate-${st.dir || 'both'}`));
    await waitJob('propagate');
    await pause(1200);
    continue;
  }
  if (st.do === 'play') { await page.keyboard.press('Space'); await pause(st.ms || 3500); await page.keyboard.press('Space'); continue; }
  if (st.do === 'key') { await page.keyboard.press(st.key); await pause(st.ms || 900); continue; }
  if (st.do === 'edge') {
    const sl = page.getByTestId(st.testid);
    await clickEl(sl);
    if (st.value !== undefined) await sl.fill(String(st.value));
    await pause(1500);
    continue;
  }
  const xf = await xform();
  if (st.do.startsWith('brush')) {
    await clickEl(page.getByTestId('roto-tool-brush'));
    await clickEl(page.getByTestId(st.do === 'brush-add' ? 'roto-brush-add' : 'roto-brush-remove'));
    if (st.size) await page.getByTestId('roto-brush-size').fill(String(st.size));
    const pts = st.pts.map(([fx, fy]) => toScreen(xf, fx, fy));
    await glide(pts[0][0], pts[0][1]);
    await page.mouse.down();
    for (const [x, y] of pts.slice(1)) await page.mouse.move(x, y, { steps: 8 });
    await page.mouse.up();
  } else {
    await clickEl(page.getByTestId('roto-tool-click'));
    const [x, y] = toScreen(xf, st.pts[0][0], st.pts[0][1]);
    await glide(x, y);
    await pause(300);
    if (st.do === 'click-') await page.mouse.click(x, y, { button: 'right' });
    else await page.mouse.click(x, y);
  }
  await page.getByTestId('roto-pending').waitFor({ state: 'detached', timeout: 120_000 }).catch(() => {});
  await pause(1600);
  const lat = await page.getByTestId('roto-latency').innerText().catch(() => '');
  mark('latency', lat);
  console.log('latency', lat);
}

// 4. export
await caption('Export : ProRes 4444 avec transparence, PNG, masque — et le plan part dans la sélection', 0);
await clickEl(page.getByTestId('roto-export'));
mark('wait-start', 'export');
await page.getByTestId('roto-export-done').waitFor({ timeout: 30 * 60_000 });
mark('wait-end', 'export');
await pause(2500);
await caption('', 0);
await page.keyboard.press('Space');
await pause(4000);
await page.keyboard.press('Space');
await pause(800);
mark('end', '');
fs.writeFileSync(path.join(out, 'demo-events.json'), JSON.stringify(events, null, 1));
await context.close();
const vid = fs.readdirSync(out).filter((f) => f.endsWith('.webm')).map((f) => path.join(out, f))
  .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0];
if (vid) fs.renameSync(vid, path.join(out, 'demo.webm'));
await browser.close();
console.log('done', JSON.stringify(events));
