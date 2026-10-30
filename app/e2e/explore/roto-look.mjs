// Drive the roto editor against a running sidecar (127.0.0.1:8731) served by
// `vite preview` on :5173 (built with VITE_API_PORT=8731), in our own
// Chromium under Xvfb. Screenshots of each step in <out dir>.
// usage: node e2e/explore/roto-look.mjs <out dir> [video name]
import { chromium } from '@playwright/test';
import path from 'node:path';

const out = process.argv[2] || '/tmp/roto-look';
const video = process.argv[3] || 'Black Clover 01 cigarette';
const exe = process.env.AMV_CHROME || (process.env.HOME + '/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome');
const browser = await chromium.launch({ headless: false, executablePath: exe });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
page.on('pageerror', (e) => console.log('pageerror', e.message));
page.on('console', (m) => { if (m.type() === 'error') console.log('console', m.text()); });
await page.addInitScript(() => { localStorage.setItem('amv-tutorial-shown', '1'); localStorage.setItem('amv-lang', 'fr'); });
await page.goto('http://127.0.0.1:5173/');
const shot = async (n) => page.screenshot({ path: path.join(out, `${n}.png`) });
await page.getByRole('button', { name: /^Recherche/ }).first().click();
const q = page.locator('input[type="text"], input:not([type])').first();
await q.fill(process.argv[4] || 'cigarette');
await q.press('Enter');
await page.waitForTimeout(2000);
const slider = page.locator('input[type="range"]').first();
if (await slider.count()) { await slider.focus(); await page.keyboard.press('Home'); }
await page.waitForTimeout(3000);
await shot('02-search');
const cards = page.locator('div.group.cursor-pointer');
await cards.first().waitFor({ timeout: 180000 });
await page.waitForTimeout(1500);
await shot('02-search');
console.log(await cards.count(), 'scene cards');
await cards.first().click();
await page.waitForTimeout(2500);
await shot('03-editor');
await page.getByTestId('roto-open').click();
await page.getByTestId('roto-editor').waitFor({ timeout: 120000 });
await page.waitForTimeout(3000);
await shot('04-roto-open');
await page.getByTestId('roto-detect').click();
await page.waitForTimeout(4000);
await shot('05-detecting');
await page.getByTestId('roto-canvas').waitFor();
await page.waitForFunction(() => !document.querySelector('[data-testid="roto-progress"]'), null, { timeout: 600000 });
await page.waitForTimeout(3000);
await shot('06-ai-done');
console.log('frame', await page.getByTestId('roto-frame-label').innerText());
console.log('suspects', await page.getByTestId('roto-suspects-left').innerText().catch(() => '?'));
console.log('models', await page.getByTestId('roto-models').innerText().catch(() => '?'));
await browser.close();
