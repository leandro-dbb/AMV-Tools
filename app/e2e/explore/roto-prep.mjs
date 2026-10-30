// Screenshots of a roto project's frames (overlay view) to choose where the
// demo corrects. usage: node e2e/explore/roto-prep.mjs <out> <query> <frames,...>
import { chromium } from '@playwright/test';
import path from 'node:path';

const [out, query, frames] = process.argv.slice(2);
const exe = process.env.AMV_CHROME || (process.env.HOME + '/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome');
const browser = await chromium.launch({ headless: false, executablePath: exe });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
page.on('pageerror', (e) => console.log('pageerror', e.message));
await page.addInitScript(() => { localStorage.setItem('amv-tutorial-shown', '1'); localStorage.setItem('amv-lang', 'fr'); });
await page.goto('http://127.0.0.1:5173/');
await page.getByRole('button', { name: /^Recherche/ }).first().click();
const q = page.locator('input[type="text"], input:not([type])').first();
await page.locator('input[type="range"]').first().fill('0');
await q.fill(query);
await q.press('Enter');
const cards = page.locator('div.group.cursor-pointer');
await cards.first().waitFor({ timeout: 180000 });
await cards.first().click();
await page.getByTestId('roto-open').click();
await page.getByTestId('roto-canvas').waitFor({ timeout: 120000 });
await page.waitForTimeout(3000);
const box = await page.getByTestId('roto-canvas').boundingBox();
console.log('canvas', JSON.stringify(box));
for (const f of (frames || '0').split(',').map(Number)) {
  await page.keyboard.press('Home');
  for (let k = 0; k < f; k += 1) await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(out, `prep-${f}.png`) });
}
console.log('suspects', await page.getByTestId('roto-suspects-left').innerText());
await browser.close();
