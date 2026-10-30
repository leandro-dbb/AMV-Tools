// Roto editor end to end (v06), against the built Electron app, the real
// sidecar and the real models: open a shot, cut it out with the AI, two brush
// corrections and one click, propagate, export ProRes 4444 with alpha.
// Meant for a real extract: AMV_E2E_MEDIA=<folder with one anime extract>
// (synthetic test patterns have no character to cut out, but the path runs).
import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import {
  apiCall, capture, closeApp, launchApp, makeProfile, makeTempDir, prepareMedia, stubDialogs,
} from './helpers.mjs';

test.describe.configure({ mode: 'serial' });

let ctx;
const media = prepareMedia();
const exportDir = path.join(makeTempDir('amv-e2e-roto-'), 'Exports détourage');
const query = process.env.AMV_E2E_ROTO_QUERY || 'cigarette';

test.beforeAll(async () => {
  fs.mkdirSync(exportDir, { recursive: true });
  const profile = makeProfile({ setupDone: true, tutorialSeen: true });
  ctx = { profile, ...(await launchApp(profile, { width: 1600, height: 900, lang: 'fr' })) };
  await stubDialogs(ctx.app, { open: [media.dir] });
});

test.afterEach(async ({}, info) => {
  if (info.status !== info.expectedStatus && ctx?.win) await capture(ctx.win, `roto-echec-${info.title.slice(0, 2).trim()}`).catch(() => {});
});

test.afterAll(async () => {
  if (ctx?.errors.length) console.log('renderer errors:', ctx.errors);
  if (ctx) await closeApp(ctx.app);
});

test('1. un vrai extrait indexé', async () => {
  const { win } = ctx;
  await win.getByRole('button', { name: /^(Bibliothèque|Library)$/ }).first().waitFor({ timeout: 180_000 });
  await apiCall(win, 'PUT', '/api/settings', { export: { output_folder: exportDir } });
  await win.getByRole('button', { name: /^Bibliothèque/ }).first().click();
  await win.getByRole('button', { name: 'Ajouter un dossier' }).click();
  await win.getByRole('button', { name: /Lancer l'indexation/ }).click();
  await expect(win.getByText('Inactif')).toBeVisible({ timeout: 45 * 60_000 });
  await expect(win.getByText(`${media.files.length} épisode`, { exact: false })).toBeVisible();
});

test('2. ouvrir un plan et le détourer avec l’IA', async () => {
  const { win } = ctx;
  await win.getByRole('button', { name: 'Recherche', exact: true }).first().click();
  const box = win.getByPlaceholder(/./).first();
  await win.locator('input[type=range]').first().fill('0');
  await box.fill(query);
  await box.press('Enter');
  const card = win.locator('div.group.cursor-pointer').first();
  await expect(card).toBeVisible({ timeout: 180_000 });
  await card.click();
  await win.getByTestId('roto-open').click();
  await expect(win.getByTestId('roto-editor')).toBeVisible({ timeout: 120_000 });
  await expect(win.getByTestId('roto-first-pass')).toBeVisible({ timeout: 120_000 });
  await capture(win, 'roto-01-ouvert');
  await win.getByTestId('roto-detect').click();
  await expect(win.getByTestId('roto-progress')).toBeVisible();
  await capture(win, 'roto-02-detourage-en-cours');
  await expect(win.getByTestId('roto-progress')).toHaveCount(0, { timeout: 30 * 60_000 });
  await expect(win.getByTestId('roto-first-pass')).toHaveCount(0, { timeout: 60_000 });
  await expect(win.getByTestId('roto-error')).toHaveCount(0);
  await capture(win, 'roto-03-passe-ia');
});

async function canvasBox(win) {
  const box = await win.getByTestId('roto-canvas').boundingBox();
  if (!box) throw new Error('no canvas');
  return box;
}

test('3. deux coups de pinceau et un clic, mesurés', async () => {
  const { win } = ctx;
  // to a suspect frame when there is one (N), else the middle of the shot
  const next = win.getByTestId('roto-next-suspect');
  if (await next.isEnabled()) await next.click();
  await win.getByTestId('roto-tool-brush').click();
  await win.getByTestId('roto-brush-add').click();
  const b = await canvasBox(win);
  const cx = b.x + b.width / 2;
  const cy = b.y + b.height / 2;
  const stroke = async (x0, y0, x1, y1) => {
    await win.mouse.move(x0, y0);
    await win.mouse.down();
    for (let k = 1; k <= 8; k += 1) await win.mouse.move(x0 + ((x1 - x0) * k) / 8, y0 + ((y1 - y0) * k) / 8);
    await win.mouse.up();
  };
  await stroke(cx - 60, cy - 20, cx + 60, cy - 10);
  await expect(win.getByTestId('roto-latency')).toBeVisible({ timeout: 60_000 });
  await expect(win.getByTestId('roto-keyframe').first()).toBeVisible();
  await win.getByTestId('roto-brush-remove').click();
  await stroke(b.x + 30, b.y + 30, b.x + 120, b.y + 40);
  await expect(win.getByTestId('roto-pending')).toHaveCount(0, { timeout: 60_000 });
  await win.getByTestId('roto-tool-click').click();
  await win.mouse.click(cx + 20, cy + 40);
  await expect(win.getByTestId('roto-pending')).toHaveCount(0, { timeout: 120_000 });
  await expect(win.getByTestId('roto-error')).toHaveCount(0);
  console.log('latency:', await win.getByTestId('roto-latency').innerText());
  await capture(win, 'roto-04-corrections');
  // undo / redo of the click
  await win.getByTestId('roto-undo').click();
  await win.getByTestId('roto-redo').click();
  await expect(win.getByTestId('roto-error')).toHaveCount(0);
});

test('4. propager dans les deux sens', async () => {
  const { win } = ctx;
  await win.getByTestId('roto-propagate-both').click();
  await expect(win.getByTestId('roto-progress')).toBeVisible();
  await expect(win.getByTestId('roto-progress')).toHaveCount(0, { timeout: 20 * 60_000 });
  await expect(win.getByTestId('roto-error')).toHaveCount(0);
  await capture(win, 'roto-05-propagation');
});

test('5. exporter en ProRes 4444 avec transparence', async () => {
  const { win } = ctx;
  await win.getByTestId('roto-export').click();
  await expect(win.getByTestId('roto-export-done')).toBeVisible({ timeout: 30 * 60_000 });
  await capture(win, 'roto-06-export');
  const text = await win.getByTestId('roto-export-done').innerText();
  expect(text).toContain('_alpha.mov');
  const found = [];
  const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).forEach((e) => {
    const p = path.join(d, e.name);
    if (e.isDirectory()) walk(p); else found.push(p);
  });
  walk(exportDir);
  const prores = found.find((f) => f.endsWith('_alpha.mov'));
  expect(prores, found.join('\n')).toBeTruthy();
  expect(fs.statSync(prores).size).toBeGreaterThan(100_000);
  expect(found.some((f) => f.endsWith('_matte.mov'))).toBe(true);
  // the cut-out shot is now in the selection, linked in the timeline export
  const kept = await apiCall(win, 'GET', '/api/derush/kept');
  expect(kept.status).toBe(200);
  expect(JSON.stringify(kept.json)).toMatch(/\d/);
});
