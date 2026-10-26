// Exploration (real extracts, indexed profile): search, characters, editor,
// finishing, settings, sizes, close. Each step logs + screenshots; a failing
// step is logged and the walk continues.
import { launchApp, capture, closeApp, stubDialogs, waitForShell, apiCall } from '../helpers.mjs';
import fs from 'node:fs';
const profile = { dir: process.argv[2], tutorialSeen: true };
const only = process.argv[3] || 'all';
const { app, win, errors } = await launchApp(profile, { lang: 'fr' });
const outDir = '/tmp/amv-b-exports';
fs.mkdirSync(outDir, { recursive: true });
await stubDialogs(app, { open: [outDir] });
const t0 = Date.now();
const el = () => ((Date.now() - t0) / 1000).toFixed(0) + 's';
async function step(name, fn) {
  if (only !== 'all' && !only.split(',').includes(name)) return;
  try { await fn(); console.log(el(), 'OK', name); }
  catch (e) { console.log(el(), 'FAIL', name, String(e).split('\n')[0]); await capture(win, `x-echec-${name}`).catch(() => {}); }
}
const body = async () => (await win.locator('body').innerText());
try {
  await waitForShell(win);
  await step('search', async () => {
    await win.getByRole('button', { name: 'Recherche', exact: true }).click();
    const box = win.getByPlaceholder(/./).first();
    for (const [i, q] of ['Goku qui crie', 'explosion', 'gros plan visage', 'Goku screaming', 'close-up face', 'Beerus'].entries()) {
      await box.fill(q); await box.press('Enter');
      await win.locator('div.group.cursor-pointer img').first().waitFor({ timeout: 90000 }).catch(() => {});
      await win.waitForTimeout(800);
      const n = await win.locator('div.group.cursor-pointer').count();
      const top = await win.locator('div.group.cursor-pointer img').evaluateAll((ims) => ims.slice(0, 4).map((i) => i.alt));
      console.log(`  query "${q}": ${n} results; top: ${top.join(' | ')}`);
      await capture(win, `11-recherche-${i}-${q.replace(/\s+/g, '-')}`);
    }
  });
  await step('hover', async () => {
    const card = win.locator('div.group.cursor-pointer').first();
    await card.hover(); await win.waitForTimeout(2500);
    console.log('  hover preview present:', await win.getByTestId('hover-preview').count());
    await capture(win, '12-survol');
  });
  await step('filters', async () => {
    await win.getByRole('button', { name: /Filtres/ }).click();
    await win.waitForTimeout(800);
    await capture(win, '13-filtres');
    console.log('  filters:', (await body()).split('Filtres')[1]?.slice(0, 600).replace(/\n/g, ' | '));
  });
  await step('similar', async () => {
    await win.mouse.move(5, 5);
    const card = win.locator('div.group.cursor-pointer').first();
    await card.hover();
    await card.getByRole('button').first().click();
    await win.waitForTimeout(3000);
    await capture(win, '14-similaires');
  });
  await step('characters', async () => {
    await win.getByRole('button', { name: 'Personnages', exact: true }).click();
    await win.waitForTimeout(1000);
    await capture(win, '15-personnages-vide');
    console.log('  chars:', (await body()).slice(0, 800).replace(/\n/g, ' | '));
  });
  await step('editor', async () => {
    await win.getByRole('button', { name: 'Recherche', exact: true }).click();
    await win.locator('div.group.cursor-pointer').first().click();
    await win.waitForTimeout(3000);
    await capture(win, '16-editeur');
    console.log('  editor:', (await body()).slice(0, 600).replace(/\n/g, ' | '));
  });
  await step('settings', async () => {
    await win.keyboard.press('Escape');
    await win.getByRole('button', { name: 'Fermer' }).click().catch(() => {});
    await win.getByRole('button', { name: 'Paramètres', exact: true }).click();
    for (const s of ['Indexation', 'Modèles', 'Recherche', 'Export', 'Interface', 'Diagnostic']) {
      await win.locator('div.w-64').getByRole('button', { name: s }).click().catch((e) => console.log('  no section', s));
      await win.waitForTimeout(1500);
      await capture(win, `17-parametres-${s}`);
    }
  });
  await step('sizes', async () => {
    for (const [w, h] of [[1280, 720], [2560, 1440]]) {
      await app.evaluate(({ BrowserWindow }, s) => BrowserWindow.getAllWindows()[0].setSize(s[0], s[1]), [w, h]);
      await win.getByRole('button', { name: 'Recherche', exact: true }).click();
      await win.waitForTimeout(1200);
      await capture(win, `18-taille-${w}x${h}-recherche`);
      await win.getByRole('button', { name: 'Dérushage', exact: true }).click();
      await win.waitForTimeout(1500);
      await capture(win, `18-taille-${w}x${h}-derush`);
    }
  });
  console.log('errors', errors);
} finally {
  const tc = Date.now();
  await closeApp(app);
  console.log('close took', Date.now() - tc, 'ms');
}
