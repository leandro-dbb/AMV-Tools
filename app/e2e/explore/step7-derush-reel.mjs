// Exploration (real extracts): derush both episodes (HEVC + H.264), keep shots, basket.
import { launchApp, capture, closeApp, waitForShell, stubDialogs } from '../helpers.mjs';
const profile = { dir: process.argv[2], tutorialSeen: true };
const { app, win } = await launchApp(profile, { lang: 'fr' });
await stubDialogs(app, {});
const body = async () => (await win.locator('body').innerText());
try {
  await waitForShell(win);
  await win.getByRole('button', { name: 'Dérushage', exact: true }).first().click();
  for (const ep of ['Black Clover S02E04 - extrait', 'DBS Beerus 01 - extrait']) {
    await win.getByText(ep, { exact: true }).first().click();
    await win.waitForTimeout(6000);
    await capture(win, `32-derush-${ep.split(' ')[0]}`);
    console.log(ep, ':', (await body()).split('ÉPISODES')[1]?.slice(0, 500).replace(/\n/g, ' | '));
    const keep = win.getByRole('button', { name: /Garder la scène/ });
    if (await keep.count()) { await keep.click(); await win.keyboard.press('ArrowRight'); await win.waitForTimeout(800); await keep.click().catch(() => {}); }
  }
  const gen = win.getByRole('button', { name: /Générer une version lisible/ });
  if (await gen.count()) {
    const t = Date.now(); await gen.click();
    await win.getByRole('button', { name: /Garder la scène/ }).waitFor({ timeout: 400000 }).catch(() => {});
    console.log('transcode took', (Date.now() - t) / 1000, 's');
    await capture(win, '33-derush-dbs-apres-transcodage');
  }
  await win.getByRole('button', { name: /^Bibliothèque \(\d+\)$/ }).click();
  await win.waitForTimeout(1500);
  await capture(win, '34-panier-reel');
  console.log('basket:', (await body()).split('Tout (')[1]?.slice(0, 400).replace(/\n/g, ' | '));
} catch (e) { console.log('FAIL', String(e).split('\n')[0]); await capture(win, 'x-echec-derush-reel'); }
finally { await closeApp(app); }
