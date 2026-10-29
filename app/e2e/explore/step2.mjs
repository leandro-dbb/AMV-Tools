// Exploration: onboarding (install mocked) -> tutorial -> library -> real indexing.
import { makeProfile, launchApp, capture, closeApp, stubDialogs, waitForShell, apiCall } from '../helpers.mjs';
import fs from 'node:fs';
import path from 'node:path';
const MEDIA = process.env.AMV_E2E_MEDIA;
const profile = makeProfile({});
fs.writeFileSync('/tmp/amv-b-profile2', profile.dir);
console.log('profile', profile.dir);
const { app, win, errors, logs } = await launchApp(profile, { lang: 'fr' });
const t0 = Date.now();
const el = () => ((Date.now() - t0) / 1000).toFixed(0) + 's';
try {
  await win.route('**/api/setup/install', async (route) => {
    fs.writeFileSync(path.join(profile.dir, '.setup_complete'), 'cpu');
    await route.fulfill({ json: { ok: true, log: '[e2e] install skipped', hint: '' } });
  });
  await win.getByText('CPU uniquement').click();
  await win.getByText('Installer et continuer').click();
  await waitForShell(win);
  await win.waitForTimeout(1500);
  await capture(win, '03-tutoriel');
  console.log(el(), 'TUTO:', (await win.locator('body').innerText()).slice(0, 1500));
  // skip tutorial
  const skip = win.getByRole('button', { name: /Passer|Skip/ });
  if (await skip.count()) await skip.first().click();
  await win.waitForTimeout(500);
  await capture(win, '04-bibliotheque-vide');
  console.log(el(), 'LIB:', (await win.locator('body').innerText()).slice(0, 1500));
  await stubDialogs(app, { open: [MEDIA] });
  await win.getByRole('button', { name: /Ajouter un dossier/ }).click();
  await win.waitForTimeout(1500);
  await capture(win, '05-file-attente');
  console.log(el(), 'QUEUE:', (await win.locator('body').innerText()).slice(0, 1500));
  await win.getByRole('button', { name: /Lancer l'indexation|Indexer/ }).first().click();
  let i = 0, last = '';
  while (Date.now() - t0 < 25 * 60 * 1000) {
    await win.waitForTimeout(5000);
    const bar = await win.locator('body').innerText();
    const line = bar.split('\n').slice(-12).join(' | ');
    if (line !== last) { console.log(el(), line.slice(0, 400)); last = line; }
    if (i % 12 === 2) await capture(win, `06-indexation-${String(i).padStart(3, '0')}`);
    i++;
    if (i > 4 && /Inactif|Idle/.test(line)) { console.log(el(), 'idle'); break; }
  }
  await win.waitForTimeout(2000);
  await capture(win, '07-indexation-fin');
  console.log(el(), 'END:', (await win.locator('body').innerText()).slice(0, 3000));
  console.log('errors', errors);
} finally {
  await closeApp(app);
  fs.writeFileSync('/tmp/amv-b-step2-logs.txt', logs.join(''));
}
