// Exploration (real extracts): reopen a profile closed mid-indexing, resume.
import { launchApp, capture, closeApp, stubDialogs, waitForShell, apiCall } from '../helpers.mjs';
const profile = { dir: process.argv[2], tutorialSeen: true };
const { app, win, errors } = await launchApp(profile, { lang: 'fr' });
const t0 = Date.now();
const el = () => ((Date.now() - t0) / 1000).toFixed(0) + 's';
try {
  await waitForShell(win);
  await win.getByRole('button', { name: 'Bibliothèque', exact: true }).first().click();
  await win.waitForTimeout(2000);
  await capture(win, '08-reprise-apres-fermeture');
  console.log(el(), 'LIB:', (await win.locator('body').innerText()).slice(0, 1200));
  // Speed: switch to the fast profile (what a CPU user now gets by default).
  await apiCall(win, 'PUT', '/api/settings', { indexing: { mode: 'fast', sub_segmentation: false } });
  await win.getByRole('button', { name: /Lancer l'indexation/ }).click();
  let i = 0, last = '';
  while (Date.now() - t0 < 40 * 60 * 1000) {
    await win.waitForTimeout(10000);
    const bar = (await win.locator('body').innerText()).split('\n').slice(-14).join(' | ');
    if (bar !== last) { console.log(el(), bar.slice(0, 300)); last = bar; }
    if (i % 18 === 1) await capture(win, `09-indexation-reelle-${String(i).padStart(3, '0')}`);
    i++;
    if (i > 2 && /Inactif/.test(bar)) break;
  }
  await win.waitForTimeout(3000);
  await capture(win, '10-indexation-reelle-fin');
  console.log(el(), 'END:', (await win.locator('body').innerText()).slice(0, 2000));
  console.log('errors', errors);
} finally { await closeApp(app); }
