// Exploration: finishing queue after a restart, then queue + cancel a job.
import { launchApp, capture, closeApp, waitForShell } from '../helpers.mjs';
const profile = { dir: process.argv[2], tutorialSeen: true };
const { app, win } = await launchApp(profile, { lang: 'fr' });
const body = async () => (await win.locator('body').innerText());
try {
  await waitForShell(win);
  await win.waitForTimeout(3000);
  console.log('finish badge after restart:', await win.getByRole('button', { name: /^Finition/ }).count());
  await capture(win, '25-apres-redemarrage-job-perdu');
  await win.getByRole('button', { name: 'Recherche', exact: true }).first().click();
  await win.locator('input[type=range]').first().fill('0');
  const box = win.getByPlaceholder(/./).first();
  await box.fill('explosion'); await box.press('Enter');
  await win.waitForTimeout(1500);
  await capture(win, '25b-premiere-recherche-chargement');
  await win.locator('div.group.cursor-pointer').first().waitFor({ timeout: 150000 });
  await win.locator('div.group.cursor-pointer').first().click();
  await win.getByRole('button', { name: /^Finition/ }).first().click();
  await win.getByRole('button', { name: /^Profondeur/ }).click();
  const dl = win.getByRole('button', { name: /^Télécharger/ });
  if (await dl.count()) { const t = Date.now(); await dl.click(); await win.getByRole('button', { name: /Ajouter à la file/ }).and(win.locator(':enabled')).waitFor({ timeout: 240000 }); console.log('depth download', (Date.now() - t) / 1000, 's'); }
  await capture(win, '26-profondeur-dialogue');
  await win.getByRole('button', { name: /Ajouter à la file/ }).click();
  await win.waitForTimeout(12000);
  await capture(win, '27-profondeur-en-cours');
  console.log('running:', (await body()).split('File de finition')[1]?.slice(0, 300).replace(/\n/g, ' | '));
  await win.getByRole('button', { name: 'Annuler', exact: true }).last().click();
  await win.waitForTimeout(4000);
  await capture(win, '28-profondeur-annulee');
  console.log('after cancel:', (await body()).split('File de finition')[1]?.slice(0, 300).replace(/\n/g, ' | '));
} catch (e) { console.log('FAIL', String(e).split('\n')[0]); await capture(win, 'x-echec-cancel2'); }
finally { await closeApp(app); }
