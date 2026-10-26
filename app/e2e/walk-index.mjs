import { _electron as electron } from '@playwright/test';
const CAP = '/home/leandro/dev/amv-v06-captures';
const profile = '/tmp/opencode/amv-profile-B';
const MEDIA = '/home/leandro/dev/amv-testdata/Séries test';
const app = await electron.launch({ cwd: '/home/leandro/dev/wt/amv-v06-B/app', args: ['.', `--user-data-dir=${profile}`, '--no-sandbox'] });
try {
  const win = await app.firstWindow();
  win.on('console', m => { if (m.type() === 'error') console.log('[renderer error]', m.text().slice(0,300)); });
  win.on('pageerror', e => console.log('[pageerror]', e.message));
  await app.evaluate(({ dialog }, dir) => {
    dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [dir] });
  }, MEDIA);
  await win.waitForSelector('text=AMV Tools', { timeout: 60000 });
  await win.waitForSelector('text=Bibliothèque', { timeout: 60000 });
  for (let i = 0; i < 10; i++) {
    await win.waitForTimeout(1000);
    const skip = win.locator('div.fixed').getByText('Passer', { exact: true });
    if (await skip.count()) { await skip.first().click(); await win.waitForTimeout(500); break; }
  }
  await win.getByText('Ajouter un dossier').click();
  await win.waitForTimeout(1500);
  await win.screenshot({ path: `${CAP}/05-file-attente.png` });
  console.log('QUEUE:', (await win.locator('body').innerText()).slice(0, 700));
  await win.getByRole('button', { name: /Lancer l'indexation/ }).click();
  const t0 = Date.now();
  let last = '';
  for (let i = 0; i < 90; i++) {
    await win.waitForTimeout(10000);
    const txt = (await win.locator('body').innerText());
    const line = txt.split('\n').filter(l => /Indexation|indexation|scène|scenes|%|Restant|remaining|Préparation|Analyse|Terminé|Termin/.test(l)).slice(-6).join(' | ');
    if (line !== last) { console.log(((Date.now()-t0)/1000).toFixed(0)+'s', line); last = line; }
    if (i === 2) await win.screenshot({ path: `${CAP}/06-indexation-en-cours.png` });
    if (/Inactif|Idle/.test(txt) && i > 3) { console.log('idle detected'); break; }
    if (/Terminé|Completed|Indexation terminée|fini/i.test(line)) break;
  }
  await win.screenshot({ path: `${CAP}/07-indexation-fin.png` });
  console.log('FINAL BODY:\n', (await win.locator('body').innerText()).slice(0, 2500));
} finally {
  await app.close().catch(e => console.log('close err', e.message));
}
