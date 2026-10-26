import { _electron as electron } from '@playwright/test';
const CAP = '/home/leandro/dev/amv-v06-captures';
const profile = '/tmp/opencode/amv-profile-B';
const app = await electron.launch({
  cwd: '/home/leandro/dev/wt/amv-v06-B/app',
  args: ['.', `--user-data-dir=${profile}`, '--no-sandbox'],
});
try {
  const win = await app.firstWindow();
  win.on('console', m => { if (m.type() === 'error') console.log('[renderer error]', m.text().slice(0,300)); });
  win.on('pageerror', e => console.log('[pageerror]', e.message));
  const t0 = Date.now();
  await win.waitForSelector('text=AMV Tools', { timeout: 30000 });
  await win.waitForTimeout(800);
  await win.screenshot({ path: `${CAP}/01-bootstrap.png` });
  await win.waitForSelector('text=CPU uniquement', { timeout: 180000 });
  console.log('onboarding after', ((Date.now()-t0)/1000).toFixed(1), 's');
  await win.screenshot({ path: `${CAP}/02-onboarding-choix.png` });
  await win.getByText('CPU uniquement').click();
  await win.getByText('Installer et continuer').click();
  await win.waitForTimeout(300);
  await win.screenshot({ path: `${CAP}/03-onboarding-install.png` });
  await win.waitForFunction(() => /Bibliothèque|Library/.test(document.body.innerText), null, { timeout: 600000 });
  console.log('installed after', ((Date.now()-t0)/1000).toFixed(1), 's');
  await win.waitForTimeout(1200);
  await win.screenshot({ path: `${CAP}/04-accueil-vide.png` });
  console.log((await win.locator('body').innerText()).slice(0, 1000));
} finally {
  await app.close().catch(e => console.log('close err', e.message));
}
