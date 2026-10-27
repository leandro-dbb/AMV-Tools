// Exploration (real extracts): create "Beerus" from a few example shots.
import { launchApp, capture, closeApp, waitForShell } from '../helpers.mjs';
const profile = { dir: process.argv[2], tutorialSeen: true };
const { app, win } = await launchApp(profile, { lang: 'fr' });
const body = async () => (await win.locator('body').innerText());
try {
  await waitForShell(win);
  await win.getByRole('button', { name: 'Personnages', exact: true }).first().click();
  await win.getByTitle('Nouveau personnage').click();
  await win.getByPlaceholder('ex. Gojo').fill('Beerus');
  await win.getByPlaceholder('ex. Gojo').press('Enter');
  await win.waitForTimeout(2000);
  await capture(win, '29-beerus-cree');
  const find = win.getByPlaceholder(/Décrivez le personnage/);
  await find.fill('purple cat god with big ears, egyptian clothes');
  await find.press('Enter');
  await win.getByTitle('Ajouter comme exemple').first().waitFor({ timeout: 150000 });
  await capture(win, '30-beerus-propositions');
  for (let i = 0; i < 3; i++) { await win.getByTitle('Ajouter comme exemple').first().click(); await win.waitForTimeout(2500); }
  await win.waitForTimeout(8000);
  await capture(win, '31-beerus-scenes-trouvees');
  console.log('chars:', (await body()).split('Personnages')[2]?.slice(0, 900).replace(/\n/g, ' | '));
} catch (e) { console.log('FAIL', String(e).split('\n')[0]); await capture(win, 'x-echec-beerus'); }
finally { await closeApp(app); }
