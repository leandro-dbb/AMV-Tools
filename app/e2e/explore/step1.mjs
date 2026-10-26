// Exploration: first launch on an empty profile (onboarding screen).
import { makeProfile, launchApp, capture, closeApp } from '../helpers.mjs';
import fs from 'node:fs';
const CAP = process.env.AMV_E2E_CAPTURES;
const profile = makeProfile({});
console.log('profile', profile.dir);
const t0 = Date.now();
const { app, win, errors } = await launchApp(profile, { lang: 'fr' });
try {
  await win.waitForTimeout(1500);
  await capture(win, '01-demarrage');
  await win.getByText('CPU', { exact: false }).first().waitFor({ timeout: 180000 });
  console.log('onboarding after', (Date.now() - t0) / 1000);
  await win.waitForTimeout(500);
  await capture(win, '02-premier-lancement');
  console.log((await win.locator('body').innerText()).slice(0, 3000));
  await win.setViewportSize?.({ width: 1280, height: 720 });
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1280, 720));
  await win.waitForTimeout(500);
  await capture(win, '02b-premier-lancement-1280');
  console.log('errors', errors);
} finally { await closeApp(app); fs.writeFileSync('/tmp/amv-b-last-profile', profile.dir); }
