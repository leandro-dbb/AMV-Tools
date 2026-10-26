import { _electron as electron } from '@playwright/test';
const app = await electron.launch({
  cwd: '/home/leandro/dev/wt/amv-v06-B/app',
  args: ['.', '--user-data-dir=/tmp/opencode/amv-profile-smoke', '--no-sandbox'],
  env: { ...process.env, AMV_DATA_DIR: undefined },
});
console.log('userData', await app.evaluate(({ app }) => app.getPath('userData')));
const win = await app.firstWindow();
await win.waitForTimeout(3000);
console.log('title', await win.title());
console.log('text', (await win.locator('body').innerText()).slice(0, 500));
await win.screenshot({ path: '/tmp/opencode/e2e/smoke.png' });
await app.close();
