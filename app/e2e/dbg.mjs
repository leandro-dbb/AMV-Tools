import { _electron as electron } from '@playwright/test';
const app = await electron.launch({
  cwd: '/home/leandro/dev/wt/amv-v06-B/app',
  args: ['.', '--user-data-dir=/tmp/opencode/amv-profile-B', '--no-sandbox'],
});
app.process().stderr?.on('data', d => console.log('[stderr]', d.toString().slice(0,500)));
const win = await app.firstWindow();
win.on('console', m => console.log('[console]', m.type(), m.text()));
win.on('pageerror', e => console.log('[pageerror]', e.message));
console.log('url', win.url());
await win.waitForTimeout(4000);
console.log('body', (await win.locator('body').innerText().catch(e=>'ERR '+e)).slice(0,400));
await win.screenshot({ path: '/tmp/opencode/dbg.png' });
await app.close();
