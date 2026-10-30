// `npm run e2e`: runs the Electron suite, under its own Xvfb server when
// there is no display (CI, SSH). Never attaches to an existing browser.
//   AMV_E2E_MEDIA=<folder>     real episodes/extracts instead of synthetic clips
//   AMV_E2E_CAPTURES=<folder>  keep one PNG per step there
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { APP_DIR } from './helpers.mjs';

if (!fs.existsSync(path.join(APP_DIR, 'dist', 'index.html')) || !fs.existsSync(path.join(APP_DIR, 'dist-electron', 'main.js'))) {
  console.error('e2e: build the app first (npm run build).');
  process.exit(2);
}

const pw = path.join(APP_DIR, 'node_modules', '@playwright', 'test', 'cli.js');
const args = [pw, 'test', '-c', path.join('e2e', 'playwright.config.mjs'), ...process.argv.slice(2)];
const needXvfb = process.platform === 'linux' && !process.env.DISPLAY;
const cmd = needXvfb ? 'xvfb-run' : process.execPath;
const full = needXvfb ? ['-a', '-s', '-screen 0 2560x1440x24', process.execPath, ...args] : args;
const r = spawnSync(cmd, full, { cwd: APP_DIR, stdio: 'inherit' });
process.exit(r.status ?? 1);
