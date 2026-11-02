// Remote driver for the INSTALLED AMV Tools app on a real Windows machine.
//
// Runs inside the interactive (autologon) session — Electron needs a
// desktop, an SSH session lives in session 0 — and executes step files
// dropped into <root>\cmd over SSH: each `<n>-<name>.mjs` exports
// `default async (ctx) => result`. The result (or the error) lands in
// <root>\out\<n>-<name>.json, screenshots in <root>\shots. One app instance
// is kept between steps, so a long manual test can be driven one step at a
// time and adjusted as it goes. Needs `playwright-core` next to it.
//
//   node driver.mjs [root]     (root defaults to C:\amv\driver)
import { _electron as electron } from 'playwright-core';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const ROOT = process.argv[2] || 'C:\\amv\\driver';
const CMD = path.join(ROOT, 'cmd');
const OUT = path.join(ROOT, 'out');
const SHOTS = path.join(ROOT, 'shots');
for (const d of [CMD, OUT, SHOTS]) fs.mkdirSync(d, { recursive: true });

const DEFAULT_EXE = path.join(process.env.LOCALAPPDATA || os.homedir(), 'Programs', 'amv-tools', 'AMV Tools.exe');

function log(...args) {
  fs.appendFileSync(path.join(ROOT, 'driver.log'), `${new Date().toISOString()} ${args.join(' ')}\n`);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const ctx = {
  app: null,
  win: null,
  errors: [],
  ROOT,
  SHOTS,
  log,
  sleep,
  async launch({ exe = process.env.AMV_EXE || DEFAULT_EXE, args = [], env = {} } = {}) {
    const out = fs.createWriteStream(path.join(ROOT, 'electron.log'), { flags: 'a' });
    const t0 = Date.now();
    ctx.app = await electron.launch({ executablePath: exe, args, env: { ...process.env, ...env }, timeout: 120_000 });
    ctx.app.process().stdout?.on('data', (d) => out.write(d));
    ctx.app.process().stderr?.on('data', (d) => out.write(d));
    ctx.win = await ctx.app.firstWindow({ timeout: 120_000 });
    ctx.win.on('pageerror', (e) => { ctx.errors.push(e.message); log('pageerror', e.message); });
    log('launched', exe, `${Date.now() - t0} ms`);
    return { pid: ctx.app.process().pid, ms: Date.now() - t0 };
  },
  async shot(name) {
    const file = path.join(SHOTS, `${name}.png`);
    await ctx.win.screenshot({ path: file });
    return file;
  },
  async api(method, route, body) {
    return ctx.win.evaluate(async ({ method, route, body }) => {
      const port = window.__AMV_API_PORT__ || await window.amvBridge.getApiPort();
      const r = await fetch(`http://127.0.0.1:${port}${route}`, {
        method,
        headers: { 'Content-Type': 'application/json', 'X-AMV-Token': window.amvBridge.apiToken },
        body: body ? JSON.stringify(body) : undefined,
      });
      return { status: r.status, json: await r.json().catch(() => null) };
    }, { method, route, body });
  },
  // Replace the native pickers: Playwright cannot click OS dialogs.
  async stubDialogs({ open = [], save = null } = {}) {
    await ctx.app.evaluate(({ dialog, shell }, r) => {
      dialog.showOpenDialog = async () => ({ canceled: r.open.length === 0, filePaths: r.open });
      dialog.showSaveDialog = async () => ({ canceled: !r.save, filePath: r.save ?? undefined });
      shell.showItemInFolder = () => {};
      shell.openPath = async () => '';
    }, { open, save });
  },
  async close(timeout = 30_000) {
    if (!ctx.app) return 'not running';
    const t0 = Date.now();
    await Promise.race([ctx.app.close().catch(() => {}), sleep(timeout)]);
    ctx.app = null;
    ctx.win = null;
    return `${Date.now() - t0} ms`;
  },
};

log('driver up', ROOT);
fs.writeFileSync(path.join(ROOT, 'driver.pid'), String(process.pid));
let stop = false;
while (!stop) {
  const steps = fs.readdirSync(CMD).filter((f) => f.endsWith('.mjs')).sort();
  for (const f of steps) {
    const src = path.join(CMD, f);
    const name = f.replace(/\.mjs$/, '');
    const running = path.join(OUT, `${name}.running`);
    fs.writeFileSync(running, new Date().toISOString());
    const t0 = Date.now();
    let res;
    try {
      const mod = await import(`${pathToFileURL(src).href}?t=${Date.now()}`);
      const result = await mod.default(ctx);
      res = { ok: true, ms: Date.now() - t0, result: result ?? null };
      if (result === 'STOP') stop = true;
    } catch (e) {
      res = { ok: false, ms: Date.now() - t0, error: String(e?.stack || e) };
      if (ctx.win) await ctx.shot(`${name}-ECHEC`).catch(() => {});
    }
    res.pageErrors = ctx.errors.splice(0);
    fs.writeFileSync(path.join(OUT, `${name}.json`), JSON.stringify(res, null, 2));
    fs.renameSync(src, path.join(OUT, f));
    fs.rmSync(running, { force: true });
    log(name, res.ok ? 'ok' : 'FAILED', `${res.ms} ms`);
    if (stop) break;
  }
  await sleep(500);
}
await ctx.close();
log('driver down');
process.exit(0);
