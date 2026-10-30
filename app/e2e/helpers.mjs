// Shared helpers for the Electron end-to-end suite. Everything runs
// against the *built* app (npm run build) and the real Python sidecar, with a throwaway profile:
// never the user's own userData, library or browser.
import { _electron as electron } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const APP_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// Screenshots of every step land here when AMV_E2E_CAPTURES is set
// (named "<step>.png"); otherwise in the Playwright output folder.
export const CAPTURE_DIR = process.env.AMV_E2E_CAPTURES
  || path.join(os.tmpdir(), 'amv-e2e-captures');

export function makeTempDir(prefix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

// A fresh userData folder. `setupDone` writes the marker the first-run
// screen sets after its `uv sync`, so a test can start directly in the app
// (the real install would reinstall torch into the dev venv).
export function makeProfile({ setupDone = false, tutorialSeen = false } = {}) {
  const dir = makeTempDir('amv-e2e-profile-');
  if (setupDone) fs.writeFileSync(path.join(dir, '.setup_complete'), 'cpu');
  return { dir, tutorialSeen };
}

// Media folder with a trap name (spaces + accents). AMV_E2E_MEDIA points at
// real anime extracts; without it, short synthetic clips with hard cuts are
// generated with ffmpeg (CI, or any machine without test footage).
export function prepareMedia() {
  const src = process.env.AMV_E2E_MEDIA;
  if (src) return { dir: src, synthetic: false, files: listVideos(src) };
  const root = makeTempDir('amv-e2e-media-');
  const dir = path.join(root, 'Séries test', 'Épisode spécial');
  fs.mkdirSync(dir, { recursive: true });
  const clips = [
    ['Clip été 01.mp4', ['testsrc2', 'smptebars', 'mandelbrot', 'rgbtestsrc']],
    ['Clip hiver 02.mp4', ['smptehdbars', 'testsrc', 'cellauto', 'yuvtestsrc']],
  ];
  for (const [name, sources] of clips) {
    const inputs = [];
    const labels = [];
    sources.forEach((s, i) => {
      inputs.push('-f', 'lavfi', '-t', '3', '-i', `${s}=size=640x360:rate=24`);
      labels.push(`[${i}:v]`);
    });
    execFileSync('ffmpeg', [
      '-v', 'error', '-y', ...inputs,
      '-f', 'lavfi', '-i', 'sine=frequency=440:duration=12',
      '-filter_complex', `${labels.join('')}concat=n=${sources.length}:v=1:a=0,format=yuv420p[v]`,
      '-map', '[v]', '-map', `${sources.length}:a`, '-c:v', 'libx264', '-preset', 'ultrafast',
      '-c:a', 'aac', '-shortest', path.join(dir, name),
    ]);
  }
  const files = listVideos(dir);
  // A corrupted episode (truncated download): indexing must say so, not
  // skip it silently. Not counted in `files` (it never gets indexed).
  fs.writeFileSync(path.join(dir, 'Épisode abîmé 03.mkv'), Buffer.alloc(4096, 7));
  return { dir, synthetic: true, files, corrupt: 'Épisode abîmé 03.mkv' };
}

function listVideos(dir) {
  return fs.readdirSync(dir).filter((f) => /\.(mkv|mp4|mov|avi|webm)$/i.test(f)).sort();
}

export async function launchApp(profile, { width = 1440, height = 900, lang } = {}) {
  const env = { ...process.env };
  delete env.AMV_DATA_DIR;
  delete env.AMV_USE_VITE;
  env.AMV_E2E = '1';
  // backend/api/setup.py: the first-run screen runs for real, minus the
  // `uv sync` that would reinstall torch into the dev venv.
  env.AMV_E2E_SKIP_SYNC = '1';
  const app = await electron.launch({
    cwd: APP_DIR,
    args: ['.', `--user-data-dir=${profile.dir}`, '--no-sandbox'],
    env,
  });
  const logs = [];
  app.process().stdout?.on('data', (d) => logs.push(String(d)));
  app.process().stderr?.on('data', (d) => logs.push(String(d)));
  const win = await app.firstWindow();
  const errors = [];
  win.on('pageerror', (e) => errors.push(e.message));
  await win.evaluate(({ seen, lang }) => {
    if (seen) localStorage.setItem('amv-tutorial-shown', '1');
    if (lang) localStorage.setItem('amv-lang', lang);
  }, { seen: profile.tutorialSeen, lang });
  // The language is read once at startup: reload so it applies (the main
  // process keeps the sidecar, the renderer resumes its bootstrap state).
  if (lang || profile.tutorialSeen) await win.reload();
  await app.evaluate(({ BrowserWindow }, size) => {
    BrowserWindow.getAllWindows()[0]?.setSize(size.width, size.height);
  }, { width, height });
  return { app, win, logs, errors };
}

// Replace the native folder/file pickers: Playwright cannot click OS dialogs.
export async function stubDialogs(app, { open = [], save = null } = {}) {
  await app.evaluate(({ dialog, shell }, r) => {
    dialog.showOpenDialog = async () => ({ canceled: r.open.length === 0, filePaths: r.open });
    dialog.showSaveDialog = async () => ({ canceled: !r.save, filePath: r.save ?? undefined });
    // "Show in folder" after an export would open a file manager.
    shell.showItemInFolder = () => {};
    shell.openPath = async () => '';
  }, { open, save });
}

export async function capture(win, name) {
  fs.mkdirSync(CAPTURE_DIR, { recursive: true });
  await win.screenshot({ path: path.join(CAPTURE_DIR, `${name}.png`) });
}

// Waits until the sidecar answered and the main shell (tabs) is visible.
export async function waitForShell(win, timeout = 180_000) {
  await win.getByRole('button', { name: /^(Bibliothèque|Library)$/ }).first().waitFor({ timeout });
}

export async function apiCall(win, method, route, body) {
  return win.evaluate(async ({ method, route, body }) => {
    const port = window.__AMV_API_PORT__ || await window.amvBridge.getApiPort();
    const r = await fetch(`http://127.0.0.1:${port}${route}`, {
      method,
      headers: { 'Content-Type': 'application/json', 'X-AMV-Token': window.amvBridge.apiToken },
      body: body ? JSON.stringify(body) : undefined,
    });
    return { status: r.status, json: await r.json().catch(() => null) };
  }, { method, route, body });
}

export async function closeApp(app) {
  await Promise.race([
    app.close().catch(() => {}),
    new Promise((r) => setTimeout(r, 15_000)),
  ]);
}
