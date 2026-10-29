// Drag and drop of files onto every drop zone (Electron 32+ has no File.path:
// the preload's webUtils.getPathForFile must resolve them).
// Xvfb has no file manager to drag from, so the OS gesture itself is
// replaced: Playwright fills a hidden <input type=file> (real File objects
// backed by paths on disk, exactly like a native drop) and the test
// dispatches dragover + drop with them on the zone. The zone's handler, the
// bridge and the backend calls are the real ones.
import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { apiCall, closeApp, launchApp, makeProfile, makeTempDir, waitForShell } from './helpers.mjs';

test.describe.configure({ mode: 'serial' });

let ctx;
const dir = path.join(makeTempDir('amv-e2e-drop-'), 'Dossier déposé été');
fs.mkdirSync(dir, { recursive: true });
const video = path.join(dir, 'Épisode 01.mp4');
const song = path.join(dir, 'Musique libre.wav');
const image = path.join(dir, 'Référence.png');
execFileSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=24:duration=2',
  '-pix_fmt', 'yuv420p', '-c:v', 'libx264', '-preset', 'ultrafast', video]);
execFileSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=frequency=220:duration=8:sample_rate=22050',
  '-af', 'volume=0.5', song]);
execFileSync('ffmpeg', ['-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=320x180', '-frames:v', '1', image]);

// Drop `files` on the element holding `text` (the event bubbles to the zone).
async function drop(win, text, files) {
  await win.evaluate(() => {
    let i = document.getElementById('e2e-drop-files');
    if (!i) {
      i = document.createElement('input');
      i.type = 'file'; i.multiple = true; i.id = 'e2e-drop-files'; i.style.display = 'none';
      document.body.appendChild(i);
    }
  });
  await win.setInputFiles('#e2e-drop-files', files);
  await win.getByText(text).first().evaluate((el) => {
    const dt = new DataTransfer();
    for (const f of document.getElementById('e2e-drop-files').files) dt.items.add(f);
    for (const type of ['dragenter', 'dragover', 'drop']) {
      el.dispatchEvent(new DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true }));
    }
  });
}

test.beforeAll(async () => {
  const profile = makeProfile({ setupDone: true, tutorialSeen: true });
  ctx = { profile, ...(await launchApp(profile, { lang: 'fr' })) };
  await waitForShell(ctx.win);
});

test.afterAll(async () => {
  if (ctx?.errors.length) console.log('renderer errors:', ctx.errors);
  if (ctx) await closeApp(ctx.app);
});

test('bibliothèque : un fichier vidéo déposé part dans la file, avec son vrai chemin', async () => {
  const { win } = ctx;
  await win.getByRole('button', { name: 'Bibliothèque' }).first().click();
  await drop(win, "Glisse tes dossiers d'épisodes ici", [video]);
  await expect.poll(async () => (await apiCall(win, 'GET', '/api/index/queue')).json.items.map((q) => q.path))
    .toContain(video);
  await expect(win.getByText(video)).toBeVisible();
});

test('musique : un morceau déposé est analysé (chemin résolu, pas de message d’erreur)', async () => {
  const { win } = ctx;
  await win.getByRole('button', { name: 'Musique' }).first().click();
  await drop(win, 'Dépose un fichier musique ici', [song]);
  await expect(win.getByText(/Impossible de lire le chemin/)).toHaveCount(0);
  await expect.poll(async () => {
    const r = await apiCall(win, 'GET', '/api/music/tracks');
    return JSON.stringify(r.json);
  }, { timeout: 120_000 }).toContain('Musique libre');
});

test('hors zone : un fichier lâché ailleurs n’ouvre pas le fichier à la place de l’appli', async () => {
  const { win } = ctx;
  const before = win.url();
  await drop(win, 'Musique', [video]);
  await win.waitForTimeout(500);
  expect(win.url()).toBe(before);
  await expect(win.getByRole('button', { name: 'Bibliothèque' }).first()).toBeVisible();
});
