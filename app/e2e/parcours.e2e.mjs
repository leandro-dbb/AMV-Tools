// The main journey of an AMV editor, steps 1 → 4, against the built Electron
// app, the real sidecar and the real models (CPU): first launch, library +
// indexing, search, derush basket + clip / timeline export.
import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import {
  apiCall, capture, closeApp, launchApp, makeProfile, makeTempDir, prepareMedia, stubDialogs, waitForShell,
} from './helpers.mjs';

test.describe.configure({ mode: 'serial' });

let ctx;
const media = prepareMedia();
const exportDir = path.join(makeTempDir('amv-e2e-out-'), 'Exports AMV été');

test.beforeAll(async () => {
  fs.mkdirSync(exportDir, { recursive: true });
  const profile = makeProfile({});
  // 1280×720: the small-window case, where layouts break first.
  ctx = { profile, ...(await launchApp(profile, { width: 1280, height: 720, lang: 'fr' })) };
  await stubDialogs(ctx.app, { open: [media.dir] });
});

test.afterEach(async ({}, info) => {
  if (info.status !== info.expectedStatus && ctx?.win) await capture(ctx.win, `echec-${info.title.slice(0, 2).trim()}`).catch(() => {});
});

test.afterAll(async () => {
  if (ctx?.errors.length) console.log('renderer errors:', ctx.errors);
  if (ctx) await closeApp(ctx.app);
});

test('1. premier lancement : écran d’accueil lisible, choix CPU, tutoriel', async () => {
  const { win } = ctx;
  const title = win.getByRole('heading', { name: 'Bienvenue dans AMV Tools' });
  await expect(title).toBeVisible({ timeout: 180_000 });
  // Regression: a vertically centred, overflowing page cut its own top off.
  expect((await title.boundingBox()).y).toBeGreaterThanOrEqual(0);
  await win.getByRole('button', { name: /CPU uniquement/ }).click();
  // Without a GPU the fast profile is pre-selected and explained.
  await expect(win.getByText(/Sans GPU, « Rapide » est recommandé/)).toBeVisible();
  await capture(win, 'e2e-01-premier-lancement');
  await win.getByRole('button', { name: 'Installer et continuer' }).click();
  await waitForShell(win);
  await expect(win.getByText('1. Ajoutez votre bibliothèque')).toBeVisible();
  await capture(win, 'e2e-01-tutoriel');
  await win.getByRole('button', { name: 'Passer' }).click();
  // Empty library: the app lands on the Library tab, no false GPU warning.
  await expect(win.getByText('Ta bibliothèque est vide')).toBeVisible();
  await expect(win.getByText(/torch is not installed/)).toHaveCount(0);
  const settings = await apiCall(win, 'GET', '/api/settings');
  expect(settings.json.indexing.device).toBe('cpu');
  expect(settings.json.indexing.sub_segmentation).toBe(false);
});

test('2. bibliothèque : ajout d’un dossier piège, indexation suivie jusqu’au bout', async () => {
  const { win } = ctx;
  await apiCall(win, 'PUT', '/api/settings', { export: { output_folder: exportDir } });
  await win.getByRole('button', { name: 'Ajouter un dossier' }).click();
  await expect(win.getByText(media.dir)).toBeVisible();
  // A double click must not start two runs (the second is ignored).
  await win.getByRole('button', { name: /Lancer l'indexation/ }).dblclick();
  await expect(win.getByText('Indexation en cours')).toBeVisible();
  await expect(win.getByRole('button', { name: /Lancer l'indexation/ })).toBeDisabled();
  await expect(win.getByTestId('index-stage')).toBeVisible();
  await capture(win, 'e2e-02-indexation');
  await expect(win.getByText('Inactif')).toBeVisible({ timeout: 7 * 60_000 });
  await expect(win.getByText(`${media.files.length} épisodes`, { exact: false })).toBeVisible();
  if (media.corrupt) {
    // The broken file is reported by name, in plain words, after the run.
    const alert = win.getByRole('alert');
    await expect(alert).toContainText("1 fichier(s) n'ont pas pu être indexés");
    await expect(alert).toContainText(media.corrupt);
  } else {
    await expect(win.getByRole('alert')).toHaveCount(0);
  }
  await capture(win, 'e2e-02-indexation-fin');
});

test('3. recherche : texte FR / EN, aperçu au survol, filtres, plans similaires', async () => {
  const { win } = ctx;
  await win.getByRole('button', { name: 'Recherche', exact: true }).first().click();
  // Regression: an indexed library used to greet with "index a folder first".
  await expect(win.getByTestId('search-ready')).toBeVisible();
  await expect(win.getByText("Indexez un dossier d'animes pour commencer la recherche.")).toHaveCount(0);
  const box = win.getByPlaceholder(/./).first();
  // Synthetic test patterns match no anime prompt well: search the whole
  // library (threshold 0) so the UI path is what gets tested.
  if (media.synthetic) await win.locator('input[type=range]').first().fill('0');
  for (const q of ['explosion', 'gros plan visage']) {
    await box.fill(q);
    await box.press('Enter');
    await expect(win.locator('div.group.cursor-pointer img').first()).toBeVisible({ timeout: 90_000 });
  }
  await capture(win, 'e2e-03-recherche');
  // The threshold slider re-runs the search by itself (used to need Enter).
  const slider = win.locator('input[type=range]').first();
  await slider.fill('100');
  await expect(win.locator('div.group.cursor-pointer')).toHaveCount(0);
  await slider.fill(media.synthetic ? '0' : '10');
  await expect(win.locator('div.group.cursor-pointer img').first()).toBeVisible({ timeout: 90_000 });
  const card = win.locator('div.group.cursor-pointer').first();
  await card.hover();
  if (media.synthetic) {
    // H.264 sources play in Chromium: the hover preview must run without proxies.
    await expect(win.getByTestId('hover-preview')).toBeVisible();
  }
  await capture(win, 'e2e-03-survol');
  await win.getByRole('button', { name: /Filtres/ }).click();
  await expect(win.getByText(/Mouvement|Intensité/).first()).toBeVisible();
  await capture(win, 'e2e-03-filtres');
  await win.getByRole('button', { name: /Filtres/ }).click();
  await win.mouse.move(5, 5);
  await card.hover();
  await card.getByRole('button').first().click();
  await expect(win.getByText(/similaires/i).first()).toBeVisible();
  await capture(win, 'e2e-03-similaires');
});

test('4. éditeur, panier de dérush, export de clips et de timelines', async () => {
  const { win, app } = ctx;
  // Open a scene, export the trimmed clip into the (accented) export folder.
  await win.locator('div.group.cursor-pointer').first().click();
  // Regression: at 1280×720 the player pushed the transport bar off-screen.
  await expect(win.getByRole('button', { name: /^(Lecture|Pause) \(Espace/ })).toBeInViewport({ ratio: 1 });
  // Regression: the Finishing menu opened under the player (only "Upscale"
  // was reachable). A trial click fails if something covers the entry.
  await win.getByRole('button', { name: /^Finition/ }).first().click();
  await win.getByRole('button', { name: /^Profondeur/ }).click({ trial: true });
  await win.getByRole('button', { name: /^Finition/ }).first().click();
  await win.getByRole('button', { name: /^Exporter .* s$/ }).click();
  await expect(win.getByText(/Exporté :/)).toBeVisible({ timeout: 120_000 });
  await capture(win, 'e2e-04-editeur-export');
  // Escape closes the editor (it used to need the small × button).
  await win.keyboard.press('Escape');
  await expect(win.getByRole('button', { name: /^Exporter .* s$/ })).toHaveCount(0);
  expect(fs.readdirSync(exportDir).some((f) => /\.(mp4|mov|mkv)$/.test(f))).toBe(true);

  // Derush: keep two scenes of the first episode, they land in the basket.
  await win.getByRole('button', { name: 'Dérushage', exact: true }).first().click();
  await win.getByText(media.files[0].replace(/\.[^.]+$/, ''), { exact: false }).first().click();
  const keep = win.getByRole('button', { name: /Garder la scène/ });
  await expect(keep).toBeVisible({ timeout: 60_000 });
  await keep.click();
  await win.keyboard.press('ArrowRight');
  await keep.click();
  await capture(win, 'e2e-04-derush');
  await win.getByRole('button', { name: /^Bibliothèque \(\d+\)$/ }).click();
  await expect(win.getByRole('button', { name: /Tout exporter \(2\)/ })).toBeVisible();
  await win.getByRole('button', { name: /Tout exporter \(2\)/ }).click();
  await expect(win.getByText('Export terminé : 2 clip(s)')).toBeVisible({ timeout: 180_000 });
  await capture(win, 'e2e-04-derush-export');

  // Timeline export: every format offered, each one written to disk.
  await win.getByRole('button', { name: /timeline/i }).first().click();
  const select = win.locator('select').first();
  const formats = await select.locator('option').evaluateAll((os) => os.map((o) => o.value));
  expect(formats.length).toBeGreaterThan(1);
  for (const f of formats) {
    await select.selectOption(f);
    const target = path.join(exportDir, `timeline-${f}`);
    await stubDialogs(app, { open: [media.dir], save: target });
    await win.getByRole('button', { name: 'Exporter', exact: true }).click();
    await expect.poll(() => fs.readdirSync(exportDir).some((n) => n.startsWith(`timeline-${f}`)), { timeout: 60_000 }).toBe(true);
    if (f === formats[0]) await capture(win, 'e2e-04-timeline');
  }
  // Once exported, the dialog offers "Fermer", not "Annuler".
  await win.getByRole('button', { name: 'Fermer', exact: true }).click();
  await expect(select).toHaveCount(0);
});

test('6. paramètres : dossier d’export refusé, langue FR ⇄ EN', async () => {
  const { win, app } = ctx;
  const locked = path.join(makeTempDir('amv-e2e-ro-'), 'Lecture seule');
  fs.mkdirSync(locked);
  fs.chmodSync(locked, 0o500);
  try {
    await stubDialogs(app, { open: [locked] });
    await win.getByRole('button', { name: 'Paramètres', exact: true }).first().click();
    await win.locator('div.w-64').getByRole('button', { name: 'Export' }).click();
    await win.getByRole('button', { name: /Exports AMV été/ }).click();
    const alert = win.getByRole('alert');
    // Refused in plain words, visible without scrolling back up, old value kept.
    await expect(alert).toContainText('ne peut pas écrire');
    await expect(alert).toBeInViewport();
    await expect(win.getByRole('button', { name: /Exports AMV été/ })).toBeVisible();
    await capture(win, 'e2e-06-dossier-refuse');
  } finally {
    fs.chmodSync(locked, 0o700);
    await stubDialogs(app, { open: [media.dir] });
  }
  await win.locator('div.w-64').getByRole('button', { name: 'Interface' }).click();
  await win.locator('select').filter({ hasText: 'Français' }).first().selectOption('en');
  await expect(win.getByRole('button', { name: 'Library', exact: true }).first()).toBeVisible();
  await capture(win, 'e2e-06-anglais');
  await win.locator('select').filter({ hasText: 'Français' }).first().selectOption('fr');
  await expect(win.getByRole('button', { name: 'Bibliothèque', exact: true }).first()).toBeVisible();
});

test('7. cas d’erreur : fichier source supprimé après indexation', async () => {
  test.skip(!media.synthetic, 'never delete real footage');
  const { win } = ctx;
  const victim = path.join(media.dir, media.files[1]);
  fs.renameSync(victim, `${victim}.moved`);
  try {
    await win.getByRole('button', { name: 'Dérushage', exact: true }).first().click();
    await win.getByRole('button', { name: /^Lecteur$/ }).click().catch(() => {});
    await win.getByText(media.files[1].replace(/\.[^.]+$/, ''), { exact: false }).first().click();
    await expect(win.getByText('Fichier source introuvable')).toBeVisible({ timeout: 60_000 });
    await capture(win, 'e2e-07-source-supprimee');
  } finally {
    fs.renameSync(`${victim}.moved`, victim);
  }
});

test('8. reprise : une file laissée en attente est signalée au relancement', async () => {
  const { win, app, profile } = ctx;
  await apiCall(win, 'POST', '/api/index/queue', { paths: [{ path: media.dir, recursive: true }] });
  await closeApp(app);
  Object.assign(ctx, await launchApp({ ...profile, tutorialSeen: true }, { width: 1280, height: 720, lang: 'fr' }));
  await waitForShell(ctx.win);
  await ctx.win.getByRole('button', { name: 'Bibliothèque', exact: true }).first().click();
  await expect(ctx.win.getByTestId('queue-leftover')).toBeVisible();
  await capture(ctx.win, 'e2e-08-file-en-attente-au-relancement');
  // Starting it clears the note; finished episodes are skipped (fast).
  await ctx.win.getByRole('button', { name: /Lancer l'indexation/ }).click();
  await expect(ctx.win.getByTestId('queue-leftover')).toHaveCount(0);
  await expect(ctx.win.getByText('Inactif')).toBeVisible({ timeout: 3 * 60_000 });
});
