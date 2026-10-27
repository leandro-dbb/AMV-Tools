// Electron end-to-end suite: `npm run e2e` (builds nothing; run
// `npm run build` first). Runs headless under Xvfb via e2e/run.mjs.
import { defineConfig } from '@playwright/test';
import os from 'node:os';
import path from 'node:path';

export default defineConfig({
  testDir: '.',
  testMatch: /.*\.e2e\.mjs$/,
  // One Electron app + one sidecar at a time: the steps share a library.
  workers: 1,
  fullyParallel: false,
  retries: 0,
  // Real indexing with the real models on CPU dominates the run time; real
  // episodes (AMV_E2E_MEDIA) take tens of minutes on a CPU.
  timeout: (process.env.AMV_E2E_MEDIA ? 50 : 8) * 60_000,
  globalTimeout: (process.env.AMV_E2E_MEDIA ? 70 : 10) * 60_000,
  expect: { timeout: 20_000 },
  reporter: [['list']],
  outputDir: path.join(os.tmpdir(), 'amv-e2e-results'),
});
