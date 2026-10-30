import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron';
import { spawn, ChildProcess } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import net from 'node:net';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import fs from 'node:fs';

const isDev = !app.isPackaged;
const distIndex = path.join(__dirname, '..', 'dist', 'index.html');
const useVite = isDev && (process.env.AMV_USE_VITE === '1' || !fs.existsSync(distIndex));
const smokeExitMs = Number(process.env.AMV_SMOKE_EXIT_MS || 0);
let pyProc: ChildProcess | null = null;
let mainWindow: BrowserWindow | null = null;
let apiPort = 0;
// How long a quitting app waits for the sidecar's own shutdown (about 1 s,
// see backend/shutdown.py) before it kills the process tree.
const SIDECAR_STOP_GRACE_MS = 3000;
let sidecarStopping: Promise<void> | null = null;
let sidecarStopped = false;
// Per-launch secret: the sidecar refuses requests without it (backend/security.py),
// so other local web pages cannot drive it. Renderer gets it via the preload.
// AMV_TOKEN is a test-only override: the Windows smoke workflow launches the
// installed app with a known token so it can drive the sidecar from PowerShell.
// A short/guessable value is ignored so a stray env var can't weaken it.
const forcedToken = process.env.AMV_TOKEN ?? '';
const apiToken = forcedToken.length >= 32 ? forcedToken : randomBytes(32).toString('hex');

type BootstrapStream = 'stdout' | 'stderr';
type BootstrapEvent =
  | { kind: 'line'; line: string; stream: BootstrapStream }
  | { kind: 'ready'; port: number }
  | { kind: 'error'; message: string };

const bootstrapBuffer: BootstrapEvent[] = [];
let rendererReady = false;
let bootstrapPhase: 'starting' | 'ready' | 'error' = 'starting';

function emitBootstrap(evt: BootstrapEvent) {
  if (evt.kind === 'ready') bootstrapPhase = 'ready';
  if (evt.kind === 'error') bootstrapPhase = 'error';
  if (rendererReady && mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(
      evt.kind === 'line' ? 'amv:bootstrap' : evt.kind === 'ready' ? 'amv:bootstrap-ready' : 'amv:bootstrap-error',
      evt.kind === 'line' ? { line: evt.line, stream: evt.stream } : evt.kind === 'ready' ? evt.port : evt.message,
    );
  } else {
    bootstrapBuffer.push(evt);
  }
}

function flushBootstrap() {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  for (const evt of bootstrapBuffer) {
    mainWindow.webContents.send(
      evt.kind === 'line' ? 'amv:bootstrap' : evt.kind === 'ready' ? 'amv:bootstrap-ready' : 'amv:bootstrap-error',
      evt.kind === 'line' ? { line: evt.line, stream: evt.stream } : evt.kind === 'ready' ? evt.port : evt.message,
    );
  }
  bootstrapBuffer.length = 0;
}

function findFreePort(): Promise<number> {
  // Test-only override so a headless smoke run knows where to find /health
  // (the real app always asks the OS for a free port).
  const forced = Number(process.env.AMV_SIDECAR_PORT || 0);
  if (Number.isInteger(forced) && forced > 0 && forced < 65536) {
    return Promise.resolve(forced);
  }
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      if (typeof addr === 'object' && addr) {
        const port = addr.port;
        server.close(() => resolve(port));
      } else {
        reject(new Error('Could not allocate port'));
      }
    });
  });
}

function resolvePythonRoot(): { uvExe: string; projectRoot: string } {
  if (isDev) {
    const projectRoot = path.resolve(__dirname, '..', '..');
    const uvExe = process.platform === 'win32'
      ? path.join(projectRoot, '.uv', 'uv.exe')
      : path.join(projectRoot, '.uv', 'uv');
    return { uvExe, projectRoot };
  }
  const resources = process.resourcesPath;
  const uvExe = process.platform === 'win32'
    ? path.join(resources, '.uv', 'uv.exe')
    : path.join(resources, '.uv', 'uv');
  return { uvExe, projectRoot: resources };
}

type SidecarExitError = Error & { moduleNotFound?: boolean; usedNoSync?: boolean };

// The backend chosen at first run ("cu130", "cpu", …): a sync must keep its
// extra, or uv swaps the CUDA torch build for the CPU one.
function installedBackend(): string | null {
  try {
    const b = fs.readFileSync(path.join(app.getPath('userData'), '.setup_complete'), 'utf8').trim();
    return /^[a-z0-9-]+$/.test(b) ? b : null;
  } catch {
    return null;
  }
}

// uv.lock of the running app vs the one the venv was last synced to: an
// update that adds a dependency (librosa for the Music tab in v06) must
// install it once, since later launches use --no-sync.
function lockStamp(): { current: string | null; file: string } {
  const file = path.join(app.getPath('userData'), '.deps-lock-sha1');
  try {
    const { projectRoot } = resolvePythonRoot();
    const lock = fs.readFileSync(path.join(projectRoot, 'uv.lock'));
    return { current: createHash('sha1').update(lock).digest('hex'), file };
  } catch {
    return { current: null, file };
  }
}

async function startSidecar(): Promise<number> {
  const stamp = lockStamp();
  const synced = (() => { try { return fs.readFileSync(stamp.file, 'utf8').trim(); } catch { return ''; } })();
  const depsChanged = !isDev && !!stamp.current && synced !== stamp.current
    && !!installedBackend() && fs.existsSync(venvPath());
  if (depsChanged) {
    const msg = '[main] the app was updated — installing new dependencies once';
    console.log(msg);
    emitBootstrap({ kind: 'line', line: msg, stream: 'stdout' });
    try {
      const port = await launchSidecar(true);
      try { fs.writeFileSync(stamp.file, stamp.current!); } catch {}
      return port;
    } catch (err) {
      // offline, index down…: start as before, a later launch retries
      console.warn('[main] dependency update failed, starting without it:', err);
      await killSidecarTree();
    }
  }
  try {
    const port = await launchSidecar(false);
    if (!synced && stamp.current && installedBackend()) {
      try { fs.writeFileSync(stamp.file, stamp.current); } catch {}
    }
    return port;
  } catch (err) {
    // uv creates the .venv directory as soon as the very first sync *starts*,
    // so an interrupted first install (network drop, app closed mid-download)
    // leaves a venv that exists but is missing base deps like uvicorn. Later
    // launches then pass --no-sync and die on ModuleNotFoundError forever.
    // Repair once with a full sync instead of staying broken.
    const e = err as SidecarExitError;
    if (e.usedNoSync && e.moduleNotFound) {
      const msg = '[main] venv is incomplete (ModuleNotFoundError) — repairing with a full dependency sync';
      console.warn(msg);
      emitBootstrap({ kind: 'line', line: msg, stream: 'stderr' });
      await killSidecarTree();
      return launchSidecar(true);
    }
    throw err;
  }
}

async function launchSidecar(forceSync: boolean): Promise<number> {
  apiPort = await findFreePort();
  const { uvExe, projectRoot } = resolvePythonRoot();
  const serverPath = path.join(projectRoot, 'backend', 'server.py');

  if (!fs.existsSync(serverPath)) {
    throw new Error(`Backend not found at ${serverPath}`);
  }

  const useUv = fs.existsSync(uvExe);
  // No uv next to the app (dev checkout without .uv/): the project's venv
  // python, not the system one (which has none of the backend's modules).
  const venvPython = process.platform === 'win32'
    ? path.join(venvPath(), 'Scripts', 'python.exe')
    : path.join(venvPath(), 'bin', 'python');
  const cmd = useUv ? uvExe
    : fs.existsSync(venvPython) ? venvPython
    : (process.platform === 'win32' ? 'python' : 'python3');
  // `uv run` re-syncs the environment to the project's *default* deps on every
  // launch. The GPU torch wheels live in optional extras (cu130, cu126, …), so
  // a plain re-sync silently uninstalls the CUDA build and reinstalls +cpu —
  // which is exactly what drops a working GPU install back to "CPU (forced
  // fallback)" on the next start. Once the venv exists (the in-app installer or
  // a dev `uv sync --extra <backend>` built it), launch with --no-sync so the
  // chosen backend sticks. Only the very first run (no venv yet) syncs, to
  // bootstrap the base deps that serve the setup UI — and startSidecar forces
  // one repair sync when a --no-sync launch dies on a missing base module.
  const noSync = useUv && !forceSync && fs.existsSync(venvPath()) ? ['--no-sync'] : [];
  // Installed app: a (repair) sync must not pull the dev group (pytest, httpx…).
  const noDev = useUv && !isDev && noSync.length === 0 ? ['--no-dev'] : [];
  // A (repair / update) sync keeps the chosen backend's extra: without it uv
  // reinstalled the CPU torch over a working CUDA install.
  const backend = installedBackend();
  const extra = useUv && noSync.length === 0 && backend
    ? ['--extra', backend, ...(backend === 'dml' ? ['--python', '3.12'] : [])] : [];
  const args = useUv
    ? ['run', ...noSync, ...noDev, ...extra, '--project', projectRoot, 'python', serverPath, '--port', String(apiPort)]
    : [serverPath, '--port', String(apiPort)];
  const userDataPath = app.getPath('userData');
  const bundledUvRoot = path.dirname(uvExe);
  const runtimeUvRoot = isDev ? path.join(projectRoot, '.uv') : path.join(userDataPath, '.uv');

  console.log(`[main] starting sidecar: ${cmd} ${args.join(' ')}`);
  emitBootstrap({ kind: 'line', line: `$ ${cmd} ${args.join(' ')}`, stream: 'stdout' });

  pyProc = spawn(cmd, args, {
    cwd: projectRoot,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      // Windows pipes default to the ANSI code page (cp1252): accented
      // paths and '—' in the sidecar's log lines reached the bootstrap
      // screen as '�'. Electron decodes the pipe as UTF-8.
      PYTHONIOENCODING: 'utf-8',
      // macOS: PyTorch's MPS backend is missing some ops; without this flag an
      // unsupported op raises instead of falling back to CPU. No-op elsewhere.
      PYTORCH_ENABLE_MPS_FALLBACK: '1',
      AMV_DATA_DIR: userDataPath,
      AMV_TOKEN: apiToken,
      UV_CACHE_DIR: isDev ? path.join(runtimeUvRoot, 'uv_cache') : path.join(userDataPath, '.uv_cache'),
      UV_PYTHON_INSTALL_DIR: path.join(runtimeUvRoot, 'python'),
      ...(isDev ? {} : { UV_PROJECT_ENVIRONMENT: path.join(userDataPath, '.venv') }),
      PATH: `${bundledUvRoot}${path.delimiter}${runtimeUvRoot}${path.delimiter}${process.env.PATH ?? ''}`,
    },
    stdio: ['ignore', 'pipe', 'pipe'],
    // POSIX: own process group (uv, python, ffmpeg…), so killSidecarGroup
    // reaches all of it. Not on Windows, where detached opens a console.
    detached: process.platform !== 'win32',
  });

  let lastActivity = Date.now();
  let sawModuleNotFound = false;
  const bump = () => { lastActivity = Date.now(); };
  const emitLines = (chunk: Buffer, stream: BootstrapStream) => {
    bump();
    const text = chunk.toString();
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trimEnd();
      if (!line) continue;
      if (line.includes('ModuleNotFoundError')) sawModuleNotFound = true;
      if (stream === 'stdout') console.log(`[py] ${line}`);
      else console.error(`[py-err] ${line}`);
      if (line.includes('progress:')) {
        try {
          const json = JSON.parse(line.split('progress:')[1].trim());
          mainWindow?.webContents.send('amv:progress', json);
        } catch {}
      }
      emitBootstrap({ kind: 'line', line, stream });
    }
  };

  pyProc.stdout?.on('data', (b: Buffer) => emitLines(b, 'stdout'));
  pyProc.stderr?.on('data', (b: Buffer) => emitLines(b, 'stderr'));

  // Reject promptly if the process dies before the port opens (a crash used to
  // sit out the 60s idle timeout before surfacing). Once the race is settled,
  // later exits (normal quit) reject into an already-settled race: no-ops.
  let rejectOnExit: ((err: Error) => void) | null = null;
  const exited = new Promise<never>((_, reject) => { rejectOnExit = reject; });
  pyProc.on('exit', (code) => {
    console.log(`[main] sidecar exited with code ${code}`);
    const err: SidecarExitError = new Error(`Sidecar exited with code ${code} before opening port ${apiPort}`);
    err.moduleNotFound = sawModuleNotFound;
    err.usedNoSync = noSync.length > 0;
    rejectOnExit?.(err);
  });

  // uv is completely silent while it downloads a single big wheel (piped
  // stdio: no progress bars), so on a slow connection minutes-long gaps
  // between log lines are normal during an install. Crashes are surfaced
  // instantly by the exit race above, so the timeouts only need to catch a
  // genuinely hung process — be generous when a sync is running, stricter on
  // a regular --no-sync boot where the port should open within seconds.
  const syncing = useUv && noSync.length === 0;
  const idleMs = syncing ? 10 * 60_000 : 2 * 60_000;
  const hardMs = syncing ? 60 * 60_000 : 5 * 60_000;
  await Promise.race([
    exited,
    waitUntilReady(apiPort, () => Date.now() - lastActivity, idleMs, hardMs),
  ]);
  return apiPort;
}

function waitUntilReady(
  port: number,
  getIdleMs: () => number,
  idleTimeoutMs: number,
  hardTimeoutMs: number,
): Promise<void> {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = net.connect(port, '127.0.0.1', () => {
        req.end();
        resolve();
      });
      req.on('error', () => {
        const elapsed = Date.now() - start;
        if (elapsed > hardTimeoutMs) {
          reject(new Error(`Backend did not start within ${hardTimeoutMs}ms (hard ceiling)`));
        } else if (getIdleMs() > idleTimeoutMs) {
          reject(new Error(`Backend went silent for >${idleTimeoutMs}ms without opening port ${port}`));
        } else {
          setTimeout(tick, 250);
        }
      });
    };
    tick();
  });
}

function venvPath(): string {
  const { projectRoot } = resolvePythonRoot();
  if (isDev) return path.join(projectRoot, '.venv');
  return path.join(app.getPath('userData'), '.venv');
}

const delay = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

// Stops the sidecar and everything it started: a graceful request
// (POST /api/shutdown, plus SIGTERM outside Windows), a bounded wait, then
// the whole process tree is killed. A plain pyProc.kill() used to leave the
// sidecar running as long as the progress websocket was open.
function stopSidecar(graceMs = SIDECAR_STOP_GRACE_MS): Promise<void> {
  const proc = pyProc;
  if (!proc || proc.exitCode !== null || proc.signalCode !== null) {
    if (proc) killSidecarGroup(proc);   // sweep a leftover ffmpeg, if any
    return Promise.resolve();
  }
  sidecarStopping ??= (async () => {
    const exited = new Promise<void>((resolve) => proc.once('exit', () => resolve()));
    if (apiPort) {
      fetch(`http://127.0.0.1:${apiPort}/api/shutdown`, {
        method: 'POST',
        headers: { 'X-AMV-Token': apiToken },
        signal: AbortSignal.timeout(graceMs),
      }).catch(() => {});
    }
    if (process.platform !== 'win32') {
      try { proc.kill('SIGTERM'); } catch {}
    }
    const inTime = await Promise.race([exited.then(() => true), delay(graceMs).then(() => false)]);
    if (!inTime) console.warn(`[main] sidecar still running after ${graceMs} ms — killing its process tree`);
    await killProcessTree(proc);
    if (!inTime) await Promise.race([exited, delay(1000)]);
  })().finally(() => { sidecarStopping = null; });
  return sidecarStopping;
}

function killSidecarGroup(proc: ChildProcess): void {
  if (process.platform === 'win32' || !proc.pid) return;
  try { process.kill(-proc.pid, 'SIGKILL'); } catch {}   // ESRCH: group already gone
}

async function killProcessTree(proc: ChildProcess): Promise<void> {
  if (process.platform !== 'win32') {
    killSidecarGroup(proc);
    return;
  }
  if (!proc.pid || proc.exitCode !== null) return;
  await new Promise<void>((resolve) => {
    const tk = spawn('taskkill', ['/F', '/T', '/PID', String(proc.pid)], { stdio: 'ignore', windowsHide: true });
    tk.on('exit', () => resolve());
    tk.on('error', () => resolve());
  });
}

async function killSidecarTree(): Promise<void> {
  await stopSidecar();
  // Give the OS a moment to release the DLL/file handles before we rm -rf.
  await new Promise((r) => setTimeout(r, 600));
}

async function performCleanReinstall(): Promise<void> {
  const userDataPath = app.getPath('userData');
  console.log('[main] clean reinstall requested — killing sidecar');
  await killSidecarTree();

  const toRemove = [
    venvPath(),
    path.join(userDataPath, '.setup_complete'),
    path.join(userDataPath, 'settings.json'),
    path.join(userDataPath, 'amv_tools.db'),
    path.join(userDataPath, 'thumbs'),
  ];
  for (const target of toRemove) {
    try {
      fs.rmSync(target, { recursive: true, force: true });
      console.log(`[main] removed ${target}`);
    } catch (err) {
      console.error(`[main] failed to remove ${target}:`, err);
    }
  }

  app.relaunch();
  app.exit(0);
}

// ── headless smoke-test hooks (CI only, inert for real users) ──────────────
// The Windows smoke workflow (.github/scripts/windows-smoke.ps1) drives the
// installed app through its sidecar. With AMV_SMOKE_CONTROL_DIR set, the app
// polls that folder for command files: `shot-<tag>` (save a capture of the
// window into AMV_SMOKE_SCREENSHOT_DIR), `reload` (reload the renderer, e.g.
// after the backend was installed through the API) and `quit` (quit through
// the normal path, sidecar shutdown included).
const smokeShotDir = process.env.AMV_SMOKE_SCREENSHOT_DIR;
const smokeControlDir = process.env.AMV_SMOKE_CONTROL_DIR;

function captureSmokeShot(tag: string, delayMs = 1500) {
  if (!smokeShotDir) return;
  setTimeout(() => {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    mainWindow.webContents.capturePage()
      .then((image) => {
        fs.mkdirSync(smokeShotDir, { recursive: true });
        fs.writeFileSync(path.join(smokeShotDir, `${tag.replace(/[^\w-]/g, '_')}.png`), image.toPNG());
        console.log(`[smoke] screenshot ${tag} written`);
      })
      .catch((err) => console.error('[smoke] screenshot failed', err));
  }, delayMs);
}

function watchSmokeControlDir() {
  if (!smokeControlDir) return;
  const timer = setInterval(() => {
    let names: string[] = [];
    try { names = fs.readdirSync(smokeControlDir); } catch { return; }
    for (const name of names) {
      // Only our own command files are consumed (and deleted): a mistyped
      // env var pointing at a real folder must never lose anyone's files.
      if (name !== 'quit' && name !== 'reload' && !/^shot-[\w-]{1,40}$/.test(name)) continue;
      try { fs.rmSync(path.join(smokeControlDir, name), { force: true }); } catch {}
      console.log(`[smoke] control: ${name}`);
      if (name === 'quit') {
        clearInterval(timer);
        app.quit();
        return;
      }
      if (name === 'reload') mainWindow?.webContents.reload();
      else if (name.startsWith('shot-')) captureSmokeShot(name.slice(5), 0);
    }
  }, 1000);
  timer.unref();
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: '#0F0F11',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      sandbox: true,
    },
  });

  // A file dropped outside a drop zone made Chromium navigate the window
  // to it (the video replaced the whole app). The app is a single page:
  // only its own reloads may navigate.
  mainWindow.webContents.on('will-navigate', (e, url) => {
    const bare = url.split(/[?#]/)[0];
    const own = useVite ? bare.startsWith('http://localhost:5173') : bare === pathToFileURL(distIndex).href;
    if (!own) e.preventDefault();
  });
  // target="_blank" links (model licences, docs) open in the browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//i.test(url)) void shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.webContents.once('did-finish-load', () => {
    rendererReady = true;
    flushBootstrap();
    captureSmokeShot('startup');
  });

  if (useVite) {
    await mainWindow.loadURL('http://localhost:5173');
  } else {
    await mainWindow.loadFile(distIndex);
  }
}

app.whenReady().then(async () => {
  // Open the window first so the user sees the BootstrapScreen while uv
  // resolves wheels. The sidecar boots concurrently and streams its stdout
  // / stderr into the renderer via amv:bootstrap events.
  await createWindow();
  watchSmokeControlDir();
  startSidecar()
    .then((port) => {
      emitBootstrap({ kind: 'ready', port });
      captureSmokeShot('ready');
    })
    .catch((err) => {
      console.error('Failed to start sidecar', err);
      emitBootstrap({ kind: 'error', message: String(err?.message ?? err) });
    });
  if (Number.isFinite(smokeExitMs) && smokeExitMs > 0) {
    setTimeout(() => app.quit(), smokeExitMs);
  }
});

// Quit on every platform: the window is never re-created (no 'activate'
// handler), so a macOS app left without window nor sidecar was useless.
app.on('window-all-closed', () => {
  app.quit();
});

app.on('before-quit', (event) => {
  if (sidecarStopped || !pyProc) return;
  event.preventDefault();
  void stopSidecar().finally(() => {
    sidecarStopped = true;
    app.quit();
  });
});

ipcMain.handle('amv:get-port', () => apiPort);
// Synchronous so the preload can expose the token before any renderer code runs.
ipcMain.on('amv:get-token', (e) => { e.returnValue = apiToken; });
ipcMain.handle('amv:bootstrap-state', () => ({ phase: bootstrapPhase, port: apiPort }));
ipcMain.handle('amv:clean-reinstall', async () => {
  await performCleanReinstall();
  // Returning is moot — performCleanReinstall calls app.relaunch + app.quit.
  return { ok: true };
});
ipcMain.handle('amv:open-dialog', async (_e, opts: { directory?: boolean; multi?: boolean }) => {
  const props: any[] = opts.directory ? ['openDirectory'] : ['openFile'];
  if (opts.multi) props.push('multiSelections');
  const r = await dialog.showOpenDialog(mainWindow!, { properties: props });
  return r.canceled ? [] : r.filePaths;
});
ipcMain.handle('amv:reveal', async (_e, p: string) => shell.showItemInFolder(p));
ipcMain.handle('amv:open-path', async (_e, p: string) => shell.openPath(p));
ipcMain.handle('amv:save-dialog', async (_e, opts: { defaultPath?: string; filters?: any[] }) => {
  const r = await dialog.showSaveDialog(mainWindow!, opts as any);
  return r.canceled ? null : r.filePath ?? null;
});
