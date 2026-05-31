import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron';
import { spawn, ChildProcess } from 'node:child_process';
import net from 'node:net';
import path from 'node:path';
import fs from 'node:fs';

const isDev = !app.isPackaged;
const distIndex = path.join(__dirname, '..', 'dist', 'index.html');
const useVite = isDev && (process.env.AMV_USE_VITE === '1' || !fs.existsSync(distIndex));
const smokeExitMs = Number(process.env.AMV_SMOKE_EXIT_MS || 0);
let pyProc: ChildProcess | null = null;
let mainWindow: BrowserWindow | null = null;
let apiPort = 0;

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

async function startSidecar(): Promise<number> {
  apiPort = await findFreePort();
  const { uvExe, projectRoot } = resolvePythonRoot();
  const serverPath = path.join(projectRoot, 'backend', 'server.py');

  if (!fs.existsSync(serverPath)) {
    throw new Error(`Backend not found at ${serverPath}`);
  }

  const useUv = fs.existsSync(uvExe);
  const cmd = useUv ? uvExe : (process.platform === 'win32' ? 'python' : 'python3');
  // `uv run` re-syncs the environment to the project's *default* deps on every
  // launch. The GPU torch wheels live in optional extras (cu130, cu126, …), so
  // a plain re-sync silently uninstalls the CUDA build and reinstalls +cpu —
  // which is exactly what drops a working GPU install back to "CPU (forced
  // fallback)" on the next start. Once the venv exists (the in-app installer or
  // a dev `uv sync --extra <backend>` built it), launch with --no-sync so the
  // chosen backend sticks. Only the very first run (no venv yet) syncs, to
  // bootstrap the base deps that serve the setup UI.
  const noSync = useUv && fs.existsSync(venvPath()) ? ['--no-sync'] : [];
  const args = useUv
    ? ['run', ...noSync, '--project', projectRoot, 'python', serverPath, '--port', String(apiPort)]
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
      AMV_DATA_DIR: userDataPath,
      UV_CACHE_DIR: isDev ? path.join(runtimeUvRoot, 'uv_cache') : path.join(userDataPath, '.uv_cache'),
      UV_PYTHON_INSTALL_DIR: path.join(runtimeUvRoot, 'python'),
      ...(isDev ? {} : { UV_PROJECT_ENVIRONMENT: path.join(userDataPath, '.venv') }),
      PATH: `${bundledUvRoot}${path.delimiter}${runtimeUvRoot}${path.delimiter}${process.env.PATH ?? ''}`,
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let lastActivity = Date.now();
  const bump = () => { lastActivity = Date.now(); };
  const emitLines = (chunk: Buffer, stream: BootstrapStream) => {
    bump();
    const text = chunk.toString();
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trimEnd();
      if (!line) continue;
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
  pyProc.on('exit', (code) => {
    console.log(`[main] sidecar exited with code ${code}`);
    if (bootstrapPhase === 'starting') {
      emitBootstrap({ kind: 'error', message: `Sidecar exited with code ${code} before opening port ${apiPort}` });
    }
  });

  // First-run install can take many minutes (PyTorch wheels etc). Be patient as
  // long as uv keeps logging; bail only if the sidecar goes silent for too long
  // or if a hard ceiling is reached.
  await waitUntilReady(apiPort, () => Date.now() - lastActivity, 60_000, 15 * 60_000);
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

async function killSidecarTree(): Promise<void> {
  if (!pyProc || pyProc.killed) return;
  const pid = pyProc.pid;
  try { pyProc.kill(); } catch {}
  if (process.platform === 'win32' && pid) {
    await new Promise<void>((resolve) => {
      const tk = spawn('taskkill', ['/F', '/T', '/PID', String(pid)], { stdio: 'ignore' });
      tk.on('exit', () => resolve());
      tk.on('error', () => resolve());
    });
  }
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

  mainWindow.webContents.once('did-finish-load', () => {
    rendererReady = true;
    flushBootstrap();
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
  startSidecar()
    .then((port) => emitBootstrap({ kind: 'ready', port }))
    .catch((err) => {
      console.error('Failed to start sidecar', err);
      emitBootstrap({ kind: 'error', message: String(err?.message ?? err) });
    });
  if (Number.isFinite(smokeExitMs) && smokeExitMs > 0) {
    setTimeout(() => app.quit(), smokeExitMs);
  }
});

app.on('window-all-closed', () => {
  pyProc?.kill();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  pyProc?.kill();
});

ipcMain.handle('amv:get-port', () => apiPort);
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
ipcMain.handle('amv:save-dialog', async (_e, opts: { defaultPath?: string; filters?: any[] }) => {
  const r = await dialog.showSaveDialog(mainWindow!, opts as any);
  return r.canceled ? null : r.filePath ?? null;
});
