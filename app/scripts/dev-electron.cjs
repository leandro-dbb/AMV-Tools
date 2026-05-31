const { spawn, spawnSync } = require('node:child_process');
const net = require('node:net');

const isWin = process.platform === 'win32';
const npmCmd = 'npm';
const npxCmd = 'npx';

function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: 'inherit',
      shell: isWin,
      ...options,
    });
    child.on('error', reject);
    child.on('exit', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${command} ${args.join(' ')} exited with code ${code}`));
    });
  });
}

function waitForPort(port, host = '127.0.0.1', timeoutMs = 30_000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const socket = net.connect(port, host, () => {
        socket.end();
        resolve();
      });
      socket.on('error', () => {
        socket.destroy();
        if (Date.now() - start > timeoutMs) {
          reject(new Error(`Vite did not open ${host}:${port} within ${timeoutMs}ms`));
        } else {
          setTimeout(tick, 250);
        }
      });
    };
    tick();
  });
}

function stopProcessTree(child) {
  if (!child || !child.pid) return;
  if (isWin) {
    spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
  } else if (!child.killed) {
    child.kill();
  }
}

async function main() {
  console.log('[dev-electron] building Electron main process...');
  await run(npmCmd, ['run', 'build:main']);

  console.log('[dev-electron] starting Vite...');
  const vite = spawn(npmCmd, ['run', 'dev', '--', '--host', '127.0.0.1'], {
    stdio: 'inherit',
    shell: isWin,
  });
  const viteExited = new Promise((_, reject) => {
    vite.on('exit', (code) => {
      if (code != null && code !== 0) {
        reject(new Error(`Vite exited with code ${code}`));
      }
    });
  });
  vite.on('exit', (code) => {
    if (code != null && code !== 0) {
      console.error(`[dev-electron] Vite exited with code ${code}`);
    }
  });

  let electron;
  const shutdown = () => {
    stopProcessTree(electron);
    stopProcessTree(vite);
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
  process.on('exit', shutdown);

  try {
    console.log('[dev-electron] waiting for Vite on 127.0.0.1:5173...');
    await Promise.race([waitForPort(5173), viteExited]);
    console.log('[dev-electron] starting Electron...');
    electron = spawn(npxCmd, ['electron', '.'], {
      stdio: 'inherit',
      shell: isWin,
      env: { ...process.env, AMV_USE_VITE: '1' },
    });
    electron.on('exit', (code) => {
      shutdown();
      process.exit(code ?? 0);
    });
    electron.on('error', (err) => {
      console.error(err);
      shutdown();
      process.exit(1);
    });
  } catch (err) {
    console.error(err);
    shutdown();
    process.exit(1);
  }
}

main();
