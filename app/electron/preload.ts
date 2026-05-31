import { contextBridge, ipcRenderer } from 'electron';

type BootstrapState = { phase: 'starting' | 'ready' | 'error'; port: number };

const bridge = {
  getApiPort: () => ipcRenderer.invoke('amv:get-port'),
  getBootstrapState: () => ipcRenderer.invoke('amv:bootstrap-state') as Promise<BootstrapState>,
  cleanReinstall: () => ipcRenderer.invoke('amv:clean-reinstall') as Promise<{ ok: boolean }>,
  openFileDialog: (opts: { directory?: boolean; multi?: boolean; filters?: { name: string; extensions: string[] }[] }) =>
    ipcRenderer.invoke('amv:open-dialog', opts) as Promise<string[]>,
  saveFileDialog: (opts: { defaultPath?: string; filters?: { name: string; extensions: string[] }[] }) =>
    ipcRenderer.invoke('amv:save-dialog', opts) as Promise<string | null>,
  reveal: (p: string) => ipcRenderer.invoke('amv:reveal', p),
  platform: () => process.platform,
  onProgress: (cb: (e: any) => void) => {
    const listener = (_: unknown, payload: any) => cb(payload);
    ipcRenderer.on('amv:progress', listener);
    return () => ipcRenderer.removeListener('amv:progress', listener);
  },
  onBootstrap: (cb: (e: { line: string; stream: 'stdout' | 'stderr' }) => void) => {
    const listener = (_: unknown, payload: any) => cb(payload);
    ipcRenderer.on('amv:bootstrap', listener);
    return () => ipcRenderer.removeListener('amv:bootstrap', listener);
  },
  onBootstrapReady: (cb: (port: number) => void) => {
    const listener = (_: unknown, port: number) => cb(port);
    ipcRenderer.on('amv:bootstrap-ready', listener);
    return () => ipcRenderer.removeListener('amv:bootstrap-ready', listener);
  },
  onBootstrapError: (cb: (message: string) => void) => {
    const listener = (_: unknown, message: string) => cb(message);
    ipcRenderer.on('amv:bootstrap-error', listener);
    return () => ipcRenderer.removeListener('amv:bootstrap-error', listener);
  },
};

contextBridge.exposeInMainWorld('amvBridge', bridge);
