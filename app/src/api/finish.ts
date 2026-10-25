// Finishing toolbox API (upscale, slow-mo / 60 fps, depth, cutout).
// Kept out of client.ts so the shared client stays small; same base URL and
// error format.
import { authHeaders, getBaseUrl, withToken } from './client';

export type FinishKind = 'upscale' | 'interpolate' | 'depth' | 'matte';
export type FinishJobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

export interface FinishSource {
  scene_id?: number;
  path?: string;
  start_ms?: number;
  end_ms?: number;
  label?: string;
}

export interface FinishModel {
  id: string;
  kind: 'upscale' | 'interpolate' | 'depth';
  label: string;
  license: string;
  non_commercial?: boolean;
  source: string;
  size: number;
  installed: boolean;
  disk_size: number;
  downloading: boolean;
  download_percent: number | null;
  download_error: string | null;
}

export interface FinishCapabilities {
  torch: boolean;
  device: { backend: string; name: string; vram_gb: number | null };
  cpu_only: boolean;
  mask_enabled: boolean;
  models: FinishModel[];
}

export interface FinishJob {
  id: string;
  kind: FinishKind;
  label: string;
  params: Record<string, unknown>;
  status: FinishJobStatus;
  progress: number;
  percent: number;
  message: string;
  output: string | null;
  error: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  eta_seconds: number | null;
}

export interface FinishEstimate {
  kind: FinishKind;
  device: string;
  device_name: string;
  cpu_only: boolean;
  frames: number;
  work_frames: number;
  seconds_per_frame: number;
  model_load_seconds: number;
  estimated_seconds: number;
  width: number;
  height: number;
  fps: number;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${getBaseUrl()}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...authHeaders(), ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try { detail = JSON.parse(text).detail ?? text; } catch { /* plain text */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

export const finishApi = {
  capabilities: () => req<FinishCapabilities>('/api/finish/capabilities'),
  models: () => req<{ models: FinishModel[]; root: string }>('/api/finish/models'),
  downloadModel: (id: string) =>
    req<{ ok: boolean; started: boolean; model: FinishModel }>(`/api/finish/models/${encodeURIComponent(id)}/download`, { method: 'POST' }),
  deleteModel: (id: string) =>
    req<{ ok: boolean; removed: boolean; model: FinishModel }>(`/api/finish/models/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  estimate: (kind: FinishKind, source: FinishSource, params: Record<string, unknown>) =>
    req<FinishEstimate>('/api/finish/estimate', { method: 'POST', body: JSON.stringify({ kind, source, params }) }),
  jobs: () => req<{ jobs: FinishJob[] }>('/api/finish/jobs'),
  createJobs: (kind: FinishKind, sources: FinishSource[], params: Record<string, unknown>) =>
    req<{ ok: boolean; jobs: FinishJob[] }>('/api/finish/jobs', { method: 'POST', body: JSON.stringify({ kind, sources, params }) }),
  cancelJob: (id: string) => req<{ ok: boolean }>(`/api/finish/jobs/${id}/cancel`, { method: 'POST' }),
  removeJob: (id: string) => req<{ ok: boolean }>(`/api/finish/jobs/${id}`, { method: 'DELETE' }),
  clearJobs: () => req<{ ok: boolean; removed: number }>('/api/finish/jobs', { method: 'DELETE' }),
};

/** Kind → registry model it needs (upscale depends on the chosen model). */
export function requiredModel(kind: FinishKind, params: Record<string, unknown>): string | null {
  if (kind === 'upscale') return String(params.model ?? 'realesr-animevideov3');
  if (kind === 'interpolate') return 'rife-4.25';
  if (kind === 'depth') return 'depth-anything-v2-small';
  return null;
}

export function formatBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KB`;
  return `${n} B`;
}

export function formatDuration(s: number): string {
  if (!isFinite(s) || s < 0) return '—';
  if (s < 60) return `${Math.max(1, Math.round(s))} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ${Math.round(s % 60).toString().padStart(2, '0')} s`;
  return `${Math.floor(m / 60)} h ${(m % 60).toString().padStart(2, '0')} min`;
}

/** Subscribe to finish_job / finish_model events on the shared progress socket. */
export function subscribeFinish(onEvent: (e: { type: string; job?: FinishJob; model_id?: string; percent?: number; done?: boolean; ok?: boolean; message?: string }) => void): () => void {
  let ws: WebSocket | null = null;
  let closed = false;
  let retry: ReturnType<typeof setTimeout> | null = null;
  const open = () => {
    ws = new WebSocket(withToken(`${getBaseUrl().replace(/^http/, 'ws')}/ws/progress`));
    ws.onmessage = (m) => {
      try {
        const e = JSON.parse(m.data);
        if (e.type === 'finish_job' || e.type === 'finish_model') onEvent(e);
      } catch { /* ignore */ }
    };
    ws.onclose = () => { if (!closed) retry = setTimeout(open, 2000); };
  };
  open();
  return () => {
    closed = true;
    if (retry) clearTimeout(retry);
    ws?.close();
  };
}
