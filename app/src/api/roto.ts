// Roto editor (v0.6) sidecar API. Follows docs/roto-editor-api.md to the
// letter. Separate from client.ts to keep the shared client untouched.
import { authHeaders, getBaseUrl } from './client';

export type RotoReason = 'lost' | 'jump' | 'specks' | 'object_lost' | 'hole' | 'rough';

export interface RotoSuspect { frame: number; score: number; reasons: RotoReason[] }

export interface RotoLayer { id: string; name: string; color: string }

export interface RotoEdgeSettings {
  keep_ink: boolean;
  softness: number;
  grow_px: number;
  temporal: number;
}

export type RotoJobKind = 'detect' | 'propagate' | 'export';

export interface RotoExportFiles { prores?: string; png_dir?: string; matte?: string }

export interface RotoJob {
  kind: RotoJobKind;
  percent: number;
  message: string;
  done: boolean;
  cancelled: boolean;
  error: string | null;
  // export: { files: {prores?, png_dir?, matte?}, added_to_selection: boolean }
  result?: any;
}

export interface RotoModels {
  device: 'cuda' | 'mps' | 'cpu';
  detector: string;
  characters: string | null;
  tracker: string | null;
  clicks: string;
  sam3_missing: boolean;
  slow_machine: boolean;
}

export interface RotoProject {
  id: string;
  scene_id: number | null;
  name: string;
  n_frames: number;
  fps: number;
  width: number;
  height: number;
  start_ms: number;
  end_ms: number;
  has_ai_pass: boolean;
  layers: RotoLayer[];
  keyframes: Record<string, number[]>;
  suspects: RotoSuspect[];
  settings: RotoEdgeSettings;
  models: RotoModels;
  job: RotoJob | null;
  version: number;
  frame_versions: Record<string, number>;
  exports: RotoExportFiles | null;
  can_undo: boolean;
  can_redo: boolean;
}

export type RotoPoint = [number, number];
export type RotoBrushMode = 'add' | 'remove' | 'erase';
export interface RotoStroke { points: RotoPoint[]; radius: number; mode: RotoBrushMode }

export interface RotoDetectBody {
  layer?: string;
  target: 'all' | 'main' | 'click';
  frame?: number;
  positive?: RotoPoint[];
  negative?: RotoPoint[];
}

export interface RotoEditResult {
  ms: number;
  version: number;
  frame: number;
  keyframes: Record<string, number[]>;
  suspects: RotoSuspect[];
}

export interface RotoClickResult extends RotoEditResult { model: string }

export interface RotoUndoResult {
  version: number;
  frame: number;
  can_undo: boolean;
  can_redo: boolean;
  keyframes: Record<string, number[]>;
  suspects: RotoSuspect[];
}

export type RotoExportFormat = 'prores' | 'png' | 'matte';

export interface RotoSceneProjects {
  projects: { id: string; start_ms: number; end_ms: number; has_ai_pass: boolean; exported: boolean }[];
}

/** Error carrying the HTTP status and the sidecar's human readable `detail`. */
export class RotoApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${getBaseUrl()}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...authHeaders(), ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    let detail = text;
    try {
      const j = JSON.parse(text);
      if (j && typeof j.detail === 'string') detail = j.detail;
      else if (j && j.detail) detail = JSON.stringify(j.detail);
    } catch { /* not JSON */ }
    throw new RotoApiError(res.status, detail || `HTTP ${res.status}`);
  }
  if (res.status === 204) return undefined as unknown as T;
  return res.json() as Promise<T>;
}

const post = <T>(path: string, body?: unknown) =>
  req<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) });

const P = (id: string) => `/api/roto/${encodeURIComponent(id)}`;

export const rotoApi = {
  open: (body: { scene_id: number; start_ms?: number; end_ms?: number }) => post<RotoProject>('/api/roto/open', body),
  get: (id: string) => req<RotoProject>(P(id)),
  detect: (id: string, body: RotoDetectBody) => post<RotoJob>(`${P(id)}/detect`, body),
  job: (id: string) => req<RotoJob | null>(`${P(id)}/job`),
  cancel: (id: string) => post<{ ok: boolean }>(`${P(id)}/cancel`),
  brush: (id: string, body: { frame: number; layer: string; strokes: RotoStroke[] }) =>
    post<RotoEditResult>(`${P(id)}/brush`, body),
  click: (id: string, body: { frame: number; layer: string; x: number; y: number; positive: boolean }) =>
    post<RotoClickResult>(`${P(id)}/click`, body),
  lock: (id: string, body: { frame: number; layer: string; locked: boolean }) =>
    post<{ keyframes: Record<string, number[]>; suspects: RotoSuspect[] }>(`${P(id)}/lock`, body),
  propagate: (id: string, body: { frame: number; layer: string; direction: 'forward' | 'backward' | 'both' }) =>
    post<RotoJob>(`${P(id)}/propagate`, body),
  undo: (id: string) => post<RotoUndoResult>(`${P(id)}/undo`),
  redo: (id: string) => post<RotoUndoResult>(`${P(id)}/redo`),
  putSettings: (id: string, s: RotoEdgeSettings) =>
    req<RotoProject>(`${P(id)}/settings`, { method: 'PUT', body: JSON.stringify(s) }),
  addLayer: (id: string, name?: string) => post<RotoProject>(`${P(id)}/layers`, name ? { name } : {}),
  deleteLayer: (id: string, layer: string) =>
    req<RotoProject>(`${P(id)}/layers/${encodeURIComponent(layer)}`, { method: 'DELETE' }),
  exportProject: (id: string, body: { formats: RotoExportFormat[]; add_to_selection: boolean }) =>
    post<RotoJob>(`${P(id)}/export`, body),
  sceneProjects: (sceneId: number) => req<RotoSceneProjects>(`/api/roto/scene/${sceneId}`),

  /** Image routes: fetched with the auth header (see rotoImages.ts). */
  frameUrl: (id: string, i: number) => `${getBaseUrl()}${P(id)}/frame/${i}`,
  matteUrl: (id: string, i: number, v: number, layer: string | null, settingsKey: string) => {
    const qs = new URLSearchParams();
    if (layer) qs.set('layer', layer);
    qs.set('v', String(v));
    // Extra, ignored by the server: makes the URL (our cache key) change when
    // the edge settings change, since the route applies them server-side.
    qs.set('s', settingsKey);
    return `${getBaseUrl()}${P(id)}/matte/${i}?${qs.toString()}`;
  },
};
