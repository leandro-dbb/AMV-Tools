import type {
  AppSettings,
  DerushFolder,
  DerushItem,
  IndexQueueItem,
  ProgressEvent,
  SceneResult,
  SearchRequest,
  SegmentPreviewResponse,
  SegmentPromptBody,
  SegmentTrackBody,
  SystemStatus,
  TagSummary,
  VideoSummary,
} from './types';

declare global {
  interface Window {
    amvBridge?: {
      getApiPort: () => Promise<number>;
      getBootstrapState?: () => Promise<{ phase: 'starting' | 'ready' | 'error'; port: number }>;
      cleanReinstall?: () => Promise<{ ok: boolean }>;
      openFileDialog: (opts: { directory?: boolean; multi?: boolean; filters?: { name: string; extensions: string[] }[] }) => Promise<string[]>;
      saveFileDialog: (opts: { defaultPath?: string; filters?: { name: string; extensions: string[] }[] }) => Promise<string | null>;
      reveal: (p: string) => void;
      platform: () => string;
      onBootstrap?: (cb: (e: { line: string; stream: 'stdout' | 'stderr' }) => void) => () => void;
      onBootstrapReady?: (cb: (port: number) => void) => () => void;
      onBootstrapError?: (cb: (message: string) => void) => () => void;
    };
  }
}

export function getBaseUrl(): string {
  const port = (window as any).__AMV_API_PORT__ ?? 8731;
  return `http://127.0.0.1:${port}`;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${getBaseUrl()}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  if (res.status === 204) return undefined as unknown as T;
  return res.json() as Promise<T>;
}

export const api = {
  status: () => req<SystemStatus>('/api/status'),

  setupGpu: (backend: 'cu130' | 'cu130-trt' | 'cu126' | 'dml' | 'xpu' | 'cpu' | 'rocm') =>
    req<{ ok: boolean; log: string; hint?: string }>('/api/setup/install', {
      method: 'POST',
      body: JSON.stringify({ backend }),
    }),

  search: (body: SearchRequest) =>
    req<{ results: SceneResult[]; total: number }>('/api/search', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  searchImage: async (file: File, opts: { threshold?: number; top_k?: number } = {}) => {
    const fd = new FormData();
    fd.append('image', file);
    if (opts.threshold != null) fd.append('threshold', String(opts.threshold));
    if (opts.top_k != null) fd.append('top_k', String(opts.top_k));
    const res = await fetch(`${getBaseUrl()}/api/search/image`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
    return res.json() as Promise<{ results: SceneResult[]; total: number }>;
  },

  recentSearches: () => req<{ queries: string[] }>('/api/search/history'),

  allTags: (limit = 500) => req<{ tags: { tag: string; count: number }[] }>(`/api/tags/all?limit=${limit}`),

  listVideos: () => req<{ videos: VideoSummary[] }>('/api/videos'),

  videoTags: (videoId: number) =>
    req<{ tags: TagSummary[] }>(`/api/videos/${videoId}/tags`),

  videoScenes: (videoId: number | number[], opts: { tag?: string; threshold?: number; sort?: string; topOnly?: boolean }) => {
    const qs = new URLSearchParams();
    if (opts.tag) qs.set('tag', opts.tag);
    if (opts.threshold != null) qs.set('threshold', String(opts.threshold));
    if (opts.sort) qs.set('sort', opts.sort);
    if (opts.topOnly) qs.set('top_only', 'true');
    if (Array.isArray(videoId)) {
      videoId.forEach((id) => qs.append('video_id', String(id)));
      return req<{ scenes: SceneResult[] }>(`/api/scenes?${qs.toString()}`);
    }
    return req<{ scenes: SceneResult[] }>(`/api/videos/${videoId}/scenes?${qs.toString()}`);
  },

  scenePreview: (sceneId: number) =>
    req<{ frames: string[]; proxy_url?: string }>(`/api/scene/${sceneId}/preview`),

  queue: () => req<{ items: IndexQueueItem[] }>('/api/index/queue'),
  enqueue: (paths: { path: string; recursive?: boolean; group?: string | null }[]) =>
    req<{ added: number }>('/api/index/queue', {
      method: 'POST',
      body: JSON.stringify({ paths }),
    }),
  startIndexing: (phases: ('tag' | 'embed')[] = ['tag', 'embed']) =>
    req<{ started: boolean; phases: string[] }>('/api/index/start', {
      method: 'POST',
      body: JSON.stringify({ phases }),
    }),

  export: (body: { scene_id: number; start_ms?: number; end_ms?: number; output_path?: string }) =>
    req<{ ok: boolean; output: string }>('/api/export', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  exportBatch: (scene_ids: number[]) =>
    req<{ exported: { scene_id: number; output: string }[]; failed: { scene_id: number; error: string }[] }>(
      '/api/export/batch',
      { method: 'POST', body: JSON.stringify({ scene_ids }) }
    ),

  getSettings: () => req<AppSettings>('/api/settings'),
  saveSettings: (settings: Partial<AppSettings>) =>
    req<AppSettings>('/api/settings', {
      method: 'PUT',
      body: JSON.stringify(settings),
    }),
  importSettings: (payload: AppSettings) =>
    req<AppSettings>('/api/settings/import', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  listDatabases: () =>
    req<{ databases: { path: string; scenes: number; videos: number; size_kb: number; primary: boolean }[] }>(
      '/api/databases'
    ),
  setPrimaryDatabase: (path: string) =>
    req<{ ok: boolean }>('/api/databases/primary', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
  addDatabase: (path: string) =>
    req<{ ok: boolean }>('/api/databases', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
  removeDatabase: (path: string) =>
    req<{ ok: boolean }>(`/api/databases?path=${encodeURIComponent(path)}`, { method: 'DELETE' }),
  cleanupDatabase: (path: string) =>
    req<{ removed: number }>('/api/databases/cleanup', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
  verifyDatabase: (path: string) =>
    req<{ missing: { id: number; filepath: string }[] }>(
      `/api/databases/verify?path=${encodeURIComponent(path)}`
    ),
  relinkVideo: (body: { db_path: string; video_id: number; new_filepath: string }) =>
    req<{ ok: boolean; error: string | null }>('/api/databases/relink', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  relinkFolder: (body: { db_path: string; search_dirs: string[]; video_ids?: number[] }) =>
    req<{ relinked: number; ambiguous: string[]; still_missing: string[] }>('/api/databases/relink_folder', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  mergeDatabases: (target: string, sources: string[]) =>
    req<{ videos_added: number; scenes_added: number; tags_added: number; skipped_videos: number }>(
      '/api/databases/merge',
      { method: 'POST', body: JSON.stringify({ target, sources }) }
    ),
  stopIndexing: () =>
    req<{ stopped: boolean }>('/api/index/stop', { method: 'POST' }),

  thumbnailUrl: (sceneId: number) => `${getBaseUrl()}/api/scene/${sceneId}/thumbnail`,
  proxyUrl: (sceneId: number) => `${getBaseUrl()}/api/scene/${sceneId}/proxy`,
  sourceUrl: (sceneId: number) => `${getBaseUrl()}/api/scene/${sceneId}/source`,

  // SAM 2 mask / alpha export
  segmentPreview: (sceneId: number, body: SegmentPromptBody) =>
    req<SegmentPreviewResponse>(`/api/scene/${sceneId}/segment/preview`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  segmentTrack: (sceneId: number, body: SegmentTrackBody) =>
    req<{ ok: boolean; session_id: string; n_frames: number; frame_w: number; frame_h: number }>(
      `/api/scene/${sceneId}/segment/track`,
      { method: 'POST', body: JSON.stringify(body) },
    ),
  segmentExport: (sceneId: number, body: { session_id: string; codec?: string; output_path?: string }) =>
    req<{ ok: boolean; output: string }>(`/api/scene/${sceneId}/segment/export`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  segmentDrop: (sessionId: string) =>
    req<{ ok: boolean; dropped: boolean }>(`/api/scene/segment/${sessionId}`, { method: 'DELETE' }),
  segmentFrameUrl: (sessionId: string, idx: number) =>
    `${getBaseUrl()}/api/scene/segment/${sessionId}/frame/${idx}`,
  segmentMaskPreviewUrl: (sessionId: string) =>
    `${getBaseUrl()}/api/scene/segment/${sessionId}/mask_preview`,
  segmentMaskUrl: (sessionId: string, idx: number) =>
    `${getBaseUrl()}/api/scene/segment/${sessionId}/mask/${idx}`,

  // Derush (selects) workflow
  derushList: () =>
    req<{ folders: DerushFolder[]; items: DerushItem[]; exporting: boolean }>('/api/derush'),
  derushKept: (videoId?: number) =>
    req<{ scene_ids: number[] }>(`/api/derush/kept${videoId != null ? `?video_id=${videoId}` : ''}`),
  derushLevel: (sceneId: number, delta: 1 | -1, folderId?: number | null) =>
    req<{ ok: boolean; level: 0 | 1 | 2; kept: boolean; favorite: boolean; item_id: number | null; scene_id: number }>('/api/derush/level', {
      method: 'POST',
      body: JSON.stringify({ scene_id: sceneId, delta, folder_id: folderId ?? null }),
    }),
  renameVideoGroup: (oldName: string, newName: string | null) =>
    req<{ ok: boolean; updated: number; group_name: string | null }>('/api/videos/groups/rename', {
      method: 'POST',
      body: JSON.stringify({ old: oldName, new: newName }),
    }),
  setVideoDerushed: (videoId: number, done: boolean) =>
    req<{ ok: boolean }>(`/api/videos/${videoId}`, {
      method: 'PATCH',
      body: JSON.stringify({ derushed: done }),
    }),
  setVideoGroup: (videoId: number, group: string | null) =>
    req<{ ok: boolean; group_name: string | null }>(`/api/videos/${videoId}`, {
      method: 'PATCH',
      body: JSON.stringify(group ? { group_name: group } : { clear_group: true }),
    }),
  derushToggle: (sceneId: number, folderId?: number | null, favorite?: boolean) =>
    req<{ ok: boolean; kept: boolean; item_id: number | null; scene_id: number; favorite: boolean }>('/api/derush/toggle', {
      method: 'POST',
      body: JSON.stringify({ scene_id: sceneId, folder_id: folderId ?? null, favorite: favorite ?? false }),
    }),
  derushPatchItem: (itemId: number, body: { custom_name?: string; folder_ids?: number[]; clear_name?: boolean; favorite?: boolean }) =>
    req<{ ok: boolean }>(`/api/derush/items/${itemId}`, { method: 'PATCH', body: JSON.stringify(body) }),
  derushDeleteItem: (itemId: number) =>
    req<{ ok: boolean }>(`/api/derush/items/${itemId}`, { method: 'DELETE' }),
  derushCreateFolder: (name: string) =>
    req<{ ok: boolean; id: number; name: string }>('/api/derush/folders', {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),
  derushRenameFolder: (id: number, name: string) =>
    req<{ ok: boolean }>(`/api/derush/folders/${id}`, { method: 'PATCH', body: JSON.stringify({ name }) }),
  derushDeleteFolder: (id: number) =>
    req<{ ok: boolean }>(`/api/derush/folders/${id}`, { method: 'DELETE' }),
  derushExport: (scope: 'all' | 'folder' | 'root' | 'favorites', folderId?: number) =>
    req<{ ok: boolean; started: boolean; total: number }>('/api/derush/export', {
      method: 'POST',
      body: JSON.stringify({ scope, folder_id: folderId ?? null }),
    }),
  mergeScenes: (sceneIds: number[]) =>
    req<{ ok: boolean; kept_scene_id: number; start_ms: number; end_ms: number; removed: number[] }>(
      '/api/scenes/merge',
      { method: 'POST', body: JSON.stringify({ scene_ids: sceneIds }) }
    ),
  videoPlayable: (videoId: number) =>
    req<{ url: string; kind: 'direct' | 'remux' | 'transcode' }>(`/api/videos/${videoId}/playable`),
  videoTranscode: (videoId: number) =>
    req<{ ok: boolean; ready: boolean; url?: string }>(`/api/videos/${videoId}/transcode`, { method: 'POST' }),

  progressSocket: () => new WebSocket(`${getBaseUrl().replace(/^http/, 'ws')}/ws/progress`),

  revealOutput: (p: string) => window.amvBridge?.reveal(p),
};

export function platformIs(plat: 'win32' | 'darwin' | 'linux'): boolean {
  const p = window.amvBridge?.platform?.();
  if (p) return p === plat;
  if (plat === 'win32') return navigator.userAgent.includes('Windows');
  if (plat === 'darwin') return navigator.userAgent.includes('Mac');
  return navigator.userAgent.includes('Linux');
}
