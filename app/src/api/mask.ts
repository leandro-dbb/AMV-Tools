// Mask Mode correction endpoints added in 0.4 (keyframes, brush, cutout
// preview). Separate from client.ts to keep the shared client untouched.
import { authHeaders, getBaseUrl, withToken } from './client';
import type { SegmentPreviewResponse } from './types';

export type SegmentSessionInfo = SegmentPreviewResponse & {
  keyframes?: number[];
  corrected_frames?: number[];
  tracked?: boolean;
};

export interface BrushStroke {
  points: [number, number][];
  radius: number;
  mode: 'add' | 'remove';
}

export interface KeyframePrompt {
  frame_idx: number;
  positive: [number, number][];
  negative: [number, number][];
  box: [number, number, number, number] | null;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${getBaseUrl()}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...authHeaders(), ...(init?.headers ?? {}) },
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const maskApi = {
  brush: (sessionId: string, body: { frame_idx: number; strokes?: BrushStroke[]; reset?: boolean }) =>
    req<{ ok: boolean; frame_idx: number; corrected_frames: number[] }>(
      `/api/scene/segment/${sessionId}/brush`, { method: 'POST', body: JSON.stringify(body) }),
  dropKeyframe: (sessionId: string, frameIdx: number) =>
    req<{ ok: boolean; removed: boolean; keyframes: number[] }>(
      `/api/scene/segment/${sessionId}/keyframes/${frameIdx}`, { method: 'DELETE' }),
  cutoutUrl: (sessionId: string, idx: number) => withToken(`${getBaseUrl()}/api/scene/segment/${sessionId}/cutout/${idx}`),
};
