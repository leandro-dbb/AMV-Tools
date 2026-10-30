// Pure helpers for the roto editor (no React, no DOM): suspect navigation,
// viewer transforms, cache keys, keyboard filtering. Unit tested in
// rotoUtils.test.ts.
import type { RotoEdgeSettings, RotoProject, RotoReason, RotoSuspect } from '../api/roto';

export type RotoView = 'overlay' | 'black' | 'white' | 'checker' | 'original';
export const ROTO_VIEWS: RotoView[] = ['overlay', 'black', 'white', 'checker', 'original'];

export const REASON_KEYS: Record<RotoReason, string> = {
  lost: 'roto.reason.lost',
  jump: 'roto.reason.jump',
  specks: 'roto.reason.specks',
  object_lost: 'roto.reason.object_lost',
  hole: 'roto.reason.hole',
  rough: 'roto.reason.rough',
};

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Frames locked (edited or validated) for a layer. */
export function lockedFrames(keyframes: Record<string, number[]> | undefined, layer: string | null): number[] {
  if (!keyframes || !layer) return [];
  return keyframes[layer] ?? [];
}

/** Suspects still to check: drop frames the user already locked on the active layer. */
export function pendingSuspects(suspects: RotoSuspect[], locked: number[]): RotoSuspect[] {
  if (!locked.length) return suspects;
  const set = new Set(locked);
  return suspects.filter((s) => !set.has(s.frame));
}

/** Next suspect frame after `current` (wraps to the first). null when none. */
export function nextSuspect(suspects: RotoSuspect[], current: number): number | null {
  if (!suspects.length) return null;
  const frames = suspects.map((s) => s.frame).sort((a, b) => a - b);
  const after = frames.find((f) => f > current);
  return after ?? frames[0];
}

/** Previous suspect frame before `current` (wraps to the last). null when none. */
export function prevSuspect(suspects: RotoSuspect[], current: number): number | null {
  if (!suspects.length) return null;
  const frames = suspects.map((s) => s.frame).sort((a, b) => a - b);
  for (let i = frames.length - 1; i >= 0; i--) if (frames[i] < current) return frames[i];
  return frames[frames.length - 1];
}

/** Viewer transform: screen = frame * scale + (tx, ty), in CSS pixels. */
export interface ViewXform { scale: number; tx: number; ty: number }

export function fitXform(viewW: number, viewH: number, frameW: number, frameH: number, margin = 0.96): ViewXform {
  if (viewW <= 0 || viewH <= 0 || frameW <= 0 || frameH <= 0) return { scale: 1, tx: 0, ty: 0 };
  const scale = Math.min(viewW / frameW, viewH / frameH) * margin;
  return { scale, tx: (viewW - frameW * scale) / 2, ty: (viewH - frameH * scale) / 2 };
}

export function screenToFrame(x: number, y: number, xf: ViewXform): [number, number] {
  return [(x - xf.tx) / xf.scale, (y - xf.ty) / xf.scale];
}

export function frameToScreen(x: number, y: number, xf: ViewXform): [number, number] {
  return [x * xf.scale + xf.tx, y * xf.scale + xf.ty];
}

export const MIN_ZOOM = 0.05;
export const MAX_ZOOM = 32;

/** Zoom by `factor` keeping the frame point under (cx, cy) fixed on screen. */
export function zoomAround(xf: ViewXform, cx: number, cy: number, factor: number): ViewXform {
  const scale = clamp(xf.scale * factor, MIN_ZOOM, MAX_ZOOM);
  const k = scale / xf.scale;
  return { scale, tx: cx - (cx - xf.tx) * k, ty: cy - (cy - xf.ty) * k };
}

/** Version used to cache-bust the matte of frame i. With temporal smoothing a
 * change on one frame alters its neighbours too, so fall back to the global
 * version then. */
export function matteVersion(p: Pick<RotoProject, 'version' | 'frame_versions' | 'settings'>, i: number): number {
  if (p.settings && p.settings.temporal > 0) return p.version;
  return p.frame_versions?.[String(i)] ?? p.version;
}

/** Short stable key for the edge settings (part of the matte URL). */
export function settingsKey(s: RotoEdgeSettings | null | undefined): string {
  if (!s) return '0';
  return `${s.keep_ink ? 1 : 0}_${s.softness.toFixed(2)}_${s.grow_px.toFixed(1)}_${s.temporal.toFixed(2)}`;
}

export const DEFAULT_EDGE_SETTINGS: RotoEdgeSettings = { keep_ink: true, softness: 0, grow_px: 0, temporal: 0 };

/** True when a key event comes from a field where the user types text. */
export function isTypingTarget(el: { tagName?: string; type?: string; isContentEditable?: boolean } | null | undefined): boolean {
  if (!el) return false;
  if (el.isContentEditable) return true;
  const tag = (el.tagName || '').toUpperCase();
  if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (tag !== 'INPUT') return false;
  const type = (el.type || 'text').toLowerCase();
  return !['range', 'checkbox', 'radio', 'button', 'submit', 'color'].includes(type);
}

/** Drop points closer than `minDist` to the previous kept one (keeps the last). */
export function thinPoints(points: [number, number][], minDist = 1): [number, number][] {
  if (points.length <= 2) return points.slice();
  const out: [number, number][] = [points[0]];
  for (let i = 1; i < points.length - 1; i++) {
    const [px, py] = out[out.length - 1];
    const [x, y] = points[i];
    if (Math.hypot(x - px, y - py) >= minDist) out.push(points[i]);
  }
  out.push(points[points.length - 1]);
  return out;
}

/** Brush size step for `[` / `]`. */
export function stepRadius(r: number, dir: 1 | -1): number {
  const step = r < 10 ? 1 : Math.max(1, Math.round(r * 0.15));
  return clamp(r + dir * step, 1, 200);
}

/** Converts a grayscale RGBA buffer (white = character) into an alpha mask in
 * place: rgb set to white, alpha = luminance (red channel of a gray PNG). */
export function lumToAlpha(data: Uint8ClampedArray): void {
  for (let i = 0; i < data.length; i += 4) {
    const a = data[i];
    data[i] = 255;
    data[i + 1] = 255;
    data[i + 2] = 255;
    data[i + 3] = a;
  }
}

/** Frames to preload around `i`, nearest first (more ahead while playing). */
export function preloadOrder(i: number, n: number, behind: number, ahead: number): number[] {
  const out: number[] = [];
  const max = Math.max(behind, ahead);
  for (let d = 1; d <= max; d++) {
    if (d <= ahead && i + d < n) out.push(i + d);
    if (d <= behind && i - d >= 0) out.push(i - d);
  }
  return out;
}

/** Frame index under a horizontal position on the timeline. */
export function timelineFrameAt(x: number, width: number, n: number): number {
  if (n <= 0 || width <= 0) return 0;
  return clamp(Math.floor((x / width) * n), 0, n - 1);
}
