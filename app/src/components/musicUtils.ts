// Pure helpers for the Music tab (no React, no DOM) so they can be unit-tested.
import type {
  ClipAlternative,
  EditProposal,
  MusicMarkerFormat,
  MusicSection,
  ProposalConstraints,
  ProposedClip,
  SectionType,
} from '../api/music';
import type { SceneResult } from '../api/types';

export const SECTION_TYPES: SectionType[] = ['intro', 'verse', 'build', 'drop', 'chorus', 'bridge', 'outro'];

/** One colour per section type (hex, used for bands, chips and the legend). */
export const SECTION_COLORS: Record<SectionType, string> = {
  intro: '#38BDF8',   // sky
  verse: '#6366F1',   // indigo
  build: '#F59E0B',   // amber
  drop: '#EF4444',    // red
  chorus: '#EC4899',  // pink
  bridge: '#14B8A6',  // teal
  outro: '#A1A1AA',   // grey
};

const FALLBACK_COLOR = '#71717A';

export function sectionColor(type: string | null | undefined): string {
  return (type && (SECTION_COLORS as Record<string, string>)[type]) || FALLBACK_COLOR;
}

export function sectionLabelKey(type: string | null | undefined): string {
  return type && (SECTION_TYPES as string[]).includes(type) ? `music.section.${type}` : 'music.section.unknown';
}

/** "#RRGGBB" + alpha (0-1) → "rgba(r,g,b,a)". */
export function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace('#', '');
  const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  const n = parseInt(full, 16);
  if (!Number.isFinite(n) || full.length !== 6) return `rgba(113,113,122,${alpha})`;
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

/** 83_456 → "1:23" ; with tenths → "1:23.4" ; ≥ 1 h → "1:01:23". */
export function formatTime(ms: number | null | undefined, opts: { tenths?: boolean } = {}): string {
  if (ms == null || !Number.isFinite(ms) || ms < 0) ms = 0;
  const totalTenths = Math.floor(ms / 100);
  const totalSec = Math.floor(totalTenths / 10);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  const base = h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${m}:${String(s).padStart(2, '0')}`;
  return opts.tenths ? `${base}.${totalTenths % 10}` : base;
}

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 8;

export function clampZoom(z: number): number {
  if (!Number.isFinite(z)) return MIN_ZOOM;
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z));
}

/** Pixels per millisecond for a viewport `width` px wide showing the whole track at zoom 1. */
export function pxPerMs(width: number, zoom: number, durationMs: number): number {
  if (!(width > 0) || !(durationMs > 0)) return 0;
  return (width * clampZoom(zoom)) / durationMs;
}

export function msToX(ms: number, ppm: number): number {
  return ms * ppm;
}

export function xToMs(x: number, ppm: number, durationMs: number): number {
  if (!(ppm > 0)) return 0;
  return Math.min(Math.max(0, x / ppm), Math.max(0, durationMs));
}

/** First index i with arr[i] >= v (arr sorted ascending). */
export function lowerBound(arr: number[], v: number): number {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < v) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/** [start, end) index range of the sorted values falling inside [lo, hi]. */
export function visibleRange(arr: number[], lo: number, hi: number): [number, number] {
  return [lowerBound(arr, lo), lowerBound(arr, hi + 1e-9)];
}

/** Max peak of the waveform buckets covering [startMs, endMs). */
export function waveformPeak(waveform: number[], durationMs: number, startMs: number, endMs: number): number {
  const n = waveform.length;
  if (n === 0 || !(durationMs > 0)) return 0;
  let a = Math.floor((startMs / durationMs) * n);
  let b = Math.ceil((endMs / durationMs) * n);
  a = Math.max(0, Math.min(n - 1, a));
  b = Math.max(a + 1, Math.min(n, b));
  let peak = 0;
  for (let i = a; i < b; i++) if (waveform[i] > peak) peak = waveform[i];
  return peak;
}

/** Energy value (0-1) at a given time, linearly interpolated. */
export function energyAt(energy: { step_ms: number; values: number[] } | null | undefined, ms: number): number {
  if (!energy || !energy.values?.length || !(energy.step_ms > 0)) return 0;
  const f = ms / energy.step_ms;
  const i = Math.floor(f);
  const v = energy.values;
  if (f <= 0) return v[0];
  if (i >= v.length - 1) return v[v.length - 1];
  return v[i] + (v[i + 1] - v[i]) * (f - i);
}

export function sectionAt(sections: MusicSection[] | null | undefined, ms: number): MusicSection | null {
  if (!sections) return null;
  return sections.find((s) => ms >= s.start_ms && ms < s.end_ms) ?? null;
}

export function clipAt(clips: ProposedClip[] | null | undefined, ms: number): ProposedClip | null {
  if (!clips) return null;
  return clips.find((c) => ms >= c.start_ms && ms < c.end_ms) ?? null;
}

export function clipToAlternative(c: ProposedClip | ClipAlternative): ClipAlternative {
  return {
    scene_id: c.scene_id,
    video_id: c.video_id,
    video_name: c.video_name,
    group: c.group,
    src_in_ms: c.src_in_ms,
    src_out_ms: c.src_out_ms,
    intensity: c.intensity,
    motion_type: c.motion_type,
    score: c.score,
  };
}

/** Swap a clip for one of its alternatives. Placement on the music (start/end,
 *  beats, section, slot, target, lock) is kept; the shot comes from the
 *  alternative; the previous shot goes to the front of the alternatives. */
export function applyAlternative(clip: ProposedClip, alt: ClipAlternative): ProposedClip {
  const others = (clip.alternatives ?? []).filter(
    (a) => !(a.scene_id === alt.scene_id && a.src_in_ms === alt.src_in_ms && a.src_out_ms === alt.src_out_ms),
  );
  return {
    ...clip,
    scene_id: alt.scene_id,
    video_id: alt.video_id,
    video_name: alt.video_name,
    group: alt.group,
    src_in_ms: alt.src_in_ms,
    src_out_ms: alt.src_out_ms,
    intensity: alt.intensity,
    motion_type: alt.motion_type,
    score: alt.score,
    alternatives: [clipToAlternative(clip), ...others],
  };
}

export function replaceClip(clips: ProposedClip[], slot: number, next: ProposedClip): ProposedClip[] {
  return clips.map((c) => (c.slot === slot ? next : c));
}

export function toggleLock(clips: ProposedClip[], slot: number): ProposedClip[] {
  return clips.map((c) => (c.slot === slot ? { ...c, locked: !c.locked } : c));
}

export function setSectionLock(clips: ProposedClip[], sectionIndex: number, locked: boolean): ProposedClip[] {
  return clips.map((c) => (c.section_index === sectionIndex ? { ...c, locked } : c));
}

export function countLocked(clips: ProposedClip[] | null | undefined): number {
  return clips ? clips.filter((c) => c.locked).length : 0;
}

/** Turn the UI constraint state into the API payload, dropping empty bits. */
export function buildConstraints(c: {
  groups: string[];
  characters: number[];
  density: number;
  avoidReuse: boolean;
  query: string;
  sectionQueries: Record<string, string>;
}): ProposalConstraints {
  const out: ProposalConstraints = {
    density: Math.min(1, Math.max(0, Number.isFinite(c.density) ? c.density : 0.5)),
    avoid_reuse: c.avoidReuse,
  };
  if (c.groups.length) out.groups = [...c.groups];
  if (c.characters.length) out.characters = [...c.characters];
  const q = c.query.trim();
  if (q) out.query = q;
  const sq: Record<string, string> = {};
  for (const [k, v] of Object.entries(c.sectionQueries ?? {})) {
    const s = (v ?? '').trim();
    if (s) sq[k] = s;
  }
  if (Object.keys(sq).length) out.section_queries = sq;
  return out;
}

/** Preview progress may come as 0-1 or 0-100; returns a 0-100 integer. */
export function progressPercent(p: number | null | undefined): number {
  if (p == null || !Number.isFinite(p) || p <= 0) return 0;
  const pct = p <= 1 ? p * 100 : p;
  return Math.min(100, Math.round(pct));
}

export function exportExtension(format: MusicMarkerFormat): string {
  switch (format) {
    case 'fcp7xml': return 'xml';
    case 'fcpxml': return 'fcpxml';
    case 'edl': return 'edl';
    case 'otio': return 'otio';
    case 'csv': return 'csv';
    default: return 'xml';
  }
}

export function safeFilename(name: string | null | undefined, fallback = 'amv-music'): string {
  const stem = (name ?? '').replace(/\.[a-z0-9]{2,5}$/i, '');
  return stem.replace(/[\\/:*?"<>|]+/g, '_').trim() || fallback;
}

/** Last path segment, for display. */
export function basename(p: string | null | undefined): string {
  if (!p) return '';
  return p.split(/[\\/]/).pop() || p;
}

/** Parent folder of a path ('' when none). */
export function dirname(p: string | null | undefined): string {
  if (!p) return '';
  const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'));
  return i > 0 ? p.slice(0, i) : '';
}

/** Minimal SceneResult so a proposed shot can open in the mini editor. */
export function clipToScene(c: ProposedClip | ClipAlternative): SceneResult {
  return {
    id: c.scene_id,
    video_id: c.video_id,
    video_display: c.video_name,
    scene_index: 0,
    start_ms: c.src_in_ms,
    end_ms: c.src_out_ms,
    score: c.score,
    tags: [],
  };
}

export function randomSeed(): number {
  return Math.floor(Math.random() * 2_147_483_647);
}

/** Stable key of an edit, to know whether a preview still matches it. */
export function clipsKey(clips: ProposedClip[] | null | undefined): string {
  if (!clips?.length) return '';
  return clips.map((c) => `${c.start_ms}-${c.end_ms}:${c.scene_id}@${c.src_in_ms}`).join('|');
}

// ── persistence ────────────────────────────────────────────────────────────

export const MUSIC_STATE_KEY = 'amv-music-state';

export interface StoredConstraints {
  groups: string[];
  characters: number[];
  density: number;
  avoidReuse: boolean;
  query: string;
  sectionQueries: Record<string, string>;
}

export interface StoredMusicState {
  hash: string | null;
  proposal: EditProposal | null;
  constraints?: StoredConstraints;
}

export const DEFAULT_CONSTRAINTS: StoredConstraints = {
  groups: [],
  characters: [],
  density: 0.5,
  avoidReuse: true,
  query: '',
  sectionQueries: {},
};

export function parseStoredState(raw: string | null | undefined): StoredMusicState | null {
  if (!raw) return null;
  try {
    const v = JSON.parse(raw);
    if (!v || typeof v !== 'object') return null;
    const hash = typeof v.hash === 'string' && v.hash ? v.hash : null;
    const proposal = v.proposal && typeof v.proposal === 'object' && Array.isArray(v.proposal.clips)
      && (!hash || v.proposal.hash === hash) ? (v.proposal as EditProposal) : null;
    let constraints: StoredConstraints | undefined;
    if (v.constraints && typeof v.constraints === 'object') {
      const c = v.constraints;
      constraints = {
        groups: Array.isArray(c.groups) ? c.groups.filter((x: unknown) => typeof x === 'string') : [],
        characters: Array.isArray(c.characters) ? c.characters.filter((x: unknown) => typeof x === 'number') : [],
        density: typeof c.density === 'number' ? c.density : 0.5,
        avoidReuse: typeof c.avoidReuse === 'boolean' ? c.avoidReuse : true,
        query: typeof c.query === 'string' ? c.query : '',
        sectionQueries: c.sectionQueries && typeof c.sectionQueries === 'object' ? c.sectionQueries : {},
      };
    }
    return { hash, proposal, constraints };
  } catch {
    return null;
  }
}

export function serializeState(s: StoredMusicState): string {
  return JSON.stringify(s);
}

/** i18n key + vars for a backend warning code ("reused:12", "empty_library"…). */
export function warningKey(code: string): { key: string; vars: Record<string, string | number> } {
  const [name, count] = String(code).split(':', 2);
  const known = ['reused', 'repeats', 'unfilled', 'metrics_missing', 'empty_library', 'no_shots_in_groups',
    'no_character_matches', 'query_tags_only', 'edl_no_markers'];
  if (known.includes(name)) return { key: `music.warn.${name}`, vars: { n: count ?? '' } };
  return { key: 'music.warn.other', vars: { code } };
}
