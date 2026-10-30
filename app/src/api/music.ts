// Music sync API (beat-synced edit proposals from the user's indexed shots).
// Kept out of client.ts so the shared client stays small; same base URL,
// token and error format.
import { authHeaders, getBaseUrl, withToken } from './client';

export type SectionType = 'intro' | 'verse' | 'build' | 'drop' | 'chorus' | 'bridge' | 'outro';

export interface MusicSection {
  index: number;
  type: SectionType;
  start_ms: number;
  end_ms: number;
  /** Mean energy of the section, 0-1. */
  energy: number;
  drop: boolean;
}

export interface MusicAnalysis {
  hash: string;
  name: string;
  path: string;
  duration_ms: number;
  bpm: number;
  meter: number;
  beats_ms: number[];
  downbeats_ms: number[];
  /** where the bar lines come from: 'beat_this' (optional model) or a heuristic */
  bars_engine?: 'beat_this' | 'drops' | 'accents';
  strong_onsets_ms: number[];
  drops_ms: number[];
  sections: MusicSection[];
  /** Smoothed energy curve, one value (0-1) every step_ms. */
  energy: { step_ms: number; values: number[] };
  /** Peak amplitude (0-1) per bucket, evenly spread over the track. */
  waveform: number[];
  analysis_seconds: number;
  cached: boolean;
}

export interface MusicTrackSummary {
  hash: string;
  name: string;
  path: string;
  duration_ms: number;
  bpm: number;
  analyzed_at: number;
}

export interface ClipAlternative {
  scene_id: number;
  video_id: number;
  video_name: string;
  group: string | null;
  src_in_ms: number;
  src_out_ms: number;
  intensity: number | null;
  /** Intensity relative to the library (0 calmest … 1 most intense). */
  intensity_rank?: number | null;
  motion_type: string | null;
  score: number;
}

export interface ProposedClip {
  /** Position in the edit, 0-based, contiguous. */
  slot: number;
  section_index: number;
  /** Placement on the music (exact, on beats / strong onsets). */
  start_ms: number;
  end_ms: number;
  beats: number;
  scene_id: number;
  video_id: number;
  video_name: string;
  group: string | null;
  /** Source range inside the episode, same length as end_ms - start_ms. */
  src_in_ms: number;
  src_out_ms: number;
  intensity: number | null;
  /** Intensity relative to the library (0 calmest … 1 most intense): what the match uses. */
  intensity_rank?: number | null;
  motion_type: string | null;
  /** Target intensity the music asked for at this slot, 0-1. */
  target: number;
  score: number;
  locked: boolean;
  alternatives: ClipAlternative[];
}

export interface ProposalConstraints {
  /** Series (import groups) to draw from; empty = all. */
  groups?: string[];
  /** Character ids that must appear (any of them). */
  characters?: number[];
  /** Cut density: 0 = calm (long shots) … 1 = nervous (short shots). Default 0.5. */
  density?: number;
  avoid_reuse?: boolean;
  /** Free-text wish for the whole edit (searched like the Search tab). */
  query?: string;
  /** Per-section keywords, keyed by section index (e.g. {"3": "Goku"}). */
  section_queries?: Record<string, string>;
}

export interface ProposeBody {
  hash: string;
  constraints?: ProposalConstraints;
  /** Current edit; locked clips are kept as-is. */
  clips?: ProposedClip[];
  /** Only re-pick the clips of these sections (others are kept). Empty/undefined = all unlocked clips. */
  regenerate_sections?: number[];
  seed?: number;
}

export interface EditProposal {
  hash: string;
  bpm: number;
  duration_ms: number;
  clips: ProposedClip[];
  warnings: string[];
  library: { scenes: number; with_metrics: number };
}

export type MusicTimelineFormat = 'fcp7xml' | 'fcpxml' | 'edl' | 'otio';
export type MusicMarkerFormat = MusicTimelineFormat | 'csv';

export interface MusicExportResult {
  ok: boolean;
  output: string;
  format: string;
  clip_count?: number;
  marker_count: number;
  sequence_fps: string;
  warnings: string[];
}

export interface PreviewJob {
  id: string;
  status: 'queued' | 'running' | 'done' | 'failed';
  progress: number;
  message: string;
  output: string | null;
  error: string | null;
  /** Slots whose shot turned out to contain a cut of the episode (off-beat cut). */
  cut_slots?: number[];
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${getBaseUrl()}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...authHeaders(), ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try {
      const parsed = JSON.parse(text);
      if (parsed && typeof parsed.detail === 'string') detail = parsed.detail;
    } catch {}
    throw new Error(detail || `API ${res.status}`);
  }
  return res.json() as Promise<T>;
}

const post = <T>(path: string, body: unknown) =>
  req<T>(path, { method: 'POST', body: JSON.stringify(body) });

export const musicApi = {
  /** Decode + analyse a file (mp3/wav/flac/m4a/ogg or a video's audio). Cached by content hash. */
  analyze: (path: string) => post<MusicAnalysis>('/api/music/analyze', { path }),
  analysis: (hash: string) => req<MusicAnalysis>(`/api/music/tracks/${hash}`),
  tracks: () => req<{ tracks: MusicTrackSummary[] }>('/api/music/tracks'),
  /** Streams the analysed audio file (for the in-app player). */
  audioUrl: (hash: string) => withToken(`${getBaseUrl()}/api/music/tracks/${hash}/audio`),
  propose: (body: ProposeBody) => post<EditProposal>('/api/music/propose', body),
  exportTimeline: (body: {
    hash: string;
    clips: ProposedClip[];
    format: MusicTimelineFormat;
    output_path: string;
    beat_markers?: boolean;
    include_source_audio?: boolean;
    lang?: 'en' | 'fr';
  }) => post<MusicExportResult>('/api/music/export/timeline', body),
  exportMarkers: (body: {
    hash: string;
    format: MusicMarkerFormat;
    output_path: string;
    beat_markers?: boolean;
    fps?: string;
    lang?: 'en' | 'fr';
  }) => post<MusicExportResult>('/api/music/export/markers', body),
  startPreview: (body: { hash: string; clips: ProposedClip[] }) =>
    post<PreviewJob>('/api/music/preview', body),
  previewJob: (id: string) => req<PreviewJob>(`/api/music/preview/${id}`),
  previewUrl: (id: string) => withToken(`${getBaseUrl()}/api/music/preview/${id}/video`),
};
