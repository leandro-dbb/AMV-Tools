import { describe, expect, it } from 'vitest';
import type { ClipAlternative, ProposedClip } from '../api/music';
import { DICT } from '../i18n';
import {
  MUSIC_STATE_KEY,
  SECTION_TYPES,
  applyAlternative,
  buildConstraints,
  clampZoom,
  clipAt,
  clipToScene,
  clipsKey,
  countLocked,
  energyAt,
  exportExtension,
  formatTime,
  hexToRgba,
  lowerBound,
  msToX,
  forLibrary,
  parseStoredState,
  progressPercent,
  pxPerMs,
  replaceClip,
  safeFilename,
  sectionAt,
  sectionColor,
  sectionLabelKey,
  serializeState,
  toggleLock,
  visibleRange,
  waveformPeak,
  xToMs,
} from './musicUtils';

const clip = (over: Partial<ProposedClip> = {}): ProposedClip => ({
  slot: 0,
  section_index: 0,
  start_ms: 1000,
  end_ms: 2000,
  beats: 2,
  scene_id: 10,
  video_id: 1,
  video_name: 'Ep 1',
  group: 'DBZ',
  src_in_ms: 5000,
  src_out_ms: 6000,
  intensity: 0.4,
  motion_type: 'static',
  target: 0.8,
  score: 0.5,
  locked: false,
  alternatives: [],
  ...over,
});

const alt = (over: Partial<ClipAlternative> = {}): ClipAlternative => ({
  scene_id: 20,
  video_id: 2,
  video_name: 'Ep 2',
  group: 'Naruto',
  src_in_ms: 9000,
  src_out_ms: 10000,
  intensity: 0.9,
  motion_type: 'shake',
  score: 0.7,
  ...over,
});

describe('time formatting', () => {
  it('formats minutes, seconds and tenths', () => {
    expect(formatTime(0)).toBe('0:00');
    expect(formatTime(83_456)).toBe('1:23');
    expect(formatTime(83_456, { tenths: true })).toBe('1:23.4');
    expect(formatTime(3_683_000)).toBe('1:01:23');
  });

  it('is safe with bad input', () => {
    expect(formatTime(-5)).toBe('0:00');
    expect(formatTime(NaN)).toBe('0:00');
    expect(formatTime(undefined)).toBe('0:00');
  });
});

describe('sections', () => {
  it('has a colour and a translated FR/EN label for every section type', () => {
    for (const type of SECTION_TYPES) {
      expect(sectionColor(type)).toMatch(/^#[0-9A-F]{6}$/i);
      const key = sectionLabelKey(type);
      expect(DICT[key]?.fr).toBeTruthy();
      expect(DICT[key]?.en).toBeTruthy();
    }
    expect(new Set(SECTION_TYPES.map(sectionColor)).size).toBe(SECTION_TYPES.length);
  });

  it('falls back for unknown types', () => {
    expect(sectionColor('weird')).toBe('#71717A');
    expect(sectionColor(null)).toBe('#71717A');
    expect(sectionLabelKey('weird')).toBe('music.section.unknown');
    expect(DICT['music.section.unknown']).toBeTruthy();
  });

  it('uses the AMV-friendly French names', () => {
    expect(DICT['music.section.build'].fr).toBe('Montée');
    expect(DICT['music.section.verse'].fr).toBe('Couplet');
    expect(DICT['music.section.chorus'].fr).toBe('Refrain');
    expect(DICT['music.section.bridge'].fr).toBe('Passage calme');
  });

  it('finds the section and the clip under a time', () => {
    const sections = [
      { index: 0, type: 'intro' as const, start_ms: 0, end_ms: 1000, energy: 0.2, drop: false },
      { index: 1, type: 'drop' as const, start_ms: 1000, end_ms: 3000, energy: 0.9, drop: true },
    ];
    expect(sectionAt(sections, 500)?.index).toBe(0);
    expect(sectionAt(sections, 1000)?.index).toBe(1);
    expect(sectionAt(sections, 5000)).toBeNull();
    expect(sectionAt([], 10)).toBeNull();
    expect(sectionAt(undefined, 10)).toBeNull();
    expect(clipAt([clip()], 1500)?.slot).toBe(0);
    expect(clipAt([clip()], 2000)).toBeNull();
    expect(clipAt(null, 0)).toBeNull();
  });

  it('converts hex to rgba', () => {
    expect(hexToRgba('#EF4444', 0.5)).toBe('rgba(239,68,68,0.5)');
    expect(hexToRgba('#fff', 1)).toBe('rgba(255,255,255,1)');
    expect(hexToRgba('nope', 0.2)).toBe('rgba(113,113,122,0.2)');
  });
});

describe('time axis mapping', () => {
  it('maps ms to pixels and back with zoom', () => {
    const ppm = pxPerMs(1000, 1, 100_000);
    expect(ppm).toBeCloseTo(0.01);
    expect(msToX(50_000, ppm)).toBeCloseTo(500);
    expect(xToMs(500, ppm, 100_000)).toBeCloseTo(50_000);
    const zoomed = pxPerMs(1000, 4, 100_000);
    expect(msToX(50_000, zoomed)).toBeCloseTo(2000);
  });

  it('clamps zoom and positions', () => {
    expect(clampZoom(0.2)).toBe(1);
    expect(clampZoom(20)).toBe(8);
    expect(clampZoom(NaN)).toBe(1);
    expect(pxPerMs(1000, 20, 100_000)).toBeCloseTo(0.08);
    expect(pxPerMs(0, 1, 100)).toBe(0);
    expect(pxPerMs(100, 1, 0)).toBe(0);
    expect(xToMs(-10, 0.01, 1000)).toBe(0);
    expect(xToMs(1e9, 0.01, 1000)).toBe(1000);
    expect(xToMs(10, 0, 1000)).toBe(0);
  });

  it('finds the visible beats with a binary search', () => {
    const beats = [0, 500, 1000, 1500, 2000];
    expect(lowerBound(beats, 1000)).toBe(2);
    expect(lowerBound(beats, 1001)).toBe(3);
    expect(lowerBound([], 5)).toBe(0);
    expect(visibleRange(beats, 400, 1500)).toEqual([1, 4]);
    expect(visibleRange(beats, 3000, 4000)).toEqual([5, 5]);
  });

  it('reads waveform peaks and energy', () => {
    const wf = [0.1, 0.9, 0.2, 0.3];
    expect(waveformPeak(wf, 4000, 0, 2000)).toBe(0.9);
    expect(waveformPeak(wf, 4000, 2000, 2100)).toBe(0.2);
    expect(waveformPeak([], 4000, 0, 100)).toBe(0);
    expect(energyAt({ step_ms: 100, values: [0, 1] }, 50)).toBeCloseTo(0.5);
    expect(energyAt({ step_ms: 100, values: [0, 1] }, 500)).toBe(1);
    expect(energyAt({ step_ms: 100, values: [] }, 50)).toBe(0);
    expect(energyAt(null, 50)).toBe(0);
  });
});

describe('editing clips', () => {
  it('swaps a clip for an alternative, keeping its place on the music', () => {
    const a = alt();
    const other = alt({ scene_id: 30 });
    const c = clip({ locked: true, alternatives: [a, other] });
    const next = applyAlternative(c, a);
    expect(next).toMatchObject({
      slot: 0, section_index: 0, start_ms: 1000, end_ms: 2000, beats: 2, target: 0.8, locked: true,
      scene_id: 20, video_id: 2, video_name: 'Ep 2', group: 'Naruto', src_in_ms: 9000, src_out_ms: 10000,
      intensity: 0.9, motion_type: 'shake', score: 0.7,
    });
    expect(next.alternatives.map((x) => x.scene_id)).toEqual([10, 30]);
    // the old shot can be swapped back
    const back = applyAlternative(next, next.alternatives[0]);
    expect(back.scene_id).toBe(10);
    expect(back.alternatives.map((x) => x.scene_id)).toEqual([20, 30]);
    // input not mutated
    expect(c.scene_id).toBe(10);
  });

  it('toggles a lock and replaces a clip by slot', () => {
    const clips = [clip({ slot: 0 }), clip({ slot: 1, start_ms: 2000, end_ms: 3000 })];
    const locked = toggleLock(clips, 1);
    expect(locked[1].locked).toBe(true);
    expect(locked[0].locked).toBe(false);
    expect(clips[1].locked).toBe(false);
    expect(countLocked(locked)).toBe(1);
    expect(countLocked(toggleLock(locked, 1))).toBe(0);
    expect(countLocked(null)).toBe(0);
    const replaced = replaceClip(clips, 0, { ...clips[0], scene_id: 99 });
    expect(replaced[0].scene_id).toBe(99);
    expect(replaced[1]).toBe(clips[1]);
  });

  it('turns a clip into a scene for the mini editor', () => {
    expect(clipToScene(clip())).toMatchObject({ id: 10, video_id: 1, video_display: 'Ep 1', start_ms: 5000, end_ms: 6000, tags: [] });
  });

  it('builds a stable key of the edit', () => {
    expect(clipsKey(null)).toBe('');
    expect(clipsKey([clip()])).toBe(clipsKey([clip()]));
    expect(clipsKey([clip()])).not.toBe(clipsKey([clip({ scene_id: 11 })]));
  });
});

describe('constraints payload', () => {
  it('drops empty fields', () => {
    expect(buildConstraints({ groups: [], characters: [], density: 0.5, avoidReuse: true, query: '  ', sectionQueries: { 2: ' ' } }))
      .toEqual({ density: 0.5, avoid_reuse: true });
  });

  it('keeps filled fields and clamps density', () => {
    expect(buildConstraints({ groups: ['DBZ'], characters: [3], density: 2, avoidReuse: false, query: ' combat ', sectionQueries: { 3: ' Goku ' } }))
      .toEqual({ groups: ['DBZ'], characters: [3], density: 1, avoid_reuse: false, query: 'combat', section_queries: { 3: 'Goku' } });
  });
});

describe('misc helpers', () => {
  it('normalises preview progress', () => {
    expect(progressPercent(0.42)).toBe(42);
    expect(progressPercent(42)).toBe(42);
    expect(progressPercent(1)).toBe(100);
    expect(progressPercent(150)).toBe(100);
    expect(progressPercent(null)).toBe(0);
  });

  it('maps export formats to extensions', () => {
    expect(exportExtension('fcp7xml')).toBe('xml');
    expect(exportExtension('fcpxml')).toBe('fcpxml');
    expect(exportExtension('edl')).toBe('edl');
    expect(exportExtension('otio')).toBe('otio');
    expect(exportExtension('csv')).toBe('csv');
  });

  it('makes safe file names', () => {
    expect(safeFilename('My: song?.mp3')).toBe('My_ song_');
    expect(safeFilename('')).toBe('amv-music');
    expect(safeFilename(null, 'x')).toBe('x');
  });
});

describe('persistence', () => {
  it('uses the agreed localStorage key', () => {
    expect(MUSIC_STATE_KEY).toBe('amv-music-state');
  });

  it('round-trips the state', () => {
    const proposal = { hash: 'abc', bpm: 128, duration_ms: 1000, clips: [clip()], warnings: [], library: { scenes: 5, with_metrics: 5 } };
    const raw = serializeState({ hash: 'abc', proposal, constraints: { groups: ['DBZ'], characters: [], density: 0.7, avoidReuse: false, query: 'x', sectionQueries: { 1: 'Goku' } } });
    const back = parseStoredState(raw);
    expect(back?.hash).toBe('abc');
    expect(back?.proposal?.clips).toHaveLength(1);
    expect(back?.constraints?.density).toBe(0.7);
    expect(back?.constraints?.sectionQueries).toEqual({ 1: 'Goku' });
  });

  it('drops a proposal that belongs to another track and survives garbage', () => {
    const raw = JSON.stringify({ hash: 'abc', proposal: { hash: 'other', clips: [] } });
    expect(parseStoredState(raw)?.proposal).toBeNull();
    expect(parseStoredState('{nope')).toBeNull();
    expect(parseStoredState(null)).toBeNull();
    expect(parseStoredState('42')).toBeNull();
    expect(parseStoredState(JSON.stringify({ hash: 5 }))).toEqual({ hash: null, proposal: null, constraints: undefined, library: null });
  });

  it('drops an edit made on another library (its scene ids point elsewhere)', () => {
    const c = { groups: ['A'], characters: [3], density: 0.7, avoidReuse: true, query: 'fight', sectionQueries: {} };
    const saved = { hash: 'h', proposal: { hash: 'h', clips: [] } as any, constraints: c, library: '/a.db' };
    const moved = forLibrary(saved, '/b.db');
    expect(moved.proposal).toBeNull();
    expect(moved.constraints).toEqual({ ...c, groups: [], characters: [] });
    expect(moved.hash).toBe('h');
    expect(forLibrary(saved, '/a.db').proposal).not.toBeNull();
  });
});
