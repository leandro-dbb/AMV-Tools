// Music tab: analyse a music, get a beat-synced edit proposal from the
// indexed shots, tweak it, preview it and export it as a timeline.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle, Database, Film, Loader2, Minus, Music2, Pause, Play, Plus, RefreshCw, SkipBack, Sparkles, Wand2,
} from 'lucide-react';
import { api } from '../api/client';
import { musicApi } from '../api/music';
import type { ClipAlternative, EditProposal, MusicAnalysis, MusicTrackSummary, PreviewJob, ProposedClip } from '../api/music';
import type { SceneResult, SearchFacets } from '../api/types';
import { useT } from '../i18n';
import MusicImport from './MusicImport';
import MusicTimeline from './MusicTimeline';
import MusicConstraints from './MusicConstraints';
import MusicClipPanel from './MusicClipPanel';
import MusicExportPanel from './MusicExportPanel';
import {
  DEFAULT_CONSTRAINTS, MAX_ZOOM, MIN_ZOOM, MUSIC_STATE_KEY, SECTION_COLORS, SECTION_TYPES,
  applyAlternative, buildConstraints, clampZoom, clipToScene, clipsKey, countLocked, formatTime,
  parseStoredState, progressPercent, randomSeed, replaceClip, sectionLabelKey, serializeState, toggleLock, warningKey,
} from './musicUtils';
import type { StoredConstraints } from './musicUtils';

interface Props {
  /** Open shots in the shared mini editor (same contract as DerushTab). */
  onOpenScene?: (scenes: SceneResult[], index: number) => void;
  /** Jump to the Library tab (empty-library state). */
  onOpenLibrary?: () => void;
}

type MediaSource = 'audio' | 'video';

function readStored() {
  try { return parseStoredState(localStorage.getItem(MUSIC_STATE_KEY)); } catch { return null; }
}

export default function MusicTab({ onOpenScene, onOpenLibrary }: Props) {
  const t = useT();
  const [initial] = useState(readStored);

  // track
  const [hash, setHash] = useState<string | null>(initial?.hash ?? null);
  const [analysis, setAnalysis] = useState<MusicAnalysis | null>(null);
  const [loadingTrack, setLoadingTrack] = useState<boolean>(!!initial?.hash);
  const [analyzing, setAnalyzing] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const [tracks, setTracks] = useState<MusicTrackSummary[]>([]);
  const [showImport, setShowImport] = useState(false);

  // edit
  const [proposal, setProposal] = useState<EditProposal | null>(initial?.proposal ?? null);
  const [constraints, setConstraints] = useState<StoredConstraints>(initial?.constraints ?? DEFAULT_CONSTRAINTS);
  const [proposing, setProposing] = useState<'all' | 'section' | null>(null);
  const [proposeError, setProposeError] = useState<string | null>(null);
  const [facets, setFacets] = useState<SearchFacets | null>(null);
  const [libraryScenes, setLibraryScenes] = useState<number | null>(null);
  const [selectedSlot, setSelectedSlot] = useState<number | null>(null);
  const [selectedSection, setSelectedSection] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);

  // playback
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [source, setSource] = useState<MediaSource>('audio');
  const sourceRef = useRef<MediaSource>('audio');
  sourceRef.current = source;
  const [playing, setPlaying] = useState(false);
  const [currentMs, setCurrentMs] = useState(0);
  const [audioError, setAudioError] = useState(false);

  // preview
  const [job, setJob] = useState<PreviewJob | null>(null);
  const [jobKey, setJobKey] = useState('');
  const [previewError, setPreviewError] = useState<string | null>(null);

  const clips = proposal?.clips ?? null;
  const sections = analysis?.sections ?? [];
  const previewReady = job?.status === 'done';

  // ── persistence ─────────────────────────────────────────────────────────
  useEffect(() => {
    try {
      localStorage.setItem(MUSIC_STATE_KEY, serializeState({ hash, proposal, constraints }));
    } catch {}
  }, [hash, proposal, constraints]);

  // ── initial loads ───────────────────────────────────────────────────────
  const refreshTracks = useCallback(() => {
    musicApi.tracks().then((r) => setTracks(r?.tracks ?? [])).catch(() => {});
  }, []);

  useEffect(() => {
    refreshTracks();
    api.searchFacets().then(setFacets).catch(() => {});
    api.listVideos()
      .then((r) => setLibraryScenes((r.videos ?? []).reduce((acc, v) => acc + (v.scene_count || 0), 0)))
      .catch(() => {});
  }, [refreshTracks]);

  useEffect(() => {
    if (!initial?.hash) return;
    let alive = true;
    musicApi.analysis(initial.hash)
      .then((a) => { if (alive) setAnalysis(a); })
      .catch((e) => {
        if (!alive) return;
        setImportError(t('music.import.loadFailed', { message: (e as Error).message }));
        setHash(null);
        setProposal(null);
      })
      .finally(() => { if (alive) setLoadingTrack(false); });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── track loading ───────────────────────────────────────────────────────
  const adoptAnalysis = useCallback((a: MusicAnalysis) => {
    if (a.hash !== hash) {
      setProposal(null);
      setConstraints((c) => ({ ...c, sectionQueries: {} }));
      setJob(null);
      setPreviewError(null);
    }
    audioRef.current?.pause();
    videoRef.current?.pause();
    setSource('audio');
    setPlaying(false);
    setCurrentMs(0);
    setAudioError(false);
    setSelectedSlot(null);
    setSelectedSection(null);
    setProposeError(null);
    setZoom(1);
    setAnalysis(a);
    setHash(a.hash);
    setShowImport(false);
    setImportError(null);
  }, [hash]);

  const analyzePath = useCallback(async (path: string) => {
    setImportError(null);
    setAnalyzing(true);
    try {
      const a = await musicApi.analyze(path);
      adoptAnalysis(a);
      refreshTracks();
    } catch (e) {
      setImportError(t('music.import.failed', { message: (e as Error).message }));
    } finally {
      setAnalyzing(false);
    }
  }, [adoptAnalysis, refreshTracks, t]);

  const openTrack = useCallback(async (h: string) => {
    setImportError(null);
    setAnalyzing(true);
    try {
      adoptAnalysis(await musicApi.analysis(h));
    } catch (e) {
      setImportError(t('music.import.loadFailed', { message: (e as Error).message }));
    } finally {
      setAnalyzing(false);
    }
  }, [adoptAnalysis, t]);

  // ── playback ────────────────────────────────────────────────────────────
  const activeEl = useCallback((): HTMLMediaElement | null => {
    if (sourceRef.current === 'video' && videoRef.current) return videoRef.current;
    return audioRef.current;
  }, []);

  const getTimeMs = useCallback(() => (activeEl()?.currentTime ?? 0) * 1000, [activeEl]);

  const seek = useCallback((ms: number) => {
    const el = activeEl();
    if (el) {
      try { el.currentTime = ms / 1000; } catch {}
    }
    setCurrentMs(ms);
  }, [activeEl]);

  const togglePlay = useCallback(() => {
    const el = activeEl();
    if (!el) return;
    if (el.paused) el.play().catch(() => setAudioError(sourceRef.current === 'audio'));
    else el.pause();
  }, [activeEl]);

  const switchSource = useCallback((next: MediaSource) => {
    if (next === sourceRef.current) return;
    const from = activeEl();
    const wasPlaying = !!from && !from.paused;
    audioRef.current?.pause();
    videoRef.current?.pause();
    sourceRef.current = next;
    setSource(next);
    const to = next === 'video' ? videoRef.current : audioRef.current;
    if (to) {
      try { to.currentTime = currentMs / 1000; } catch {}
      if (wasPlaying) to.play().catch(() => {});
    }
  }, [activeEl, currentMs]);

  const mediaHandlers = (kind: MediaSource) => ({
    onPlay: () => {
      if (kind !== sourceRef.current) {
        // the user pressed play on the other element (e.g. native video controls)
        (kind === 'video' ? audioRef.current : videoRef.current)?.pause();
        sourceRef.current = kind;
        setSource(kind);
      }
      setPlaying(true);
    },
    onPause: () => { if (kind === sourceRef.current) setPlaying(false); },
    onEnded: () => { if (kind === sourceRef.current) setPlaying(false); },
    onTimeUpdate: (e: React.SyntheticEvent<HTMLMediaElement>) => {
      if (kind === sourceRef.current) setCurrentMs(e.currentTarget.currentTime * 1000);
    },
    onSeeked: (e: React.SyntheticEvent<HTMLMediaElement>) => {
      if (kind === sourceRef.current) setCurrentMs(e.currentTarget.currentTime * 1000);
    },
  });

  // Space = play / pause (outside text fields)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || !analysis) return;
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'BUTTON' || tag === 'VIDEO' || el?.isContentEditable) return;
      e.preventDefault();
      togglePlay();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [analysis, togglePlay]);

  // ── proposals ───────────────────────────────────────────────────────────
  const runPropose = useCallback(async (mode: 'fresh' | 'regen' | 'section', sectionIndex?: number) => {
    if (!hash || proposing) return;
    setProposeError(null);
    setProposing(mode === 'section' ? 'section' : 'all');
    try {
      const body = {
        hash,
        constraints: buildConstraints(constraints),
        seed: randomSeed(),
        ...(mode !== 'fresh' && clips ? { clips } : {}),
        ...(mode === 'section' && sectionIndex != null ? { regenerate_sections: [sectionIndex] } : {}),
      };
      const p = await musicApi.propose(body);
      const next: EditProposal = { ...p, clips: p.clips ?? [], warnings: p.warnings ?? [] };
      setProposal(next);
      setSelectedSlot((s) => (s != null && next.clips.some((c) => c.slot === s) ? s : null));
      if (p.library && typeof p.library.scenes === 'number') setLibraryScenes(p.library.scenes);
    } catch (e) {
      setProposeError(t('music.proposeFailed', { message: (e as Error).message }));
    } finally {
      setProposing(null);
    }
  }, [hash, proposing, constraints, clips, t]);

  const updateClips = useCallback((fn: (c: ProposedClip[]) => ProposedClip[]) => {
    setProposal((p) => (p ? { ...p, clips: fn(p.clips ?? []) } : p));
  }, []);

  const selectedClip = useMemo(
    () => (selectedSlot != null && clips ? clips.find((c) => c.slot === selectedSlot) ?? null : null),
    [clips, selectedSlot],
  );

  const pickAlternative = (alt: ClipAlternative) => {
    if (!selectedClip) return;
    updateClips((cs) => replaceClip(cs, selectedClip.slot, applyAlternative(selectedClip, alt)));
  };

  const onSelectClip = useCallback((slot: number) => {
    setSelectedSlot((s) => (s === slot ? null : slot));
  }, []);

  const onSelectSection = useCallback((index: number | null) => {
    setSelectedSection((s) => (index == null || s === index ? null : index));
  }, []);

  // ── preview ─────────────────────────────────────────────────────────────
  async function startPreview() {
    if (!hash || !clips || clips.length === 0) return;
    setPreviewError(null);
    switchSource('audio');
    try {
      const j = await musicApi.startPreview({ hash, clips });
      setJobKey(clipsKey(clips));
      setJob(j);
      if (j.status === 'failed') setPreviewError(j.error || j.message || '?');
    } catch (e) {
      setPreviewError((e as Error).message);
    }
  }

  useEffect(() => {
    if (!job || job.status === 'done' || job.status === 'failed') return;
    const id = window.setInterval(() => {
      musicApi.previewJob(job.id)
        .then((j) => {
          setJob(j);
          if (j.status === 'failed') setPreviewError(j.error || j.message || '?');
        })
        .catch((e) => { setPreviewError((e as Error).message); setJob(null); });
    }, 1000);
    return () => window.clearInterval(id);
  }, [job?.id, job?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  // a finished preview becomes the player
  useEffect(() => {
    if (job?.status !== 'done') return;
    audioRef.current?.pause();
    sourceRef.current = 'video';
    setSource('video');
    setPlaying(false);
  }, [job?.id, job?.status]);

  const previewStale = previewReady && jobKey !== clipsKey(clips);
  // Shots the preview found to contain a cut of the episode (an off-beat
  // cut): flagged in the strip until the edit changes.
  const flaggedSlots = useMemo(
    () => new Set<number>(job?.status === 'done' && !previewStale ? job.cut_slots ?? [] : []),
    [job, previewStale],
  );

  // ── render ──────────────────────────────────────────────────────────────
  const libraryEmpty = libraryScenes === 0 || (proposal?.library?.scenes === 0);
  const lockedCount = countLocked(clips);
  const beatsCount = analysis?.beats_ms?.length ?? 0;
  const dropsCount = analysis?.drops_ms?.length ?? 0;
  const selectedClipSection = selectedClip ? sections.find((s) => s.index === selectedClip.section_index) ?? null : null;

  return (
    <div className="h-full overflow-auto relative">
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[500px] h-[500px] bg-[#EC4899]/5 rounded-full blur-[120px] pointer-events-none" />

      <div className="max-w-[1500px] mx-auto p-6 relative z-10 flex flex-col gap-5">
        <div className="flex items-end gap-4">
          <div className="flex-1 min-w-0">
            <h2 className="text-3xl font-bold bg-gradient-to-r from-[#FAFAFA] to-[#A1A1AA] bg-clip-text text-transparent">{t('music.title')}</h2>
            <div className="h-1 w-20 bg-gradient-to-r from-[#EC4899] to-[#8B5CF6] rounded-full mt-2 mb-2" />
            <div className="text-sm text-[#A1A1AA] max-w-3xl">{t('music.subtitle')}</div>
          </div>
          {analysis && (
            <button type="button" onClick={() => setShowImport((v) => !v)}
                    className="flex items-center gap-2 bg-[#18181B] border-2 border-[#27272A] hover:border-[#EC4899]/50 px-4 py-2 rounded-xl text-sm text-[#FAFAFA] transition-all shrink-0">
              <Music2 size={14} className="text-[#EC4899]" /> {t('music.import.change')}
            </button>
          )}
        </div>

        {loadingTrack && !analysis && (
          <div className="flex items-center gap-2 text-sm text-[#A1A1AA]"><Loader2 size={16} className="animate-spin text-[#EC4899]" /> {t('music.import.analyzing')}</div>
        )}

        {(!analysis || showImport || analyzing) && !loadingTrack && (
          <MusicImport
            analyzing={analyzing}
            error={importError}
            tracks={tracks}
            currentHash={analysis?.hash ?? null}
            onAnalyze={analyzePath}
            onOpenTrack={openTrack}
            onError={(m) => setImportError(m)}
            compact={!!analysis}
          />
        )}

        {analysis && (
          <>
            {/* ── overview + timeline ─────────────────────────────────── */}
            <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-4 flex flex-col gap-3">
              <div className="flex items-center gap-4 flex-wrap">
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <Music2 size={18} className="text-[#EC4899] shrink-0" />
                  <span className="text-lg font-bold text-[#FAFAFA] truncate" title={analysis.path}>{analysis.name || '—'}</span>
                </div>
                <Stat label={t('music.overview.tempo')} value={analysis.bpm ? t('music.overview.bpm', { bpm: Math.round(analysis.bpm) }) : '—'} />
                <Stat label={t('music.overview.duration')} value={formatTime(analysis.duration_ms)} />
                <Stat label={t('music.overview.beats')} value={String(beatsCount)} />
                <Stat label={t('music.overview.sections')} value={String(sections.length)} />
                <Stat label={t('music.overview.drops')} value={String(dropsCount)} accent={dropsCount > 0 ? '#EF4444' : undefined} />
              </div>

              {/* player + zoom */}
              <div className="flex items-center gap-2 flex-wrap">
                <button type="button" onClick={togglePlay} title={playing ? t('music.player.pause') : t('music.player.play')}
                        className="w-9 h-9 rounded-full flex items-center justify-center bg-gradient-to-br from-[#EC4899] to-[#8B5CF6] text-white hover:shadow-lg hover:shadow-[#EC4899]/30">
                  {playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" className="ml-0.5" />}
                </button>
                <button type="button" onClick={() => seek(0)} title={t('music.player.toStart')}
                        className="p-2 rounded-lg text-[#A1A1AA] hover:text-[#FAFAFA] hover:bg-[#27272A]">
                  <SkipBack size={15} />
                </button>
                <span className="font-mono text-sm text-[#FAFAFA] tabular-nums">{formatTime(currentMs, { tenths: true })}</span>
                <span className="font-mono text-sm text-[#52525B]">/ {formatTime(analysis.duration_ms)}</span>
                <span className="ml-2 text-[11px] text-[#71717A] flex items-center gap-1">
                  {source === 'video' ? <Film size={12} /> : <Music2 size={12} />}
                  {source === 'video' ? t('music.player.source.preview') : t('music.player.source.music')}
                </span>
                {audioError && source === 'audio' && (
                  <span className="text-[11px] text-amber-300 flex items-center gap-1"><AlertTriangle size={12} /> {t('music.player.audioError')}</span>
                )}
                <span className="flex-1" />
                <span className="text-[11px] uppercase tracking-wider text-[#71717A]">{t('music.zoom')}</span>
                <button type="button" onClick={() => setZoom((z) => clampZoom(z / 1.5))} disabled={zoom <= MIN_ZOOM} title={t('music.zoomOut')}
                        className="p-1.5 rounded-md text-[#A1A1AA] hover:text-[#FAFAFA] hover:bg-[#27272A] disabled:opacity-30">
                  <Minus size={14} />
                </button>
                <input type="range" min={MIN_ZOOM} max={MAX_ZOOM} step={0.25} value={zoom}
                       onChange={(e) => setZoom(clampZoom(Number(e.target.value)))} className="w-32 accent-[#8B5CF6]" />
                <button type="button" onClick={() => setZoom((z) => clampZoom(z * 1.5))} disabled={zoom >= MAX_ZOOM} title={t('music.zoomIn')}
                        className="p-1.5 rounded-md text-[#A1A1AA] hover:text-[#FAFAFA] hover:bg-[#27272A] disabled:opacity-30">
                  <Plus size={14} />
                </button>
                <span className="font-mono text-xs text-[#A1A1AA] w-10">{zoom.toFixed(zoom % 1 ? 1 : 0)}x</span>
              </div>

              <MusicTimeline
                analysis={analysis}
                clips={clips}
                zoom={zoom}
                currentMs={currentMs}
                playing={playing}
                getTimeMs={getTimeMs}
                onSeek={seek}
                selectedSlot={selectedSlot}
                onSelectClip={onSelectClip}
                selectedSection={selectedSection}
                onSelectSection={onSelectSection}
                sectionQueries={constraints.sectionQueries}
                emptyStrip={proposal && (clips?.length ?? 0) === 0 ? t('music.edit.noClips') : t('music.edit.empty')}
                flaggedSlots={flaggedSlots}
              />

              {/* legend */}
              <div className="flex items-center gap-x-4 gap-y-1.5 flex-wrap text-[11px] text-[#A1A1AA]">
                {SECTION_TYPES.map((type) => (
                  <span key={type} className="inline-flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-sm" style={{ background: SECTION_COLORS[type] }} />
                    {t(sectionLabelKey(type))}
                  </span>
                ))}
                <span className="w-px h-3 bg-[#3F3F46]" />
                <span className="inline-flex items-center gap-1.5"><span className="w-4 h-0.5 bg-[#FBBF24] rounded" />{t('music.legend.energy')}</span>
                <span className="inline-flex items-center gap-1.5"><span className="w-px h-2 bg-[#FAFAFA]/50" />{t('music.legend.beat')}</span>
                <span className="inline-flex items-center gap-1.5"><span className="w-px h-3 bg-[#FAFAFA]" />{t('music.legend.bar')}</span>
                <span className="inline-flex items-center gap-1.5"><span className="w-0.5 h-3 bg-[#EF4444]" />{t('music.legend.drop')}</span>
                <span className="flex-1" />
                <span className="text-[#71717A]">{t('music.sections.hint')}</span>
              </div>

              {beatsCount === 0 && (
                <div className="text-xs text-amber-300 flex items-center gap-1.5"><AlertTriangle size={13} /> {t('music.overview.noBeats')}</div>
              )}

              <audio
                ref={audioRef}
                src={musicApi.audioUrl(analysis.hash)}
                preload="auto"
                onError={() => setAudioError(true)}
                onCanPlay={() => setAudioError(false)}
                {...mediaHandlers('audio')}
              />
            </div>

            {/* ── actions ─────────────────────────────────────────────── */}
            <div className="flex items-center gap-3 flex-wrap">
              <button type="button" onClick={() => runPropose('fresh')} disabled={!!proposing}
                      className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold bg-gradient-to-r from-[#EC4899] to-[#8B5CF6] text-white hover:shadow-lg hover:shadow-[#EC4899]/30 disabled:opacity-50 disabled:cursor-not-allowed">
                {proposing === 'all' ? <Loader2 size={15} className="animate-spin" /> : <Wand2 size={15} />}
                {proposing === 'all' ? t('music.proposing') : t('music.propose')}
              </button>
              {clips && clips.length > 0 && (
                <button type="button" onClick={() => runPropose('regen')} disabled={!!proposing} title={t('music.regenerateHint')}
                        className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold bg-[#18181B] border-2 border-[#27272A] hover:border-[#8B5CF6]/60 text-[#FAFAFA] disabled:opacity-50">
                  <RefreshCw size={14} /> {t('music.regenerate')}
                </button>
              )}
              {clips && clips.length > 0 && (
                <span className="text-xs text-[#A1A1AA]">{t('music.edit.summary', { count: clips.length, locked: lockedCount })}</span>
              )}
              {clips && clips.length > 0 && selectedSlot == null && (
                <span className="text-xs text-[#71717A]">{t('music.clip.pick')}</span>
              )}
              {proposeError && (
                <span className="text-xs text-red-300 flex items-center gap-1.5 break-words"><AlertTriangle size={13} /> {proposeError}</span>
              )}
            </div>

            {libraryEmpty && (
              <div className="bg-[#F59E0B]/10 border-2 border-[#F59E0B]/40 rounded-xl p-4 flex items-start gap-3">
                <Database size={20} className="text-[#F59E0B] shrink-0 mt-0.5" />
                <div className="flex-1">
                  <div className="text-sm font-semibold text-[#FAFAFA]">{t('music.library.empty.title')}</div>
                  <div className="text-xs text-[#D4D4D8] mt-1">{t('music.library.empty.body')}</div>
                </div>
                {onOpenLibrary && (
                  <button type="button" onClick={onOpenLibrary}
                          className="shrink-0 bg-gradient-to-r from-[#F59E0B] to-[#EC4899] text-white px-4 py-2 rounded-lg text-xs font-semibold">
                    {t('music.library.open')}
                  </button>
                )}
              </div>
            )}

            {proposal && (proposal.warnings ?? []).length > 0 && (
              <div className="bg-amber-500/10 border border-amber-500/40 rounded-lg px-3 py-2">
                <div className="text-[11px] uppercase tracking-wider text-amber-300 mb-1">{t('music.edit.warnings')}</div>
                <ul className="text-xs text-amber-200 list-disc pl-4 space-y-0.5">
                  {proposal.warnings.map((w, i) => { const wk = warningKey(w); return <li key={i}>{t(wk.key, wk.vars)}</li>; })}
                </ul>
              </div>
            )}

            {selectedClip && clips && (
              <MusicClipPanel
                clip={selectedClip}
                section={selectedClipSection}
                total={clips.length}
                busy={!!proposing}
                onClose={() => setSelectedSlot(null)}
                onPrev={() => setSelectedSlot((s) => (s != null && s > 0 ? s - 1 : s))}
                onNext={() => setSelectedSlot((s) => (s != null && s < clips.length - 1 ? s + 1 : s))}
                onPickAlternative={pickAlternative}
                onToggleLock={() => updateClips((cs) => toggleLock(cs, selectedClip.slot))}
                onRegenerateSection={() => runPropose('section', selectedClip.section_index)}
                onPlayHere={() => { seek(selectedClip.start_ms); const el = activeEl(); el?.play().catch(() => {}); }}
                onOpenEditor={onOpenScene ? () => {
                  const idx = clips.findIndex((c) => c.slot === selectedClip.slot);
                  onOpenScene(clips.map(clipToScene), Math.max(0, idx));
                } : undefined}
              />
            )}

            {/* ── constraints + preview ───────────────────────────────── */}
            <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-5 items-start">
              <MusicConstraints
                value={constraints}
                onChange={setConstraints}
                facets={facets}
                sections={sections}
                selectedSection={selectedSection}
                onSelectSection={onSelectSection}
              />

              <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-4 flex flex-col gap-3">
                <div className="flex items-center gap-2 text-[#A1A1AA]">
                  <Film size={15} className="text-[#EC4899]" />
                  <span className="uppercase tracking-wider text-xs font-medium">{t('music.preview.title')}</span>
                  <span className="flex-1" />
                  {previewReady && (
                    <button type="button" onClick={() => switchSource(source === 'video' ? 'audio' : 'video')}
                            className="text-[11px] text-[#A1A1AA] hover:text-[#FAFAFA] underline">
                      {source === 'video' ? t('music.preview.musicOnly') : t('music.preview.useVideo')}
                    </button>
                  )}
                </div>

                {previewReady && job && (
                  <video
                    ref={videoRef}
                    key={job.id}
                    src={musicApi.previewUrl(job.id)}
                    controls
                    preload="auto"
                    className="w-full aspect-video rounded-lg bg-black"
                    onLoadedMetadata={(e) => { try { e.currentTarget.currentTime = currentMs / 1000; } catch {} }}
                    {...mediaHandlers('video')}
                  />
                )}
                {previewStale && (
                  <div className="text-[11px] text-amber-300 flex items-center gap-1.5"><AlertTriangle size={12} /> {t('music.preview.stale')}</div>
                )}
                {flaggedSlots.size > 0 && (
                  <div className="text-[11px] text-amber-300 flex items-start gap-1.5"><AlertTriangle size={12} className="shrink-0 mt-0.5" /> {t('music.preview.cutsInside', { n: flaggedSlots.size })}</div>
                )}

                {job && (job.status === 'queued' || job.status === 'running') ? (
                  <div className="flex flex-col gap-1.5">
                    <div className="flex items-center gap-2 text-xs text-[#A1A1AA]">
                      <Loader2 size={13} className="animate-spin text-[#EC4899]" />
                      {t('music.preview.running', { percent: progressPercent(job.progress) })}
                      {job.message && <span className="text-[#71717A] truncate">· {job.message}</span>}
                    </div>
                    <div className="h-1.5 bg-[#27272A] rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-[#EC4899] to-[#8B5CF6] transition-all" style={{ width: `${progressPercent(job.progress)}%` }} />
                    </div>
                  </div>
                ) : (
                  <>
                    {!previewReady && (
                      <div className="text-xs text-[#71717A]">{clips && clips.length > 0 ? t('music.preview.none') : t('music.preview.needEdit')}</div>
                    )}
                    <button type="button" onClick={startPreview} disabled={!clips || clips.length === 0}
                            className="self-start flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold bg-[#27272A] hover:bg-[#3F3F46] text-[#FAFAFA] disabled:opacity-40 disabled:cursor-not-allowed">
                      <Sparkles size={14} className="text-[#EC4899]" />
                      {previewReady ? t('music.preview.redo') : t('music.preview.start')}
                    </button>
                  </>
                )}
                {previewError && (
                  <div className="bg-red-500/10 border border-red-500/40 text-red-300 text-xs rounded-lg px-3 py-2 break-words">
                    {t('music.preview.failed', { message: previewError })}
                  </div>
                )}
              </div>
            </div>

            <MusicExportPanel key={analysis.hash} analysis={analysis} clips={clips} />
          </>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex flex-col items-start bg-[#0F0F11] border border-[#27272A] rounded-lg px-3 py-1.5 min-w-[72px]">
      <span className="text-[10px] uppercase tracking-wider text-[#71717A]">{label}</span>
      <span className="text-sm font-bold font-mono" style={{ color: accent ?? '#FAFAFA' }}>{value}</span>
    </div>
  );
}
