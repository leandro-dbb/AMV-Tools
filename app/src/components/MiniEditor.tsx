import { ChevronLeft, ChevronRight, ChevronDown, Play, Pause, Download, Plus, X, Loader2, Sparkles } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api/client';
import type { AppSettings, SceneResult, TagSummary, VideoSummary } from '../api/types';
import MaskMode from './MaskMode';

interface MiniEditorProps {
  scenes: SceneResult[];
  index: number;
  onIndexChange: (i: number) => void;
  onJumpTo?: (newScenes: SceneResult[], newIndex: number) => void;
  onClose: () => void;
  queryLabel?: string;
}

const PADDING_MS = 3000;

export default function MiniEditor({ scenes, index, onIndexChange, onJumpTo, onClose, queryLabel }: MiniEditorProps) {
  const scene = scenes[index];
  const [trimStart, setTrimStart] = useState(scene.start_ms);
  const [trimEnd, setTrimEnd] = useState(scene.end_ms);
  const [playing, setPlaying] = useState(false);
  const [currentMs, setCurrentMs] = useState(scene.start_ms);
  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState<string | null>(null);
  const [playbackError, setPlaybackError] = useState<string | null>(null);
  const [waveformData, setWaveformData] = useState<number[] | null>(null);
  const [allVideos, setAllVideos] = useState<VideoSummary[]>([]);
  const [tagsInVideo, setTagsInVideo] = useState<TagSummary[]>([]);
  const [activeTag, setActiveTag] = useState<string>(scene.tags[0] || '');
  const [switching, setSwitching] = useState(false);
  const [maskOpen, setMaskOpen] = useState(false);
  const [maskEnabled, setMaskEnabled] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const timelineRef = useRef<HTMLDivElement | null>(null);

  // Load videos + tags once per scene to enable contextual switching
  useEffect(() => {
    api.listVideos().then((r) => setAllVideos(r.videos)).catch(() => {});
    // Read settings to know whether the Mask toggle should show. The setting
    // gates the feature so users who don't want the SAM 2 download (~350 MB)
    // never see the button.
    api.getSettings().then((s: AppSettings) => setMaskEnabled(!!s.models.enable_mask)).catch(() => {});
  }, []);
  useEffect(() => {
    api.videoTags(scene.video_id).then((r) => setTagsInVideo(r.tags)).catch(() => {});
    setActiveTag(scene.tags[0] || '');
  }, [scene.video_id, scene.id]);

  async function jumpTo(newVideoId: number, newTag: string) {
    if (!onJumpTo) return;
    setSwitching(true);
    try {
      const r = await api.videoScenes(newVideoId, { tag: newTag || undefined, threshold: 0.3, sort: 'timecode' });
      if (r.scenes.length > 0) onJumpTo(r.scenes, 0);
      else if (r.scenes.length === 0 && newTag) {
        // no occurrence of this tag in the chosen video — still switch to its first scene
        const fallback = await api.videoScenes(newVideoId, { threshold: 0, sort: 'timecode' });
        if (fallback.scenes.length > 0) onJumpTo(fallback.scenes, 0);
      }
    } finally { setSwitching(false); }
  }

  // Timeline window for the trim handles: ±3s around the detected boundary.
  // When playing from a proxy (already trimmed), the player's currentTime is
  // 0-based on the scene. When playing from the source file, currentTime is
  // the absolute position in the source. The two converters below normalize
  // that so the rest of the component always works in absolute-ms.
  const windowStart = Math.max(0, scene.start_ms - PADDING_MS);
  const windowEnd = scene.end_ms + PADDING_MS;
  const windowDuration = windowEnd - windowStart;
  const proxyStart = scene.start_ms;
  const proxyEnd = scene.end_ms;
  const usingProxy = !!scene.proxy_path;

  const videoTimeToMs = (t: number) =>
    usingProxy ? proxyStart + Math.floor(t * 1000) : Math.floor(t * 1000);
  const msToVideoTime = (ms: number) =>
    usingProxy ? Math.max(0, (ms - proxyStart) / 1000) : Math.max(0, ms / 1000);

  // Reset trim when scene changes
  useEffect(() => {
    setTrimStart(scene.start_ms);
    setTrimEnd(scene.end_ms);
    setCurrentMs(scene.start_ms);
    setExportResult(null);
    setPlaybackError(null);
    if (videoRef.current) {
      // For source playback we have to seek to the scene's start_ms;
      // for proxies, currentTime=0 already lines up with the scene start.
      videoRef.current.currentTime = usingProxy ? 0 : scene.start_ms / 1000;
      videoRef.current.pause();
      setPlaying(false);
    }
  }, [scene.id, usingProxy]);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    setPlaybackError(null);
    if (v.paused) {
      const result = v.play();
      // `play()` returns a Promise that can reject (codec missing, range
      // request still loading, autoplay policy). Without catching this, the
      // button stayed stuck because setPlaying(true) never fired.
      if (result && typeof result.then === 'function') {
        result
          .then(() => setPlaying(true))
          .catch((err) => {
            console.error('Playback failed:', err);
            setPlaybackError(`Lecture impossible: ${err?.message || err}`);
            setPlaying(false);
          });
      } else {
        setPlaying(true);
      }
    } else {
      v.pause();
      setPlaying(false);
    }
  }, []);

  // Keyboard scrubbing J / K / L + arrows
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement | null)?.tagName === 'INPUT' || (e.target as HTMLElement | null)?.tagName === 'TEXTAREA') return;
      const v = videoRef.current;
      if (!v) return;
      if (e.key === 'k' || e.key === 'K' || e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.key === 'j' || e.key === 'J') {
        e.preventDefault();
        v.currentTime = Math.max(0, v.currentTime - 1);
      } else if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        v.currentTime = v.currentTime + 1;
      } else if (e.key === ',') {
        v.currentTime = Math.max(0, v.currentTime - 1 / 24);
      } else if (e.key === '.') {
        v.currentTime = v.currentTime + 1 / 24;
      } else if (e.key === 'ArrowLeft') {
        if (index > 0) onIndexChange(index - 1);
      } else if (e.key === 'ArrowRight') {
        if (index < scenes.length - 1) onIndexChange(index + 1);
      } else if (e.key === 'i' || e.key === 'I') {
        setTrimStart(Math.floor(videoTimeToMs(v.currentTime)));
      } else if (e.key === 'o' || e.key === 'O') {
        setTrimEnd(Math.floor(videoTimeToMs(v.currentTime)));
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [index, scenes.length, onIndexChange, windowStart, togglePlay]);

  // Track video currentTime for the playhead
  useEffect(() => {
    let raf: number;
    function tick() {
      const v = videoRef.current;
      if (v && !v.paused) {
        setCurrentMs(Math.floor(videoTimeToMs(v.currentTime)));
      }
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [windowStart]);

  // Build a waveform from the proxy via WebAudio (best-effort, falls back to synthetic).
  useEffect(() => {
    let cancelled = false;
    setWaveformData(null);
    if (!scene.proxy_path) return;
    (async () => {
      try {
        const resp = await fetch(api.proxyUrl(scene.id));
        if (!resp.ok) throw new Error('no proxy');
        const buf = await resp.arrayBuffer();
        const Ctx = (window as any).AudioContext || (window as any).webkitAudioContext;
        if (!Ctx) return;
        const audioCtx = new Ctx();
        const decoded = await audioCtx.decodeAudioData(buf.slice(0));
        const channel = decoded.getChannelData(0);
        const buckets = 200;
        const step = Math.floor(channel.length / buckets);
        const peaks: number[] = [];
        for (let i = 0; i < buckets; i++) {
          let max = 0;
          for (let j = 0; j < step; j++) {
            const s = Math.abs(channel[i * step + j] ?? 0);
            if (s > max) max = s;
          }
          peaks.push(max);
        }
        if (!cancelled) setWaveformData(peaks);
        audioCtx.close();
      } catch {
        // no audio in proxy (we strip it for size) — leave null, will use synthetic
      }
    })();
    return () => { cancelled = true; };
  }, [scene.id, scene.proxy_path]);

  const trimPct = useMemo(() => ({
    left: ((trimStart - windowStart) / windowDuration) * 100,
    width: ((trimEnd - trimStart) / windowDuration) * 100,
    head: ((currentMs - windowStart) / windowDuration) * 100,
  }), [trimStart, trimEnd, currentMs, windowStart, windowDuration]);

  // When playing from the source file the player can scrub anywhere in
  // [windowStart, windowEnd] including the ±3s padding around the detected
  // scene boundary. When playing from a proxy (already-trimmed clip), the
  // file itself only covers [proxyStart, proxyEnd] so we have to clamp.
  const seekableMin = usingProxy ? proxyStart : windowStart;
  const seekableMax = usingProxy ? proxyEnd : windowEnd;

  function seekVideoTo(ms: number) {
    const v = videoRef.current;
    if (!v) return;
    const clamped = Math.max(seekableMin, Math.min(seekableMax, ms));
    v.currentTime = msToVideoTime(clamped);
    setCurrentMs(clamped);
  }

  const dragHandle = useCallback((edge: 'start' | 'end') => (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const timeline = timelineRef.current;
    if (!timeline) return;
    const rect = timeline.getBoundingClientRect();
    const move = (ev: MouseEvent) => {
      const pct = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
      const ms = Math.floor(windowStart + pct * windowDuration);
      if (edge === 'start') {
        const newStart = Math.min(ms, trimEnd - 200);
        setTrimStart(newStart);
        // Live preview: scrub the player to wherever the handle is.
        seekVideoTo(newStart);
      } else {
        const newEnd = Math.max(ms, trimStart + 200);
        setTrimEnd(newEnd);
        seekVideoTo(newEnd);
      }
    };
    const up = () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  }, [trimStart, trimEnd, windowStart, windowDuration, seekableMin, seekableMax]);

  function seekTimeline(e: React.MouseEvent) {
    const timeline = timelineRef.current;
    if (!timeline) return;
    const rect = timeline.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const targetMs = windowStart + pct * windowDuration;
    seekVideoTo(targetMs);
  }

  async function doExport() {
    setExporting(true);
    setExportResult(null);
    try {
      const r = await api.export({ scene_id: scene.id, start_ms: trimStart, end_ms: trimEnd });
      setExportResult(r.output);
      api.revealOutput(r.output);
    } catch (err) {
      setExportResult(`Error: ${(err as Error).message}`);
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className="fixed inset-0 bg-black/90 backdrop-blur-sm flex items-center justify-center z-50 p-8" onClick={onClose}>
      <div className="w-full h-full bg-gradient-to-br from-[#18181B] via-[#0F0F11] to-[#18181B] rounded-2xl border-2 border-[#27272A] flex flex-col overflow-hidden shadow-2xl shadow-[#8B5CF6]/20" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b-2 border-[#27272A] bg-[#18181B]/50 backdrop-blur-sm">
          <div className="flex items-center gap-3 min-w-0">
            {queryLabel && (
              <>
                <span className="text-[#71717A] text-sm uppercase tracking-wider whitespace-nowrap">Query</span>
                <ChevronRight size={14} className="text-[#8B5CF6]" />
                <span className="text-[#A1A1AA] text-sm font-mono bg-[#27272A] px-3 py-1.5 rounded-lg truncate max-w-[200px]">{queryLabel}</span>
                <ChevronRight size={14} className="text-[#71717A]" />
              </>
            )}
            {onJumpTo ? (
              <>
                <div className="relative">
                  <select value={scene.video_id} onChange={(e) => jumpTo(Number(e.target.value), activeTag)}
                          className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 hover:border-[#8B5CF6] rounded-lg pl-4 pr-9 py-2 text-[#FAFAFA] text-sm max-w-[260px] appearance-none cursor-pointer">
                    {allVideos.map((v) => <option key={v.id} value={v.id}>{v.display_name}</option>)}
                  </select>
                  <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 text-[#8B5CF6] pointer-events-none" size={14} />
                </div>
                <ChevronRight size={14} className="text-[#EC4899]" />
                <div className="relative">
                  <select value={activeTag} onChange={(e) => { setActiveTag(e.target.value); jumpTo(scene.video_id, e.target.value); }}
                          className="bg-[#0F0F11] border-2 border-[#EC4899]/30 hover:border-[#EC4899] rounded-lg pl-4 pr-9 py-2 text-[#FAFAFA] text-sm appearance-none cursor-pointer">
                    {tagsInVideo.length === 0 && <option value="">(no tags)</option>}
                    {tagsInVideo.map((t) => <option key={t.tag} value={t.tag}>{t.tag} ({t.count})</option>)}
                  </select>
                  <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 text-[#EC4899] pointer-events-none" size={14} />
                </div>
                {switching && <Loader2 size={14} className="animate-spin text-[#8B5CF6]" />}
              </>
            ) : (
              <>
                <div className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-4 py-2 text-[#FAFAFA] text-sm max-w-[300px] truncate">{scene.video_display}</div>
                <ChevronRight size={14} className="text-[#EC4899]" />
                <div className="bg-[#0F0F11] border-2 border-[#EC4899]/30 rounded-lg px-4 py-2 text-[#FAFAFA] text-sm">{scene.tags.slice(0, 2).join(', ') || 'untagged'}</div>
              </>
            )}
            <ChevronRight size={14} className="text-[#71717A]" />
            <span className="text-[#A1A1AA] text-sm font-mono bg-[#27272A] px-3 py-1.5 rounded-lg whitespace-nowrap">{index + 1} / {scenes.length}</span>
          </div>
          <div className="flex items-center gap-2">
            <button disabled={index === 0} onClick={() => onIndexChange(index - 1)} className="p-2.5 hover:bg-[#27272A] rounded-lg transition-all group disabled:opacity-30" title="Prev occurrence (←)">
              <ChevronLeft size={20} className="text-[#A1A1AA] group-hover:text-[#8B5CF6]" />
            </button>
            <button disabled={index >= scenes.length - 1} onClick={() => onIndexChange(index + 1)} className="p-2.5 hover:bg-[#27272A] rounded-lg transition-all group disabled:opacity-30" title="Next occurrence (→)">
              <ChevronRight size={20} className="text-[#A1A1AA] group-hover:text-[#8B5CF6]" />
            </button>
            {/* Mask toggle — only visible if the feature is enabled in Settings.
                Lives next to Export so users find both clip-output paths in one
                spot. */}
            {maskEnabled && (
              <button
                onClick={() => setMaskOpen((on) => !on)}
                className={
                  maskOpen
                    ? 'ml-3 px-5 py-2.5 rounded-xl text-sm font-semibold flex items-center gap-2 bg-gradient-to-r from-[#A855F7] to-[#EC4899] text-white shadow-lg shadow-[#A855F7]/50'
                    : 'ml-3 px-5 py-2.5 rounded-xl text-sm font-semibold flex items-center gap-2 bg-[#27272A] hover:bg-[#3F3F46] text-[#A1A1AA] border-2 border-[#A855F7]/40 hover:border-[#A855F7]'
                }
                title={maskOpen ? 'Close mask mode' : 'Open mask mode (click→mask, alpha export)'}
              >
                <Sparkles size={16} />
                Mask
              </button>
            )}
            {/* Export button lives in the header so it stays visible regardless
                of window size or how busy the bottom bar gets. */}
            <button
              disabled={exporting}
              onClick={doExport}
              className="ml-3 bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-[#FAFAFA] px-5 py-2.5 rounded-xl text-sm font-semibold hover:shadow-xl hover:shadow-[#8B5CF6]/50 transition-all flex items-center gap-2 disabled:opacity-50 whitespace-nowrap"
              title={`Export ${((trimEnd - trimStart) / 1000).toFixed(2)}s clip via ffmpeg`}
            >
              {exporting ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
              {exporting ? 'Exporting…' : `Export ${((trimEnd - trimStart) / 1000).toFixed(1)}s`}
            </button>
            <button onClick={onClose} className="p-2.5 hover:bg-[#27272A] rounded-lg transition-all group ml-2">
              <X size={20} className="text-[#A1A1AA] group-hover:text-[#EC4899]" />
            </button>
          </div>
        </div>

        {/* Video player */}
        <div className="flex-1 flex items-center justify-center p-8 relative">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-[#8B5CF6]/10 rounded-full blur-[150px] pointer-events-none"></div>
          <div className="relative w-full max-w-6xl">
            <div className="absolute -inset-4 bg-gradient-to-r from-[#8B5CF6]/20 via-[#EC4899]/20 to-[#8B5CF6]/20 rounded-2xl blur-xl"></div>
            <div className="relative aspect-video bg-black rounded-xl overflow-hidden border-2 border-[#8B5CF6]/30 shadow-2xl shadow-[#8B5CF6]/30">
              <video
                ref={videoRef}
                src={scene.proxy_path ? api.proxyUrl(scene.id) : api.sourceUrl(scene.id)}
                preload="metadata"
                poster={api.thumbnailUrl(scene.id)}
                onLoadedMetadata={(e) => {
                  if (!scene.proxy_path) {
                    e.currentTarget.currentTime = scene.start_ms / 1000;
                  }
                }}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onTimeUpdate={(e) => setCurrentMs(videoTimeToMs(e.currentTarget.currentTime))}
                onError={(e) => {
                  const err = e.currentTarget.error;
                  const msg = err
                    ? `Code ${err.code}: ${err.message || 'codec unsupported or source unreachable'}`
                    : 'unknown video error';
                  setPlaybackError(msg);
                }}
                className="w-full h-full cursor-pointer"
                onClick={togglePlay}
              />
              {playbackError && (
                <div className="absolute inset-0 bg-black/70 flex items-center justify-center pointer-events-none">
                  <div className="text-center text-[#A1A1AA] text-sm max-w-md px-6">
                    <div className="text-red-400 font-semibold mb-2">Playback error</div>
                    <div className="font-mono text-xs">{playbackError}</div>
                    <div className="text-[#71717A] mt-3">
                      Tip: Settings → Indexing → Generate proxies = ON for unsupported codecs.
                    </div>
                  </div>
                </div>
              )}
              {maskOpen && (
                <MaskMode
                  key={`${scene.id}:${trimStart}:${trimEnd}`}
                  scene={scene}
                  trimStart={trimStart}
                  trimEnd={trimEnd}
                  onClose={() => setMaskOpen(false)}
                  onExportError={(msg) => setExportResult(`Error: ${msg}`)}
                  onExportSuccess={(out) => { setExportResult(out); api.revealOutput(out); }}
                />
              )}
            </div>
          </div>
        </div>

        {/* Timeline + waveform + handles */}
        <div className="p-6 border-t-2 border-[#27272A] bg-[#18181B]/50 backdrop-blur-sm space-y-4">
          <div ref={timelineRef} className="h-24 bg-gradient-to-br from-[#0F0F11] to-[#18181B] rounded-xl border-2 border-[#27272A] overflow-hidden relative cursor-crosshair" onClick={seekTimeline}>
            {/* Waveform */}
            <div className="absolute inset-0 opacity-30 flex items-end">
              {(waveformData ?? syntheticWaveform).map((v, i, arr) => (
                <div key={i} className="bg-gradient-to-t from-[#8B5CF6] to-[#EC4899]"
                  style={{ width: `${100 / arr.length}%`, height: `${Math.min(95, v * 95 + 5)}%` }} />
              ))}
            </div>

            {/* Detected scene bounds (dim, behind the trim region) */}
            <div className="absolute inset-y-0 bg-white/5 border-l border-r border-white/20 pointer-events-none"
                 style={{
                   left: `${((proxyStart - windowStart) / windowDuration) * 100}%`,
                   width: `${((proxyEnd - proxyStart) / windowDuration) * 100}%`,
                 }} title="Detected scene bounds" />

            {/* Trim region (user-editable) */}
            <div className="absolute inset-y-0 bg-gradient-to-r from-[#8B5CF6]/20 via-[#8B5CF6]/40 to-[#8B5CF6]/20 border-l-4 border-r-4 border-[#8B5CF6]"
                 style={{ left: `${trimPct.left}%`, width: `${trimPct.width}%` }}>
              <div onMouseDown={dragHandle('start')} className="absolute -left-1.5 top-0 bottom-0 w-3 bg-white rounded-full cursor-ew-resize shadow-lg shadow-[#8B5CF6]/50 z-20" title="Drag to adjust scene start"></div>
              <div onMouseDown={dragHandle('end')} className="absolute -right-1.5 top-0 bottom-0 w-3 bg-white rounded-full cursor-ew-resize shadow-lg shadow-[#8B5CF6]/50 z-20" title="Drag to adjust scene end"></div>
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <span className="text-white text-xs font-mono font-bold bg-black/60 px-2 py-1 rounded">{((trimEnd - trimStart) / 1000).toFixed(2)}s</span>
              </div>
            </div>

            {/* Playhead */}
            <div className="absolute top-0 bottom-0 w-0.5 bg-[#EC4899] z-30 pointer-events-none" style={{ left: `${trimPct.head}%` }}>
              <div className="absolute -top-1 -left-1.5 w-3 h-3 bg-[#EC4899] rounded-full"></div>
            </div>

            <div className="absolute bottom-1 left-3 right-3 flex justify-between text-[#71717A] text-[10px] font-mono pointer-events-none">
              <span>{fmt(windowStart)}</span>
              <span>{fmt(windowEnd)}</span>
            </div>
          </div>

          <div className="flex items-center gap-5 flex-wrap">
            <button
              onClick={togglePlay}
              className="w-14 h-14 bg-gradient-to-br from-[#8B5CF6] to-[#7C3AED] rounded-full flex items-center justify-center hover:scale-105 hover:shadow-2xl hover:shadow-[#8B5CF6]/70 transition-all shadow-lg shadow-[#8B5CF6]/50 active:scale-95 flex-shrink-0"
              title={playing ? 'Pause (Space / K)' : 'Play (Space / K)'}
            >
              {playing ? <Pause size={24} className="text-white" fill="white" /> : <Play size={24} className="text-white ml-1" fill="white" />}
            </button>
            <div className="font-mono text-3xl font-bold bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] bg-clip-text text-transparent">{fmtFull(currentMs)}</div>
            <div className="hidden lg:flex gap-2 text-[#71717A] text-sm self-center">
              <kbd className="px-2 py-1 bg-[#27272A] text-[#A1A1AA] text-xs rounded font-mono border border-[#3f3f46]">␣</kbd>
              <span className="self-center mr-1">play ·</span>
              <kbd className="px-2 py-1 bg-[#27272A] text-[#A1A1AA] text-xs rounded font-mono border border-[#3f3f46]">J</kbd>
              <kbd className="px-2 py-1 bg-[#27272A] text-[#A1A1AA] text-xs rounded font-mono border border-[#3f3f46]">K</kbd>
              <kbd className="px-2 py-1 bg-[#27272A] text-[#A1A1AA] text-xs rounded font-mono border border-[#3f3f46]">L</kbd>
              <span className="self-center ml-1">scrub ·</span>
              <kbd className="px-2 py-1 bg-[#27272A] text-[#A1A1AA] text-xs rounded font-mono border border-[#3f3f46]">I</kbd>
              <kbd className="px-2 py-1 bg-[#27272A] text-[#A1A1AA] text-xs rounded font-mono border border-[#3f3f46]">O</kbd>
              <span className="self-center ml-1">trim</span>
            </div>
            {exportResult && !exportResult.startsWith('Error') && (
              <span className="ml-auto text-xs font-mono truncate max-w-[280px] text-[#8B5CF6]">
                ✓ Exported: {exportResult.split(/[\\/]/).pop()}
              </span>
            )}
          </div>
          {exportResult && exportResult.startsWith('Error') && (
            <div className="mt-3 bg-red-500/10 border-2 border-red-500/40 text-red-200 rounded-xl p-3 flex items-start gap-3">
              <div className="flex-1">
                <div className="font-semibold text-red-300 text-sm mb-1">Export failed</div>
                <div className="font-mono text-xs whitespace-pre-wrap break-all">{exportResult.replace(/^Error:\s*/, '')}</div>
                <div className="text-[#A1A1AA] text-xs mt-2">
                  Tip: check Settings → Export → Codec. <span className="font-mono">libx264</span> is the most compatible.
                </div>
              </div>
              <button onClick={() => setExportResult(null)} className="text-red-300 hover:text-red-200 p-1 -mt-1">
                <X size={16} />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const syntheticWaveform = Array.from({ length: 200 }, () => Math.random() * 0.7 + 0.1);

function fmt(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}
function fmtFull(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const f = Math.floor((ms % 1000) / 1000 * 24);
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}:${f.toString().padStart(2, '0')}`;
}
