import { Loader2, X, Eraser, RotateCcw, Sparkles, Download, MousePointerClick, Wand2, Brush, Undo2, RefreshCw } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api/client';
import { maskApi, type BrushStroke, type SegmentSessionInfo } from '../api/mask';
import { useT } from '../i18n';
import type { SceneResult } from '../api/types';
import MaskBrushLayer from './MaskBrushLayer';

type Engine = 'birefnet' | 'sam2';

type Polarity = 'positive' | 'negative';
type Point = { x: number; y: number; polarity: Polarity };
type Box = [number, number, number, number] | null;

type Phase = 'idle' | 'previewing' | 'reviewing' | 'tracking' | 'ready';
type Tool = 'points' | 'add' | 'remove';
type ViewMode = 'overlay' | 'checker' | 'color';

interface MaskModeProps {
  scene: SceneResult;
  trimStart: number;
  trimEnd: number;
  onClose: () => void;
  onExportError: (message: string) => void;
  onExportSuccess: (outputPath: string) => void;
}

const CHECKER_BG = {
  backgroundColor: '#ffffff',
  backgroundImage:
    'linear-gradient(45deg, #c8c8c8 25%, transparent 25%), linear-gradient(-45deg, #c8c8c8 25%, transparent 25%),' +
    'linear-gradient(45deg, transparent 75%, #c8c8c8 75%), linear-gradient(-45deg, transparent 75%, #c8c8c8 75%)',
  backgroundSize: '20px 20px',
  backgroundPosition: '0 0, 0 10px, 10px -10px, -10px 0px',
};

/** SAM 2 / BiRefNet mask + alpha-export panel.
 *
 * Sits inside the MiniEditor's video area when the user toggles "Mask".
 * Owns its own canvas + click/box/brush state and talks to
 * /api/scene/<id>/segment. Any frame can carry clicks (keyframes) and brush
 * corrections; "Re-propagate" re-tracks with all of them. The component
 * releases the server session on unmount so the tempdir of extracted frames
 * gets cleaned up. */
export default function MaskMode({ scene, trimStart, trimEnd, onClose, onExportError, onExportSuccess }: MaskModeProps) {
  const t = useT();
  const [phase, setPhase] = useState<Phase>('idle');
  const [points, setPoints] = useState<Point[]>([]);
  const [box, setBox] = useState<Box>(null);
  const [dragBox, setDragBox] = useState<Box>(null);
  const [session, setSession] = useState<SegmentSessionInfo | null>(null);
  const [maskCacheBust, setMaskCacheBust] = useState(0);
  const [viewIdx, setViewIdx] = useState(0);
  const [trackPct, setTrackPct] = useState(0);
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [alphaCodec, setAlphaCodec] = useState<'prores_4444_alpha' | 'vp9_alpha'>('prores_4444_alpha');
  const [engine, setEngine] = useState<Engine>('birefnet');
  const [autoTarget, setAutoTarget] = useState<'main' | 'all'>('all');
  const [tool, setTool] = useState<Tool>('points');
  const [brushRadius, setBrushRadius] = useState(14);
  const [viewMode, setViewMode] = useState<ViewMode>('overlay');
  const [bgColor, setBgColor] = useState('#00b140');
  const [corrected, setCorrected] = useState<number[]>([]);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewIdxCommitted = useRef(0);

  // Read the saved engine choice once, so the chip starts on the user's
  // preferred default. The chip lets them flip without leaving the editor.
  useEffect(() => {
    Promise.all([api.getSettings(), api.status()])
      .then(([s, status]) => {
        if (s.models?.mask_engine === 'sam2' && status.device !== 'dml') setEngine('sam2');
        if (s.models?.mask_auto_target === 'main') setAutoTarget('main');
      })
      .catch(() => {});
  }, []);

  // ── init on mount: extract the reference frame, run mask if engine=auto ──
  // BiRefNet auto-runs on the reference frame as soon as the session is ready
  // (no click required). SAM 2 only gets the session + frame_url at this
  // stage; the user has to click before any mask is produced.
  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    setViewIdx(0);
    viewIdxCommitted.current = 0;
    api.segmentPreview(scene.id, { positive: [], negative: [], box: null, reference_frame_offset: 0, start_ms: trimStart, end_ms: trimEnd, engine, auto_target: autoTarget })
      .then((r) => {
        if (cancelled) return;
        setSession(r as SegmentSessionInfo);
        if (r.has_mask) {
          setPhase('reviewing');
          setMaskCacheBust((n) => n + 1);
        }
      })
      .catch((err) => { if (!cancelled) setErrorMsg(t('mask.error.init', { message: (err as Error).message })); })
      .finally(() => { if (!cancelled) setBusy(false); });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene.id, trimStart, trimEnd, engine, autoTarget]);

  // ── cleanup on unmount: drop the server session (frees the tempdir) ──────
  useEffect(() => {
    return () => {
      if (session?.session_id) {
        api.segmentDrop(session.session_id).catch(() => {});
      }
    };
  }, [session?.session_id]);

  // ── progress feed for the tracking phase ────────────────────────────────
  useEffect(() => {
    if (phase !== 'tracking' || !session) return;
    const ws = api.progressSocket();
    ws.onmessage = (m) => {
      try {
        const e = JSON.parse(m.data);
        if (e.type === 'sam2_track' && e.session_id === session.session_id) {
          setTrackPct(e.percent ?? 0);
        }
      } catch { /* ignore */ }
    };
    return () => ws.close();
  }, [phase, session?.session_id]);

  // ── moving to another frame before tracking ─────────────────────────────
  // Every frame can be a keyframe: the server switches its reference frame
  // (Auto re-runs BiRefNet there; Manual just shows the frame, waiting for
  // clicks). Debounced so dragging the slider does not spam the model.
  useEffect(() => {
    if (!session || phase === 'ready' || phase === 'tracking') return;
    if (viewIdx === viewIdxCommitted.current) return;
    const handle = setTimeout(() => {
      viewIdxCommitted.current = viewIdx;
      setPoints([]);
      setBox(null);
      setBusy(true);
      api.segmentPreview(scene.id, {
        positive: [], negative: [], box: null, reference_frame_offset: viewIdx,
        start_ms: trimStart, end_ms: trimEnd, engine,
      })
        .then((r) => {
          setSession(r as SegmentSessionInfo);
          setMaskCacheBust((n) => n + 1);
          if (r.has_mask && phase === 'idle') setPhase('reviewing');
        })
        .catch((err) => setErrorMsg(t('mask.error.preview', { message: (err as Error).message })))
        .finally(() => setBusy(false));
    }, 300);
    return () => clearTimeout(handle);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewIdx, session?.session_id, phase]);

  // The reference frame dictates the natural coords. Until the first preview
  // the user can still place points on the thumbnail — we'll send them at the
  // thumbnail size and trust the server to scale. For accuracy though, the
  // first preview pins the natural size.
  const naturalSize = useMemo(() => {
    if (session) return { w: session.frame_w, h: session.frame_h };
    return { w: 1280, h: 720 }; // sensible fallback
  }, [session?.frame_w, session?.frame_h]);

  function translateClick(e: React.MouseEvent<HTMLDivElement>): [number, number] {
    const target = containerRef.current;
    if (!target) return [0, 0];
    const rect = target.getBoundingClientRect();
    const xRel = (e.clientX - rect.left) / rect.width;
    const yRel = (e.clientY - rect.top) / rect.height;
    return [Math.round(xRel * naturalSize.w), Math.round(yRel * naturalSize.h)];
  }

  // ── canvas event handlers ───────────────────────────────────────────────
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (phase === 'tracking' || tool !== 'points') return;
    if (e.button !== 0) return;
    // Shift = negative point. Otherwise plain click = positive point (we
    // disambiguate from drag→box on mouseup).
    e.preventDefault();
    const [x, y] = translateClick(e);
    const polarity: Polarity = e.shiftKey ? 'negative' : 'positive';

    // Start a potential drag for a box. We commit it as a box only if the
    // pointer moved > 6 px during the drag. The latest box coords live in a
    // closure-local so the mouseup reads them without waiting for a React
    // re-render (a ref synced via useEffect would be a frame stale).
    let startedDrag = false;
    let latestBox: Box = null;
    const startX = e.clientX;
    const startY = e.clientY;

    const move = (ev: MouseEvent) => {
      if (!startedDrag && Math.hypot(ev.clientX - startX, ev.clientY - startY) > 6) {
        startedDrag = true;
      }
      if (startedDrag) {
        const target = containerRef.current;
        if (!target) return;
        const rect = target.getBoundingClientRect();
        const xRel = (ev.clientX - rect.left) / rect.width;
        const yRel = (ev.clientY - rect.top) / rect.height;
        const nx = Math.round(xRel * naturalSize.w);
        const ny = Math.round(yRel * naturalSize.h);
        latestBox = [
          Math.min(x, nx),
          Math.min(y, ny),
          Math.max(x, nx),
          Math.max(y, ny),
        ];
        setDragBox(latestBox);
      }
    };

    const up = () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
      if (startedDrag && latestBox) {
        setBox(latestBox);
        setDragBox(null);
      } else {
        setPoints((prev) => [...prev, { x, y, polarity }]);
      }
    };

    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  }, [phase, tool, naturalSize.w, naturalSize.h]);

  function clearPrompts() {
    setPoints([]);
    setBox(null);
    setDragBox(null);
  }

  function removeLastPrompt() {
    if (box || dragBox) {
      setBox(null);
      setDragBox(null);
      return;
    }
    setPoints((prev) => prev.slice(0, -1));
  }

  // Esc to clear prompts (parent's Escape-to-close is intercepted only if
  // we're not in idle and have prompts).
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (points.length || box) {
          e.preventDefault();
          e.stopPropagation();
          clearPrompts();
        }
      }
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [points.length, box]);

  // ── server calls ────────────────────────────────────────────────────────
  async function runPreview() {
    // BiRefNet auto-ran on mount — re-running is only for SAM 2 (the user
    // has added/changed clicks). Guarded for SAM 2 only:
    if (engine === 'sam2' && !points.some((p) => p.polarity === 'positive') && !box) {
      setErrorMsg(t('mask.error.clickFirst'));
      return;
    }
    setBusy(true);
    setErrorMsg(null);
    setPhase('previewing');
    try {
      const r = await api.segmentPreview(scene.id, {
        positive: points.filter((p) => p.polarity === 'positive').map((p) => [p.x, p.y] as [number, number]),
        negative: points.filter((p) => p.polarity === 'negative').map((p) => [p.x, p.y] as [number, number]),
        box: box ?? null,
        reference_frame_offset: viewIdx,
        start_ms: trimStart,
        end_ms: trimEnd,
        engine,
        auto_target: autoTarget,
      });
      viewIdxCommitted.current = viewIdx;
      setSession(r as SegmentSessionInfo);
      setMaskCacheBust((n) => n + 1);
      setPhase('reviewing');
    } catch (err) {
      setErrorMsg(t('mask.error.preview', { message: (err as Error).message }));
      setPhase('idle');
    } finally {
      setBusy(false);
    }
  }

  async function runTrack() {
    if (!session) return;
    setBusy(true);
    setErrorMsg(null);
    setPhase('tracking');
    setTrackPct(0);
    try {
      const r = await api.segmentTrack(scene.id, {
        session_id: session.session_id,
        positive: points.filter((p) => p.polarity === 'positive').map((p) => [p.x, p.y] as [number, number]),
        negative: points.filter((p) => p.polarity === 'negative').map((p) => [p.x, p.y] as [number, number]),
        box: box ?? null,
        reference_frame_offset: viewIdx,
        start_ms: trimStart,
        end_ms: trimEnd,
        engine,
        auto_target: autoTarget,
      }) as Awaited<ReturnType<typeof api.segmentTrack>> & { keyframes?: number[]; corrected_frames?: number[] };
      setSession({ ...session, n_frames: r.n_frames, keyframes: r.keyframes, corrected_frames: r.corrected_frames, tracked: true });
      if (r.corrected_frames) setCorrected(r.corrected_frames);
      setMaskCacheBust((n) => n + 1);
      setPhase('ready');
    } catch (err) {
      setErrorMsg(t('mask.error.tracking', { message: (err as Error).message }));
      setPhase('reviewing');
    } finally {
      setBusy(false);
    }
  }

  async function applyStroke(stroke: BrushStroke) {
    if (!session) return;
    try {
      const r = await maskApi.brush(session.session_id, { frame_idx: viewIdx, strokes: [stroke] });
      setCorrected(r.corrected_frames);
      setMaskCacheBust((n) => n + 1);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function resetCorrection() {
    if (!session) return;
    try {
      const r = await maskApi.brush(session.session_id, { frame_idx: viewIdx, reset: true });
      setCorrected(r.corrected_frames);
      setMaskCacheBust((n) => n + 1);
    } catch (err) {
      setErrorMsg((err as Error).message);
    }
  }

  async function runExport() {
    if (!session) return;
    setBusy(true);
    setErrorMsg(null);
    try {
      const r = await api.segmentExport(scene.id, {
        session_id: session.session_id,
        codec: alphaCodec,
      });
      onExportSuccess(r.output);
    } catch (err) {
      const msg = (err as Error).message;
      setErrorMsg(t('mask.error.export', { message: msg }));
      onExportError(msg);
    } finally {
      setBusy(false);
    }
  }

  // ── render ──────────────────────────────────────────────────────────────
  const tracked = phase === 'ready' && !!session;
  const frameHasMask = tracked || (!!session?.has_mask && session.reference_frame_offset === viewIdx);
  const bgUrl =
    session
      ? api.segmentFrameUrl(session.session_id, tracked ? viewIdx : session.reference_frame_offset)
      : api.thumbnailUrl(scene.id);
  // Sidecar URLs already carry ?token= in Electron: append the cache-buster.
  const bust = (url: string) => `${url}${url.includes('?') ? '&' : '?'}v=${maskCacheBust}`;
  const maskUrl =
    tracked && session
      ? bust(api.segmentMaskUrl(session.session_id, viewIdx))
      : (phase === 'reviewing' || phase === 'previewing') && session && frameHasMask
        ? bust(api.segmentMaskPreviewUrl(session.session_id))
        : null;
  const cutoutUrl = session && frameHasMask ? bust(maskApi.cutoutUrl(session.session_id, viewIdx)) : null;
  const showCutout = viewMode !== 'overlay' && !!cutoutUrl;

  const positiveCount = points.filter((p) => p.polarity === 'positive').length;
  const negativeCount = points.length - positiveCount;
  const hasAnyPrompt = points.length > 0 || box !== null;
  const keyframes = session?.keyframes ?? [];
  const correctedFrames = corrected.length ? corrected : (session?.corrected_frames ?? []);
  const nFrames = session?.n_frames ?? 1;

  return (
    <div className="absolute inset-0 z-30 bg-black/85 backdrop-blur-md flex flex-col p-4">
      {/* toolbar */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <div className="bg-gradient-to-r from-[#A855F7] to-[#EC4899] text-white text-xs font-bold px-3 py-1.5 rounded-lg uppercase tracking-wider">
          {t('mask.badge')}
        </div>
        {/* Engine chip — quick A/B switch without leaving the editor. */}
        <div className="flex items-center bg-[#0F0F11] border border-[#27272A] rounded-lg p-0.5">
          {([
            { id: 'birefnet', label: t('mask.engine.auto'), icon: Wand2, title: t('mask.engine.autoTitle') },
            { id: 'sam2', label: t('mask.engine.manual'), icon: MousePointerClick, title: t('mask.engine.manualTitle') },
          ] as const).map(({ id, label, icon: Icon, title }) => (
            <button
              key={id}
              onClick={() => setEngine(id as Engine)}
              disabled={busy && phase !== 'idle'}
              className={`px-3 py-1 rounded text-xs font-semibold flex items-center gap-1.5 transition-all ${
                engine === id
                  ? 'bg-gradient-to-r from-[#A855F7] to-[#EC4899] text-white'
                  : 'text-[#A1A1AA] hover:text-[#FAFAFA]'
              }`}
              title={title}
            >
              <Icon size={12} /> {label}
            </button>
          ))}
        </div>
        {/* Auto engine: which characters to track (no click needed). */}
        {engine === 'birefnet' && (
          <div className="flex items-center bg-[#0F0F11] border border-[#27272A] rounded-lg p-0.5">
            {([
              { id: 'main', label: t('mask.autoTarget.main'), title: t('mask.autoTarget.mainTitle') },
              { id: 'all', label: t('mask.autoTarget.all'), title: t('mask.autoTarget.allTitle') },
            ] as const).map(({ id, label, title }) => (
              <button
                key={id}
                onClick={() => setAutoTarget(id)}
                disabled={busy}
                className={`px-3 py-1 rounded text-xs font-semibold transition-all ${
                  autoTarget === id
                    ? 'bg-gradient-to-r from-[#A855F7] to-[#EC4899] text-white'
                    : 'text-[#A1A1AA] hover:text-[#FAFAFA]'
                }`}
                title={title}
              >
                {label}
              </button>
            ))}
          </div>
        )}
        {/* Tool chip — click prompts vs. brush corrections. */}
        <div className="flex items-center bg-[#0F0F11] border border-[#27272A] rounded-lg p-0.5">
          {([
            { id: 'points', label: t('finish.mask.tool.points'), title: t('mask.legend.clickEq') },
            { id: 'add', label: t('finish.mask.tool.add'), title: t('finish.mask.tool.addTitle') },
            { id: 'remove', label: t('finish.mask.tool.remove'), title: t('finish.mask.tool.removeTitle') },
          ] as const).map(({ id, label, title }) => (
            <button
              key={id}
              onClick={() => setTool(id)}
              disabled={phase === 'tracking' || (id !== 'points' && !frameHasMask)}
              title={id !== 'points' && !frameHasMask ? t('finish.mask.brushNeedsMask') : title}
              className={`px-2.5 py-1 rounded text-xs font-semibold flex items-center gap-1 disabled:opacity-40 ${
                tool === id ? 'bg-[#3F3F46] text-white' : 'text-[#A1A1AA] hover:text-[#FAFAFA]'
              }`}
            >
              {id === 'points' ? <MousePointerClick size={12} /> : <Brush size={12} />} {label}
            </button>
          ))}
        </div>
        {tool !== 'points' && (
          <label className="flex items-center gap-1.5 text-[11px] text-[#A1A1AA]">
            {t('finish.mask.brushSize')}
            <input type="range" min={3} max={60} value={brushRadius} onChange={(e) => setBrushRadius(Number(e.target.value))} className="w-20 accent-[#A855F7]" />
          </label>
        )}
        {/* Preview background */}
        <div className="flex items-center bg-[#0F0F11] border border-[#27272A] rounded-lg p-0.5" title={t('finish.mask.viewTitle')}>
          {(['overlay', 'checker', 'color'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setViewMode(m)}
              className={`px-2.5 py-1 rounded text-xs ${viewMode === m ? 'bg-[#3F3F46] text-white' : 'text-[#A1A1AA] hover:text-[#FAFAFA]'}`}
            >{t(`finish.mask.view.${m}`)}</button>
          ))}
          {viewMode === 'color' && (
            <input type="color" value={bgColor} onChange={(e) => setBgColor(e.target.value)} className="w-6 h-6 ml-1 bg-transparent border-0 cursor-pointer" />
          )}
        </div>
        <div className="flex-1" />
        {engine === 'sam2' && tool === 'points' && (
          <>
            <button
              disabled={phase === 'tracking' || !hasAnyPrompt}
              onClick={removeLastPrompt}
              className="px-3 py-1.5 rounded-lg bg-[#27272A] hover:bg-[#3F3F46] text-[#A1A1AA] text-xs flex items-center gap-1.5 disabled:opacity-40"
              title={t('mask.undoTitle')}
            >
              <RotateCcw size={12} /> {t('mask.undo')}
            </button>
            <button
              disabled={phase === 'tracking' || !hasAnyPrompt}
              onClick={clearPrompts}
              className="px-3 py-1.5 rounded-lg bg-[#27272A] hover:bg-[#3F3F46] text-[#A1A1AA] text-xs flex items-center gap-1.5 disabled:opacity-40"
              title={t('mask.clearTitle')}
            >
              <Eraser size={12} /> {t('mask.clear')}
            </button>
          </>
        )}
        {correctedFrames.includes(viewIdx) && (
          <button
            disabled={phase === 'tracking'}
            onClick={resetCorrection}
            className="px-3 py-1.5 rounded-lg bg-[#27272A] hover:bg-[#3F3F46] text-[#A1A1AA] text-xs flex items-center gap-1.5 disabled:opacity-40"
          >
            <Undo2 size={12} /> {t('finish.mask.resetFrame')}
          </button>
        )}
        <button
          onClick={onClose}
          disabled={busy}
          className="ml-2 p-1.5 rounded-lg hover:bg-[#27272A] text-[#A1A1AA] disabled:opacity-40"
          title={t('mask.closeTitle')}
        >
          <X size={16} />
        </button>
      </div>

      {/* status + legend */}
      <div className="flex items-center gap-3 mb-3 text-[11px] text-[#A1A1AA] bg-[#0F0F11]/80 border border-[#27272A] rounded-lg px-4 py-2 flex-wrap">
        {engine === 'sam2' ? <MousePointerClick size={14} className="text-[#A855F7]" /> : <Wand2 size={14} className="text-[#A855F7]" />}
        <span className="text-[#FAFAFA] font-medium">
          {phase === 'idle' && (
            engine === 'birefnet'
              ? t('mask.status.detecting')
              : (hasAnyPrompt
                ? t('mask.status.ready', { positive: positiveCount, negative: negativeCount, box: box ? t('mask.status.boxPart') : '' })
                : t('mask.status.step1'))
          )}
          {phase === 'previewing' && (engine === 'sam2' ? t('mask.status.previewingSam2') : t('mask.status.previewingBirefnet'))}
          {phase === 'reviewing' && (engine === 'sam2' ? t('mask.status.reviewSam2') : t('mask.status.reviewBirefnet'))}
          {phase === 'tracking' && (
            engine === 'birefnet'
              ? t('mask.status.maskingAll', { pct: trackPct })
              : t('mask.status.trackingAll', { pct: trackPct })
          )}
          {phase === 'ready' && t('mask.status.step3')}
        </span>
        {engine === 'sam2' && tool === 'points' && (
          <>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-full bg-green-400 border border-white" />
              {t('mask.legend.clickEq')} <b className="text-[#FAFAFA]">{t('mask.legend.subject')}</b>
            </span>
            <span className="flex items-center gap-1.5">
              <kbd className="bg-[#27272A] px-1.5 rounded text-[10px] border border-[#3F3F46]">{t('mask.legend.shiftKey')}</kbd>{t('mask.legend.shiftClickEq')} <b className="text-[#FAFAFA]">{t('mask.legend.background')}</b>
            </span>
            <span className="text-[#71717A]">{t('finish.mask.keyframeHint')}</span>
          </>
        )}
        {keyframes.length > 0 && (
          <span className="flex items-center gap-1">
            {t('finish.mask.keyframes', { list: '' })}
            {keyframes.map((k) => (
              <button key={k} onClick={() => setViewIdx(k)} className={`px-1.5 rounded font-mono ${k === viewIdx ? 'bg-[#A855F7] text-white' : 'bg-[#27272A] hover:bg-[#3F3F46]'}`}>{k + 1}</button>
            ))}
          </span>
        )}
        {correctedFrames.length > 0 && (
          <span className="flex items-center gap-1">
            {t('finish.mask.corrected', { list: '' })}
            {correctedFrames.map((k) => (
              <button key={k} onClick={() => setViewIdx(k)} className={`px-1.5 rounded font-mono ${k === viewIdx ? 'bg-emerald-500 text-white' : 'bg-[#27272A] hover:bg-[#3F3F46]'}`}>{k + 1}</button>
            ))}
          </span>
        )}
      </div>

      {/* canvas */}
      <div className="flex-1 flex items-center justify-center min-h-0">
        <div
          ref={containerRef}
          className="relative max-h-full max-w-full rounded-xl overflow-hidden border-2 border-[#A855F7]/50 shadow-xl shadow-[#A855F7]/30 cursor-crosshair select-none"
          style={{ aspectRatio: `${naturalSize.w} / ${naturalSize.h}`, ...(showCutout ? (viewMode === 'checker' ? CHECKER_BG : { backgroundColor: bgColor }) : {}) }}
          onMouseDown={handleMouseDown}
        >
          {showCutout ? (
            <img src={cutoutUrl!} alt={t('mask.alt.maskOverlay')} className="block w-full h-full object-cover pointer-events-none" draggable={false} />
          ) : (
            <>
              <img src={bgUrl} alt={t('mask.alt.referenceFrame')} className="block w-full h-full object-cover pointer-events-none" draggable={false} />
              {/* The server renders the mask as a "spotlight matte" already
                  (dark outside, transparent inside, magenta outline). Plain
                  alpha compositing — no mix-blend hacks. */}
              {maskUrl && (
                <img
                  src={maskUrl}
                  alt={t('mask.alt.maskOverlay')}
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  draggable={false}
                />
              )}
            </>
          )}
          {/* prompts */}
          {tool === 'points' && points.map((p, i) => {
            const num = points.slice(0, i + 1).filter((q) => q.polarity === p.polarity).length;
            const isPos = p.polarity === 'positive';
            return (
              <div
                key={i}
                className={`absolute rounded-full border-2 border-white shadow-lg shadow-black/70 flex items-center justify-center font-bold text-[11px] ${
                  isPos ? 'bg-green-500 text-white' : 'bg-red-500 text-white'
                }`}
                style={{
                  left: `${(p.x / naturalSize.w) * 100}%`,
                  top: `${(p.y / naturalSize.h) * 100}%`,
                  width: 22, height: 22,
                  transform: 'translate(-50%, -50%)',
                }}
                title={t(isPos ? 'mask.point.subject' : 'mask.point.background', { num })}
              >
                {isPos ? '+' : '−'}
              </div>
            );
          })}
          {/* box */}
          {tool === 'points' && (box ?? dragBox) && (() => {
            const b = (dragBox ?? box)!;
            const [x0, y0, x1, y1] = b;
            return (
              <div
                className="absolute border-2 border-cyan-400 bg-cyan-400/10 pointer-events-none rounded-sm shadow-lg shadow-cyan-400/30"
                style={{
                  left: `${(x0 / naturalSize.w) * 100}%`,
                  top: `${(y0 / naturalSize.h) * 100}%`,
                  width: `${((x1 - x0) / naturalSize.w) * 100}%`,
                  height: `${((y1 - y0) / naturalSize.h) * 100}%`,
                }}
              />
            );
          })()}
          {tool !== 'points' && session && (
            <MaskBrushLayer
              frameW={naturalSize.w}
              frameH={naturalSize.h}
              mode={tool}
              radius={brushRadius}
              disabled={!frameHasMask || phase === 'tracking' || busy}
              onStroke={applyStroke}
            />
          )}
          {/* empty state hint — only in Manual mode; Auto auto-runs */}
          {!hasAnyPrompt && phase === 'idle' && !busy && engine === 'sam2' && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-[#A855F7]/90 text-white text-xs font-semibold px-4 py-2 rounded-full shadow-lg pointer-events-none animate-pulse">
              {t('mask.hint.clickCharacter')}
            </div>
          )}
          {busy && (phase === 'idle' || phase === 'reviewing') && (
            <div className="absolute inset-0 bg-black/60 flex items-center justify-center pointer-events-none">
              <div className="text-[#FAFAFA] text-sm flex items-center gap-2">
                <Loader2 size={16} className="animate-spin" />
                {t('mask.loadingFrame')}
              </div>
            </div>
          )}
          {phase === 'tracking' && (
            <div className="absolute inset-x-0 bottom-0 bg-black/80 p-3">
              <div className="text-white text-xs font-mono mb-1.5">{t('mask.trackingProgress', { pct: trackPct })}</div>
              <div className="h-2 bg-[#27272A] rounded">
                <div className="h-full bg-gradient-to-r from-[#A855F7] to-[#EC4899] rounded transition-all" style={{ width: `${trackPct}%` }} />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* error */}
      {errorMsg && (
        <div className="mt-3 bg-red-500/10 border-2 border-red-500/40 text-red-200 rounded-xl p-3 text-xs font-mono">
          {errorMsg}
        </div>
      )}

      {/* footer */}
      <div className="mt-3 flex items-center gap-3 flex-wrap">
        {/* Frame slider — before tracking it picks the keyframe to click on;
            after tracking it reviews every frame. */}
        {session && nFrames > 1 && (
          <div className="flex items-center gap-3 bg-[#0F0F11] border-2 border-[#27272A] rounded-xl px-4 py-2 flex-1 min-w-[260px]">
            <span className="text-[#A1A1AA] text-xs font-mono whitespace-nowrap">{t('mask.frame')}</span>
            <input
              type="range"
              min={0}
              max={nFrames - 1}
              value={viewIdx}
              disabled={phase === 'tracking'}
              onChange={(e) => { setViewIdx(Number(e.target.value)); if (tracked) setMaskCacheBust((n) => n + 1); }}
              className="flex-1 accent-[#A855F7]"
            />
            <span className="text-[#FAFAFA] text-xs font-mono whitespace-nowrap">
              {viewIdx + 1} / {nFrames}
            </span>
          </div>
        )}

        <div className="flex-1" />

        {/* Manual: "Generate mask" on the current frame (adds a keyframe). */}
        {engine === 'sam2' && tool === 'points' && phase !== 'tracking' && (phase === 'idle' || hasAnyPrompt) && (
          <button
            disabled={busy}
            onClick={runPreview}
            className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-[#A855F7] to-[#EC4899] text-white text-sm font-semibold hover:shadow-lg hover:shadow-[#A855F7]/50 disabled:opacity-50 flex items-center gap-2"
            title={t('mask.repreviewTitle')}
          >
            {busy ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
            {phase === 'idle' ? t('mask.generate') : t('mask.repreview')}
          </button>
        )}

        {(phase === 'reviewing' || phase === 'ready') && (
          <button
            disabled={busy}
            onClick={runTrack}
            title={phase === 'ready' ? t('finish.mask.repropagateTitle') : undefined}
            className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-[#A855F7] to-[#EC4899] text-white text-sm font-semibold hover:shadow-lg hover:shadow-[#A855F7]/50 disabled:opacity-50 flex items-center gap-2"
          >
            {busy ? <Loader2 size={16} className="animate-spin" /> : phase === 'ready' ? <RefreshCw size={16} /> : <Sparkles size={16} />}
            {phase === 'ready'
              ? (engine === 'sam2' || correctedFrames.length > 0 ? t('finish.mask.repropagate') : t('mask.rerunClip'))
              : (engine === 'sam2' ? t('mask.trackClip') : t('mask.runClip'))}
          </button>
        )}

        {phase === 'ready' && (
          <div className="flex items-center gap-2">
            <select
              value={alphaCodec}
              onChange={(e) => setAlphaCodec(e.target.value as typeof alphaCodec)}
              className="bg-[#0F0F11] border-2 border-[#27272A] rounded-lg px-3 py-2 text-[#FAFAFA] text-xs"
            >
              <option value="prores_4444_alpha">{t('mask.codec.prores')}</option>
              <option value="vp9_alpha">{t('mask.codec.vp9')}</option>
            </select>
            <button
              disabled={busy}
              onClick={runExport}
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-green-500 to-emerald-500 text-white text-sm font-semibold hover:shadow-lg hover:shadow-green-500/50 disabled:opacity-50 flex items-center gap-2"
              title={t('mask.exportTitle')}
            >
              {busy ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
              {t('mask.exportAlpha')}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
