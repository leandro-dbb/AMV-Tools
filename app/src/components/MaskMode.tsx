import { Loader2, X, Eraser, RotateCcw, Sparkles, Download, MousePointerClick, Wand2 } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api/client';
import type { SceneResult, SegmentPreviewResponse } from '../api/types';

type Engine = 'birefnet' | 'sam2';

type Polarity = 'positive' | 'negative';
type Point = { x: number; y: number; polarity: Polarity };
type Box = [number, number, number, number] | null;

type Phase = 'idle' | 'previewing' | 'reviewing' | 'tracking' | 'ready';

interface MaskModeProps {
  scene: SceneResult;
  trimStart: number;
  trimEnd: number;
  onClose: () => void;
  onExportError: (message: string) => void;
  onExportSuccess: (outputPath: string) => void;
}

/** SAM 2 mask + alpha-export panel.
 *
 * Sits inside the MiniEditor's video area when the user toggles "🎭 Mask".
 * Owns its own canvas + click/box state and talks to /api/scene/<id>/segment.
 * The component releases the server session on unmount so the tempdir of
 * extracted frames gets cleaned up. */
export default function MaskMode({ scene, trimStart, trimEnd, onClose, onExportError, onExportSuccess }: MaskModeProps) {
  const [phase, setPhase] = useState<Phase>('idle');
  const [points, setPoints] = useState<Point[]>([]);
  const [box, setBox] = useState<Box>(null);
  const [dragBox, setDragBox] = useState<Box>(null);
  const [session, setSession] = useState<SegmentPreviewResponse | null>(null);
  const [maskCacheBust, setMaskCacheBust] = useState(0);
  const [reviewIdx, setReviewIdx] = useState(0);
  const [trackPct, setTrackPct] = useState(0);
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [alphaCodec, setAlphaCodec] = useState<'prores_4444_alpha' | 'vp9_alpha'>('prores_4444_alpha');
  const [engine, setEngine] = useState<Engine>('birefnet');
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Read the saved engine choice once, so the chip starts on the user's
  // preferred default. The chip lets them flip without leaving the editor.
  useEffect(() => {
    Promise.all([api.getSettings(), api.status()])
      .then(([s, status]) => {
        if (s.models?.mask_engine === 'sam2' && status.device !== 'dml') setEngine('sam2');
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
    api.segmentPreview(scene.id, { positive: [], negative: [], box: null, reference_frame_offset: 0, start_ms: trimStart, end_ms: trimEnd, engine })
      .then((r) => {
        if (cancelled) return;
        setSession(r);
        if (r.has_mask) {
          setPhase('reviewing');
          setMaskCacheBust((n) => n + 1);
        }
      })
      .catch((err) => { if (!cancelled) setErrorMsg(`Init failed: ${(err as Error).message}`); })
      .finally(() => { if (!cancelled) setBusy(false); });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene.id, trimStart, trimEnd, engine]);

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
    if (phase === 'tracking') return;
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
  }, [phase, naturalSize.w, naturalSize.h]);

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
      setErrorMsg('Click at least one point on the subject before previewing.');
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
        reference_frame_offset: session?.reference_frame_offset ?? null,
        start_ms: trimStart,
        end_ms: trimEnd,
        engine,
      });
      setSession(r);
      setMaskCacheBust((n) => n + 1);
      setPhase('reviewing');
    } catch (err) {
      setErrorMsg(`Preview failed: ${(err as Error).message}`);
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
        reference_frame_offset: session.reference_frame_offset,
        start_ms: trimStart,
        end_ms: trimEnd,
        engine,
      });
      setSession({ ...session, n_frames: r.n_frames });
      setReviewIdx(0);
      setMaskCacheBust((n) => n + 1);
      setPhase('ready');
    } catch (err) {
      setErrorMsg(`Tracking failed: ${(err as Error).message}`);
      setPhase('reviewing');
    } finally {
      setBusy(false);
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
      setErrorMsg(`Alpha export failed: ${msg}`);
      onExportError(msg);
    } finally {
      setBusy(false);
    }
  }

  // ── render ──────────────────────────────────────────────────────────────
  const showingReviewFrame = phase === 'ready' && session;
  const bgUrl =
    showingReviewFrame && session
      ? api.segmentFrameUrl(session.session_id, reviewIdx)
      : session
        ? api.segmentFrameUrl(session.session_id, session.reference_frame_offset)
        : api.thumbnailUrl(scene.id);
  const maskUrl =
    phase === 'ready' && session
      ? `${api.segmentMaskUrl(session.session_id, reviewIdx)}?v=${maskCacheBust}`
      : phase === 'reviewing' && session
        ? `${api.segmentMaskPreviewUrl(session.session_id)}?v=${maskCacheBust}`
        : null;

  const positiveCount = points.filter((p) => p.polarity === 'positive').length;
  const negativeCount = points.length - positiveCount;
  const hasAnyPrompt = points.length > 0 || box !== null;

  return (
    <div className="absolute inset-0 z-30 bg-black/85 backdrop-blur-md flex flex-col p-4">
      {/* toolbar */}
      <div className="flex items-center gap-2 mb-3">
        <div className="bg-gradient-to-r from-[#A855F7] to-[#EC4899] text-white text-xs font-bold px-3 py-1.5 rounded-lg uppercase tracking-wider">
          Mask mode
        </div>
        {/* Engine chip — quick A/B switch without leaving the editor. */}
        <div className="flex items-center bg-[#0F0F11] border border-[#27272A] rounded-lg p-0.5">
          {([
            { id: 'birefnet', label: 'Auto', icon: Wand2, title: 'BiRefNet - automatic, recommended for anime' },
            { id: 'sam2', label: 'Manual', icon: MousePointerClick, title: 'SAM 2 - click to pick the subject, useful when there are multiple characters' },
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
        <div className="text-[#FAFAFA] text-sm font-medium ml-2">
          {phase === 'idle' && (
            engine === 'birefnet'
              ? 'Detecting foreground automatically...'
              : (hasAnyPrompt
                ? `Ready: ${positiveCount} subject, ${negativeCount} background${box ? ', 1 box' : ''} - click "Generate mask"`
                : 'Step 1 - click on the character you want to isolate')
          )}
          {phase === 'previewing' && (
            engine === 'sam2'
              ? 'Running SAM 2 on the reference frame…'
              : 'Running BiRefNet on the reference frame…'
          )}
          {phase === 'reviewing' && (
            engine === 'sam2'
              ? 'Step 2 - happy with the mask? Track it across the clip. Or click more to refine.'
              : 'Step 2 - happy with the mask? Run it across the clip. Or switch to Manual to refine.'
          )}
          {phase === 'tracking' && (
            engine === 'birefnet'
              ? `Masking every frame — ${trackPct}%`
              : `Tracking the mask across the clip — ${trackPct}%`
          )}
          {phase === 'ready' && 'Step 3 — scrub to check every frame, re-run if needed, then "Export with alpha"'}
        </div>
        <div className="flex-1" />
        {engine === 'sam2' && (
          <>
            <button
              disabled={phase === 'tracking' || !hasAnyPrompt}
              onClick={removeLastPrompt}
              className="px-3 py-1.5 rounded-lg bg-[#27272A] hover:bg-[#3F3F46] text-[#A1A1AA] text-xs flex items-center gap-1.5 disabled:opacity-40"
              title="Remove last prompt"
            >
              <RotateCcw size={12} /> Undo
            </button>
            <button
              disabled={phase === 'tracking' || !hasAnyPrompt}
              onClick={clearPrompts}
              className="px-3 py-1.5 rounded-lg bg-[#27272A] hover:bg-[#3F3F46] text-[#A1A1AA] text-xs flex items-center gap-1.5 disabled:opacity-40"
              title="Clear all prompts (Esc)"
            >
              <Eraser size={12} /> Clear
            </button>
          </>
        )}
        <button
          onClick={onClose}
          disabled={busy}
          className="ml-2 p-1.5 rounded-lg hover:bg-[#27272A] text-[#A1A1AA] disabled:opacity-40"
          title="Close mask mode"
        >
          <X size={16} />
        </button>
      </div>

      {/* legend — only visible in Manual mode (Auto has nothing to teach) */}
      {engine === 'sam2' ? (
        <div className="flex items-center gap-4 mb-3 text-[11px] text-[#A1A1AA] bg-[#0F0F11]/80 border border-[#27272A] rounded-lg px-4 py-2">
          <MousePointerClick size={14} className="text-[#A855F7]" />
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded-full bg-green-400 border border-white" />
            Click = <b className="text-[#FAFAFA]">subject</b> (keep)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded-full bg-red-500 border border-white" />
            <kbd className="bg-[#27272A] px-1.5 rounded text-[10px] border border-[#3F3F46]">Shift</kbd>+click = <b className="text-[#FAFAFA]">background</b> (drop)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-4 h-3 border-2 border-cyan-400 rounded-sm" />
            Drag = <b className="text-[#FAFAFA]">bounding box</b>
          </span>
          <span className="flex items-center gap-1.5">
            <kbd className="bg-[#27272A] px-1.5 rounded text-[10px] border border-[#3F3F46]">Esc</kbd> clears all
          </span>
        </div>
      ) : (
        <div className="flex items-center gap-2 mb-3 text-[11px] text-[#A1A1AA] bg-[#0F0F11]/80 border border-[#27272A] rounded-lg px-4 py-2">
          <Wand2 size={14} className="text-[#A855F7]" />
          <span><b className="text-[#FAFAFA]">BiRefNet</b> auto-detects the foreground subject. No clicks needed. If you have several characters and want a specific one, switch to <b className="text-[#FAFAFA]">Manual</b> above.</span>
        </div>
      )}

      {/* canvas */}
      <div className="flex-1 flex items-center justify-center min-h-0">
        <div
          ref={containerRef}
          className="relative max-h-full max-w-full rounded-xl overflow-hidden border-2 border-[#A855F7]/50 shadow-xl shadow-[#A855F7]/30 cursor-crosshair select-none"
          style={{ aspectRatio: `${naturalSize.w} / ${naturalSize.h}` }}
          onMouseDown={handleMouseDown}
        >
          <img src={bgUrl} alt="reference frame" className="block w-full h-full object-cover pointer-events-none" draggable={false} />
          {/* The server renders the mask as a "spotlight matte" already
              (dark outside, transparent inside, magenta outline). Plain
              alpha compositing — no mix-blend hacks. */}
          {maskUrl && (
            <img
              src={maskUrl}
              alt="mask overlay"
              className="absolute inset-0 w-full h-full pointer-events-none"
              draggable={false}
            />
          )}
          {/* prompts */}
          {points.map((p, i) => {
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
                title={`${isPos ? 'Subject' : 'Background'} #${num}`}
              >
                {isPos ? '+' : '−'}
              </div>
            );
          })}
          {/* box */}
          {(box ?? dragBox) && (() => {
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
          {/* empty state hint — only in Manual mode; Auto auto-runs */}
          {!hasAnyPrompt && phase === 'idle' && !busy && engine === 'sam2' && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-[#A855F7]/90 text-white text-xs font-semibold px-4 py-2 rounded-full shadow-lg pointer-events-none animate-pulse">
              Click on the character you want
            </div>
          )}
          {busy && phase === 'idle' && (
            <div className="absolute inset-0 bg-black/60 flex items-center justify-center pointer-events-none">
              <div className="text-[#FAFAFA] text-sm flex items-center gap-2">
                <Loader2 size={16} className="animate-spin" />
                Loading reference frame…
              </div>
            </div>
          )}
          {phase === 'tracking' && (
            <div className="absolute inset-x-0 bottom-0 bg-black/80 p-3">
              <div className="text-white text-xs font-mono mb-1.5">Tracking… {trackPct}%</div>
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
        {/* Frame slider — only after tracking */}
        {phase === 'ready' && session && (
          <div className="flex items-center gap-3 bg-[#0F0F11] border-2 border-[#27272A] rounded-xl px-4 py-2 flex-1 min-w-[260px]">
            <span className="text-[#A1A1AA] text-xs font-mono whitespace-nowrap">Frame</span>
            <input
              type="range"
              min={0}
              max={session.n_frames - 1}
              value={reviewIdx}
              onChange={(e) => { setReviewIdx(Number(e.target.value)); setMaskCacheBust((n) => n + 1); }}
              className="flex-1 accent-[#A855F7]"
            />
            <span className="text-[#FAFAFA] text-xs font-mono whitespace-nowrap">
              {reviewIdx + 1} / {session.n_frames}
            </span>
          </div>
        )}

        <div className="flex-1" />

        {/* Manual-only: "Generate mask" while in idle. Auto auto-ran on mount,
            so it never shows this button. */}
        {phase === 'idle' && engine === 'sam2' && (
          <button
            disabled={busy}
            onClick={runPreview}
            className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-[#A855F7] to-[#EC4899] text-white text-sm font-semibold hover:shadow-lg hover:shadow-[#A855F7]/50 disabled:opacity-50 flex items-center gap-2"
          >
            {busy ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
            Generate mask
          </button>
        )}

        {(phase === 'reviewing' || phase === 'ready') && (
          <>
            {engine === 'sam2' && (
              <button
                disabled={busy}
                onClick={runPreview}
                className="px-4 py-2.5 rounded-xl bg-[#27272A] hover:bg-[#3F3F46] text-[#A1A1AA] text-sm flex items-center gap-2 disabled:opacity-40"
                title="Re-run preview on the reference frame"
              >
                <Sparkles size={14} /> Re-preview
              </button>
            )}
            <button
              disabled={busy}
              onClick={runTrack}
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-[#A855F7] to-[#EC4899] text-white text-sm font-semibold hover:shadow-lg hover:shadow-[#A855F7]/50 disabled:opacity-50 flex items-center gap-2"
            >
              {busy && phase !== 'ready' ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
              {phase === 'ready'
                ? (engine === 'sam2' ? 'Re-track from prompts' : 'Re-run across clip')
                : (engine === 'sam2' ? 'Track across clip' : 'Run across clip')}
            </button>
          </>
        )}

        {phase === 'ready' && (
          <div className="flex items-center gap-2">
            <select
              value={alphaCodec}
              onChange={(e) => setAlphaCodec(e.target.value as typeof alphaCodec)}
              className="bg-[#0F0F11] border-2 border-[#27272A] rounded-lg px-3 py-2 text-[#FAFAFA] text-xs"
            >
              <option value="prores_4444_alpha">ProRes 4444 (.mov)</option>
              <option value="vp9_alpha">VP9 alpha (.webm)</option>
            </select>
            <button
              disabled={busy}
              onClick={runExport}
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-green-500 to-emerald-500 text-white text-sm font-semibold hover:shadow-lg hover:shadow-green-500/50 disabled:opacity-50 flex items-center gap-2"
              title="Encode the clip with the SAM 2 mask as the alpha channel"
            >
              {busy ? <Loader2 size={16} className="animate-spin" /> : <Download size={16} />}
              Export with alpha
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
