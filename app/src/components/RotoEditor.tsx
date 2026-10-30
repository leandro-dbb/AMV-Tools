import {
  AlertTriangle, CheckCircle2, Cpu, HelpCircle, Loader2, Maximize, Maximize2, Minimize, MousePointerClick, Sparkles, Users, User, X,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  rotoApi, type RotoDetectBody, type RotoEdgeSettings, type RotoEditResult, type RotoExportFiles, type RotoExportFormat,
  type RotoJob, type RotoLayer, type RotoProject, type RotoStroke,
} from '../api/roto';
import { useT } from '../i18n';
import RotoEditorCanvas, { viewerUrls, type RotoClickMark, type RotoPendingStroke, type RotoTool } from './RotoEditorCanvas';
import { RotoImageCache } from './rotoImages';
import { EdgesSection, ExportSection, LayersSection, PropagateSection, ToolsSection, type RotoExportOptions } from './RotoSidePanel';
import RotoTimeline from './RotoTimeline';
import {
  DEFAULT_EDGE_SETTINGS, ROTO_VIEWS, clamp, isTypingTarget, lockedFrames, nextSuspect, pendingSuspects, prevSuspect,
  stepRadius, type RotoView,
} from './rotoUtils';

interface RotoEditorProps {
  sceneId: number;
  startMs: number;
  endMs: number;
  onClose: () => void;
}

type Target = 'all' | 'main' | 'click';

// The sidecar's job messages (English), shown translated.
const JOB_MESSAGES: Record<string, string> = {
  'Cutting out the characters': 'roto.msg.cutting',
  'Looking for frames to check': 'roto.msg.suspects',
  'Following the corrections along the shot': 'roto.msg.following',
  'Writing the files': 'roto.msg.writing',
};
interface PickPoint { x: number; y: number; positive: boolean }

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** Full-window roto editor for one shot: the AI does a first cut-out of the
 * characters, the user fixes it with a brush / clicks, propagates the fixes
 * through the shot, checks the flagged frames and exports with transparency.
 * Everything is saved server-side after every change. */
export default function RotoEditor({ sceneId, startMs, endMs, onClose }: RotoEditorProps) {
  const t = useT();
  const rootRef = useRef<HTMLDivElement | null>(null);
  const cache = useMemo(() => new RotoImageCache(), []);

  const [project, setProjectState] = useState<RotoProject | null>(null);
  const projectRef = useRef<RotoProject | null>(null);
  const setProject = useCallback((p: RotoProject | null) => { projectRef.current = p; setProjectState(p); }, []);
  const patchProject = useCallback((fn: (p: RotoProject) => RotoProject) => {
    const cur = projectRef.current;
    if (!cur) return null;
    const next = fn(cur);
    projectRef.current = next;
    setProjectState(next);
    return next;
  }, []);

  const [openError, setOpenError] = useState<string | null>(null);
  const [openNonce, setOpenNonce] = useState(0);
  const [frame, setFrameState] = useState(0);
  const [activeLayer, setActiveLayer] = useState<string | null>(null);
  const [solo, setSolo] = useState(false);
  const [view, setView] = useState<RotoView>('overlay');
  const [tool, setTool] = useState<RotoTool>('brush');
  const [brushMode, setBrushMode] = useState<'add' | 'remove'>('add');
  const [altHeld, setAltHeld] = useState(false);
  const [radius, setRadius] = useState(16);
  const [job, setJob] = useState<RotoJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [pendingCount, setPendingCount] = useState(0);
  const [pendingStrokes, setPendingStrokes] = useState<RotoPendingStroke[]>([]);
  const [marks, setMarks] = useState<RotoClickMark[]>([]);
  const [latency, setLatency] = useState<{ server: number; total: number } | null>(null);
  const [playing, setPlaying] = useState(false);
  const [target, setTarget] = useState<Target>('all');
  const [armClick, setArmClick] = useState(false);
  const [armPoint, setArmPoint] = useState<{ frame: number; x: number; y: number } | null>(null);
  const [layerPick, setLayerPick] = useState<{ layer: string; frame: number; points: PickPoint[] } | null>(null);
  const [fitSignal, setFitSignal] = useState(0);
  const [fullscreen, setFullscreen] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [settings, setSettings] = useState<RotoEdgeSettings>(DEFAULT_EDGE_SETTINGS);
  const [exportOpts, setExportOpts] = useState<RotoExportOptions>({ prores: true, png: true, matte: true, add_to_selection: true });
  const [exportResult, setExportResult] = useState<{ files: RotoExportFiles; added: boolean } | null>(null);
  const spaceRef = useRef({ down: false, used: false });
  const strokeIdRef = useRef(0);
  const queueRef = useRef<Promise<void>>(Promise.resolve());
  const settingsSeq = useRef(0);
  const settingsTimer = useRef<number | null>(null);

  const nFrames = project?.n_frames ?? 0;
  const setFrame = useCallback((f: number) => {
    const n = projectRef.current?.n_frames ?? 0;
    setFrameState(n > 0 ? clamp(Math.round(f), 0, n - 1) : 0);
  }, []);

  // ── open / resume the project ────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    setOpenError(null);
    rotoApi.open({ scene_id: sceneId, start_ms: startMs, end_ms: endMs })
      .then((p) => {
        if (cancelled) return;
        setProject(p);
        setSettings(p.settings ?? DEFAULT_EDGE_SETTINGS);
        setActiveLayer(p.layers[0]?.id ?? null);
        if (p.exports && Object.values(p.exports).some(Boolean)) setExportResult({ files: p.exports, added: false });
        if (p.job && !p.job.done) setJob(p.job);
      })
      .catch((e) => { if (!cancelled) setOpenError(errMsg(e)); });
    return () => { cancelled = true; };
  }, [sceneId, startMs, endMs, openNonce, setProject]);

  const reloadProject = useCallback(async () => {
    const cur = projectRef.current;
    if (!cur) return;
    try {
      const p = await rotoApi.get(cur.id);
      setProject(p);
      setActiveLayer((a) => (a && p.layers.some((l) => l.id === a) ? a : p.layers[0]?.id ?? null));
    } catch (e) { setError(errMsg(e)); }
  }, [setProject]);

  // ── background jobs (detect / propagate / export) ────────────────────────
  const finishJob = useCallback((j: RotoJob | null) => {
    void reloadProject();
    if (!j) return;
    if (j.error) setError(t('roto.job.error', { message: j.error }));
    else if (j.cancelled) setNotice(t('roto.job.cancelled'));
    else if (j.kind === 'export' && j.result) {
      setExportResult({ files: (j.result.files ?? {}) as RotoExportFiles, added: !!j.result.added_to_selection });
    }
  }, [reloadProject, t]);

  const jobRunning = !!job && !job.done;
  useEffect(() => {
    const id = project?.id;
    if (!id || !jobRunning) return;
    let stop = false;
    let timer: number | undefined;
    const tick = async () => {
      try {
        const j = await rotoApi.job(id);
        if (stop) return;
        setJob(j);
        if (!j || j.done) { finishJob(j); return; }
      } catch { /* transient: keep polling */ }
      if (!stop) timer = window.setTimeout(tick, 300);
    };
    timer = window.setTimeout(tick, 300);
    return () => { stop = true; if (timer) clearTimeout(timer); };
  }, [project?.id, jobRunning, finishJob]);

  const startJob = useCallback(async (run: () => Promise<RotoJob>) => {
    setError(null);
    setNotice(null);
    setPlaying(false);
    try {
      const j = await run();
      setJob(j);
      if (j.done) finishJob(j);
    } catch (e) { setError(errMsg(e)); }
  }, [finishJob]);

  const cancelJob = useCallback(() => {
    const p = projectRef.current;
    if (p) rotoApi.cancel(p.id).catch((e) => setError(errMsg(e)));
  }, []);

  const detect = useCallback((body: RotoDetectBody) => {
    const p = projectRef.current;
    if (!p) return;
    void startJob(() => rotoApi.detect(p.id, body));
  }, [startJob]);

  // ── gestures: queued, never dropped ──────────────────────────────────────
  const enqueue = useCallback((run: () => Promise<void>) => {
    setPendingCount((c) => c + 1);
    queueRef.current = queueRef.current
      .then(run)
      .catch((e) => setError(errMsg(e)))
      .finally(() => setPendingCount((c) => c - 1));
  }, []);

  // The view settings the canvas uses, kept in refs so queued gestures can
  // preload the matte they produced before dropping the live stroke.
  const viewRef = useRef({ view, activeLayer, solo });
  viewRef.current = { view, activeLayer, solo };

  const applyEdit = useCallback(async (r: RotoEditResult, t0: number) => {
    const next = patchProject((p) => ({
      ...p,
      version: r.version,
      frame_versions: { ...p.frame_versions, [String(r.frame)]: r.version },
      keyframes: r.keyframes ?? p.keyframes,
      suspects: r.suspects ?? p.suspects,
      can_undo: true,
      can_redo: false,
    }));
    if (next) {
      // Wait for the corrected matte so the stroke never "blinks" away first.
      const v = viewRef.current;
      const u = viewerUrls(next, r.frame, v.view, v.activeLayer, v.solo);
      await Promise.all(u.mattes.map((m) => cache.matte(m.url).catch(() => null)));
    }
    setLatency({ server: Math.round(r.ms), total: Math.round(performance.now() - t0) });
  }, [patchProject, cache]);

  const layerId = activeLayer ?? project?.layers[0]?.id ?? '';
  const editDisabled = !project || jobRunning;

  const onStroke = useCallback((stroke: RotoStroke) => {
    const p = projectRef.current;
    if (!p || !layerId) return;
    const id = ++strokeIdRef.current;
    const f = frame;
    setPlaying(false);
    setPendingStrokes((s) => [...s, { id, frame: f, stroke }]);
    enqueue(async () => {
      const t0 = performance.now();
      try {
        const r = await rotoApi.brush(p.id, { frame: f, layer: layerId, strokes: [stroke] });
        await applyEdit(r, t0);
      } finally {
        setPendingStrokes((s) => s.filter((x) => x.id !== id));
      }
    });
  }, [layerId, frame, enqueue, applyEdit]);

  const onClickPoint = useCallback((x: number, y: number, positive: boolean) => {
    const p = projectRef.current;
    if (!p || !layerId) return;
    const f = frame;
    setPlaying(false);
    setMarks((m) => [...m, { frame: f, layer: layerId, x, y, positive }]);
    enqueue(async () => {
      const t0 = performance.now();
      const r = await rotoApi.click(p.id, { frame: f, layer: layerId, x, y, positive });
      await applyEdit(r, t0);
    });
  }, [layerId, frame, enqueue, applyEdit]);

  const undoRedo = useCallback((which: 'undo' | 'redo') => {
    const p = projectRef.current;
    if (!p || jobRunning) return;
    if (which === 'undo' ? !p.can_undo : !p.can_redo) return;
    enqueue(async () => {
      const r = await (which === 'undo' ? rotoApi.undo(p.id) : rotoApi.redo(p.id));
      patchProject((cur) => ({
        ...cur,
        version: r.version,
        frame_versions: { ...cur.frame_versions, [String(r.frame)]: r.version },
        can_undo: r.can_undo,
        can_redo: r.can_redo,
        keyframes: r.keyframes ?? cur.keyframes,
        suspects: r.suspects ?? cur.suspects,
      }));
      setMarks((m) => m.filter((k) => k.frame !== r.frame));
      if (r.frame != null && r.frame >= 0) setFrame(r.frame);
      // An undone propagation touches many frames: refresh the per-frame versions.
      await reloadProject();
    });
  }, [enqueue, patchProject, reloadProject, setFrame, jobRunning]);

  const lockedNow = useMemo(() => lockedFrames(project?.keyframes, layerId), [project?.keyframes, layerId]);
  const isLocked = lockedNow.includes(frame);

  const toggleLock = useCallback(() => {
    const p = projectRef.current;
    if (!p || !layerId || jobRunning) return;
    const f = frame;
    const locked = !lockedFrames(p.keyframes, layerId).includes(f);
    enqueue(async () => {
      const r = await rotoApi.lock(p.id, { frame: f, layer: layerId, locked });
      patchProject((cur) => ({ ...cur, keyframes: r.keyframes ?? cur.keyframes, suspects: r.suspects ?? cur.suspects }));
    });
  }, [layerId, frame, enqueue, patchProject, jobRunning]);

  const propagate = useCallback((direction: 'forward' | 'backward' | 'both') => {
    const p = projectRef.current;
    if (!p || !layerId || jobRunning) return;
    const f = frame;
    // Let the pending corrections land first.
    void queueRef.current.then(() => startJob(() => rotoApi.propagate(p.id, { frame: f, layer: layerId, direction })));
  }, [layerId, frame, startJob, jobRunning]);

  // ── edge settings (debounced PUT) ────────────────────────────────────────
  const changeSettings = useCallback((s: RotoEdgeSettings) => {
    setSettings(s);
    const seq = ++settingsSeq.current;
    if (settingsTimer.current) clearTimeout(settingsTimer.current);
    settingsTimer.current = window.setTimeout(() => {
      const p = projectRef.current;
      if (!p) return;
      rotoApi.putSettings(p.id, s)
        .then((np) => {
          if (seq !== settingsSeq.current) return;
          // keep the client-side edits (gestures may have landed meanwhile)
          patchProject((cur) => ({ ...cur, settings: np.settings ?? s, version: Math.max(cur.version, np.version), frame_versions: np.frame_versions ?? cur.frame_versions }));
        })
        .catch((e) => setError(errMsg(e)));
    }, 150);
  }, [patchProject]);
  useEffect(() => () => { if (settingsTimer.current) clearTimeout(settingsTimer.current); }, []);

  // ── layers ───────────────────────────────────────────────────────────────
  const addLayer = useCallback(async () => {
    const p = projectRef.current;
    if (!p) return;
    try {
      const before = new Set(p.layers.map((l) => l.id));
      const np = await rotoApi.addLayer(p.id);
      setProject(np);
      const added = np.layers.find((l) => !before.has(l.id)) ?? np.layers[np.layers.length - 1];
      if (added) {
        setActiveLayer(added.id);
        setLayerPick({ layer: added.id, frame, points: [] });
        setArmClick(false);
      }
    } catch (e) { setError(errMsg(e)); }
  }, [frame, setProject]);

  const deleteLayer = useCallback(async (l: RotoLayer) => {
    const p = projectRef.current;
    if (!p) return;
    if (!window.confirm(t('roto.layers.deleteConfirm', { name: l.name }))) return;
    try {
      const np = await rotoApi.deleteLayer(p.id, l.id);
      setProject(np);
      setActiveLayer((a) => (a && np.layers.some((x) => x.id === a) ? a : np.layers[0]?.id ?? null));
      setLayerPick((lp) => (lp && lp.layer === l.id ? null : lp));
    } catch (e) { setError(errMsg(e)); }
  }, [setProject, t]);

  // ── picking a character for detection ────────────────────────────────────
  const pickMode = armClick || !!layerPick;
  useEffect(() => {
    // points belong to one frame: moving clears them
    setLayerPick((lp) => (lp && lp.frame !== frame ? { ...lp, frame, points: [] } : lp));
  }, [frame]);

  const onPick = useCallback((x: number, y: number, positive: boolean) => {
    if (layerPick) {
      setLayerPick({ ...layerPick, frame, points: [...layerPick.points, { x, y, positive }] });
      return;
    }
    if (armClick && positive) {
      setArmClick(false);
      setArmPoint({ frame, x, y });
      detect({ target: 'click', frame, positive: [[x, y]], ...(layerId ? { layer: layerId } : {}) });
    }
  }, [layerPick, armClick, frame, detect, layerId]);

  const detectLayer = useCallback(() => {
    if (!layerPick) return;
    const pos = layerPick.points.filter((q) => q.positive).map((q) => [q.x, q.y] as [number, number]);
    const neg = layerPick.points.filter((q) => !q.positive).map((q) => [q.x, q.y] as [number, number]);
    if (!pos.length) return;
    detect({ layer: layerPick.layer, target: 'click', frame: layerPick.frame, positive: pos, negative: neg });
    setLayerPick(null);
  }, [layerPick, detect]);

  const onDetectButton = useCallback(() => {
    if (target === 'click') { setArmClick(true); return; }
    detect({ target });
  }, [target, detect]);

  // clear the "clicked character" dot once detection is over
  useEffect(() => { if (!jobRunning) setArmPoint(null); }, [jobRunning]);

  // ── playback ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!playing || !nFrames) return;
    const fps = project?.fps && project.fps > 0 ? project.fps : 24;
    const h = window.setInterval(() => {
      setFrameState((f) => (f + 1 >= nFrames ? 0 : f + 1));
    }, 1000 / fps);
    return () => clearInterval(h);
  }, [playing, nFrames, project?.fps]);
  const togglePlay = useCallback(() => setPlaying((p) => !p), []);

  // ── suspects ─────────────────────────────────────────────────────────────
  const suspects = project?.suspects ?? [];
  const pending = useMemo(() => pendingSuspects(suspects, lockedNow), [suspects, lockedNow]);
  const goNextSuspect = useCallback(() => {
    const f = nextSuspect(pending.length ? pending : suspects, frame);
    if (f != null) { setPlaying(false); setFrame(f); }
  }, [pending, suspects, frame, setFrame]);
  const goPrevSuspect = useCallback(() => {
    const f = prevSuspect(pending.length ? pending : suspects, frame);
    if (f != null) { setPlaying(false); setFrame(f); }
  }, [pending, suspects, frame, setFrame]);

  // ── fullscreen ───────────────────────────────────────────────────────────
  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else rootRef.current?.requestFullscreen?.().catch(() => {});
  }, []);
  useEffect(() => {
    const h = () => { setFullscreen(!!document.fullscreenElement); setFitSignal((n) => n + 1); };
    document.addEventListener('fullscreenchange', h);
    return () => document.removeEventListener('fullscreenchange', h);
  }, []);
  const close = useCallback(() => {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    onClose();
  }, [onClose]);

  // ── export ───────────────────────────────────────────────────────────────
  const doExport = useCallback(() => {
    const p = projectRef.current;
    if (!p) return;
    const formats: RotoExportFormat[] = [];
    if (exportOpts.prores) formats.push('prores');
    if (exportOpts.png) formats.push('png');
    if (exportOpts.matte) formats.push('matte');
    if (!formats.length) return;
    setExportResult(null);
    void queueRef.current.then(() => startJob(() => rotoApi.exportProject(p.id, { formats, add_to_selection: exportOpts.add_to_selection })));
  }, [exportOpts, startJob]);

  // ── keyboard ─────────────────────────────────────────────────────────────
  const keyDown = useRef<(e: KeyboardEvent) => void>(() => {});
  const keyUp = useRef<(e: KeyboardEvent) => void>(() => {});
  keyDown.current = (e: KeyboardEvent) => {
    if (isTypingTarget(e.target as HTMLElement | null)) return;
    // The roto editor owns the keyboard: the MiniEditor underneath must not react.
    e.stopPropagation();
    const k = e.key;
    if (k === 'Alt' || k === 'AltGraph') { e.preventDefault(); setAltHeld(true); return; }
    if (e.code === 'Space' || k === ' ') {
      e.preventDefault();
      if (!e.repeat && !spaceRef.current.down) spaceRef.current = { down: true, used: false };
      return;
    }
    if (showHelp && (k === 'Escape' || k === '?')) { e.preventDefault(); setShowHelp(false); return; }
    if (k === 'Escape') {
      if (document.fullscreenElement) return;
      if (layerPick) { setLayerPick(null); return; }
      if (armClick) { setArmClick(false); return; }
      close();
      return;
    }
    // [ and ] first: on AZERTY they come with AltGr (= Ctrl+Alt on Windows).
    if (k === '[' || k === ']') { e.preventDefault(); setRadius((r) => stepRadius(r, k === ']' ? 1 : -1)); return; }
    if (e.ctrlKey || e.metaKey) {
      const lk = k.toLowerCase();
      if (lk === 'z') { e.preventDefault(); undoRedo(e.shiftKey ? 'redo' : 'undo'); }
      else if (lk === 'y') { e.preventDefault(); undoRedo('redo'); }
      return;
    }
    if (k === '?') { e.preventDefault(); setShowHelp((s) => !s); return; }
    const digit = /^(Digit|Numpad)([0-9])$/.exec(e.code)?.[2] ?? (/^[0-9]$/.test(k) ? k : null);
    if (digit != null && !e.shiftKey) {
      e.preventDefault();
      const d = Number(digit);
      if (d === 0) setFitSignal((n) => n + 1);
      else if (d <= ROTO_VIEWS.length) setView(ROTO_VIEWS[d - 1]);
      return;
    }
    switch (k) {
      case 'ArrowLeft': e.preventDefault(); setPlaying(false); setFrame(frame - 1); return;
      case 'ArrowRight': e.preventDefault(); setPlaying(false); setFrame(frame + 1); return;
      case 'Home': e.preventDefault(); setPlaying(false); setFrame(0); return;
      case 'End': e.preventDefault(); setPlaying(false); setFrame(nFrames - 1); return;
      default: break;
    }
    switch (k.toLowerCase()) {
      case 'b': setTool('brush'); break;
      case 'e': setTool('eraser'); break;
      case 'c': setTool('click'); break;
      case 'f': toggleFullscreen(); break;
      case 'p': propagate('both'); break;
      case 'v': toggleLock(); break;
      case 'n': if (e.shiftKey) goPrevSuspect(); else goNextSuspect(); break;
      default: return;
    }
    e.preventDefault();
  };
  keyUp.current = (e: KeyboardEvent) => {
    if (isTypingTarget(e.target as HTMLElement | null)) return;
    e.stopPropagation();
    if (e.key === 'Alt' || e.key === 'AltGraph') { e.preventDefault(); setAltHeld(false); return; }
    if (e.code === 'Space' || e.key === ' ') {
      e.preventDefault();
      const s = spaceRef.current;
      spaceRef.current = { down: false, used: false };
      if (s.down && !s.used) togglePlay();
    }
  };
  useEffect(() => {
    const down = (e: KeyboardEvent) => keyDown.current(e);
    const up = (e: KeyboardEvent) => keyUp.current(e);
    const blur = () => { setAltHeld(false); spaceRef.current = { down: false, used: false }; };
    window.addEventListener('keydown', down, true);
    window.addEventListener('keyup', up, true);
    window.addEventListener('blur', blur);
    return () => {
      window.removeEventListener('keydown', down, true);
      window.removeEventListener('keyup', up, true);
      window.removeEventListener('blur', blur);
    };
  }, []);

  // notices fade out
  useEffect(() => {
    if (!notice) return;
    const h = window.setTimeout(() => setNotice(null), 4000);
    return () => clearTimeout(h);
  }, [notice]);

  // ── derived view data ────────────────────────────────────────────────────
  const visibleMarks = useMemo<RotoClickMark[]>(() => {
    const out = marks.filter((m) => m.frame === frame && m.layer === layerId);
    if (layerPick && layerPick.frame === frame) {
      for (const q of layerPick.points) out.push({ frame, layer: layerPick.layer, x: q.x, y: q.y, positive: q.positive });
    }
    if (armPoint && armPoint.frame === frame) out.push({ frame, layer: layerId, x: armPoint.x, y: armPoint.y, positive: true });
    return out;
  }, [marks, frame, layerId, layerPick, armPoint]);
  const visibleStrokes = useMemo(() => pendingStrokes.filter((s) => s.frame === frame), [pendingStrokes, frame]);

  const m = project?.models;
  const deviceLabel = m ? t(`roto.device.${m.device}`) : '';
  const modelLine = m ? [
    t('roto.models.characters', { name: m.characters ?? m.detector }),
    m.tracker ? t('roto.models.tracker', { name: m.tracker }) : t('roto.models.noTracker'),
    deviceLabel,
  ].join(' · ') : '';
  const layerPickLayer = layerPick ? project?.layers.find((l) => l.id === layerPick.layer) : null;
  const [redoPass, setRedoPass] = useState(false);
  const showFirstPass = !!project && (!project.has_ai_pass || redoPass) && !jobRunning && !layerPick;
  useEffect(() => { if (jobRunning) setRedoPass(false); }, [jobRunning]);

  const jobTitle = job ? t(`roto.job.${job.kind}`) : '';

  // ── render ───────────────────────────────────────────────────────────────
  const iconBtn = 'p-2 rounded-lg hover:bg-[#27272A] text-[#A1A1AA] hover:text-[#FAFAFA]';
  const viewLabels: Record<RotoView, [string, string]> = {
    overlay: [t('roto.view.overlay'), t('roto.view.overlayTitle')],
    black: [t('roto.view.black'), t('roto.view.blackTitle')],
    white: [t('roto.view.white'), t('roto.view.whiteTitle')],
    checker: [t('roto.view.checker'), t('roto.view.checkerTitle')],
    original: [t('roto.view.original'), t('roto.view.originalTitle')],
  };

  const content = (
    <div ref={rootRef} className="fixed inset-0 z-[70] bg-[#0A0A0B] flex flex-col text-[#FAFAFA]" data-testid="roto-editor"
         onClick={(e) => e.stopPropagation()}>
      {/* top bar */}
      <div className="flex items-center gap-3 px-4 py-2 border-b-2 border-[#27272A] bg-[#18181B] min-h-[52px]">
        <div className="bg-gradient-to-r from-[#A855F7] to-[#EC4899] text-white text-xs font-bold px-3 py-1.5 rounded-lg uppercase tracking-wider shrink-0">
          {t('roto.badge')}
        </div>
        <div className="min-w-0 flex flex-col">
          <div className="text-sm font-semibold truncate" data-testid="roto-title">{project?.name ?? t('roto.opening')}</div>
          {m && (
            <div className="text-[11px] text-[#A1A1AA] flex items-center gap-1.5 truncate" title={t('roto.models.clicksTitle', { name: m.clicks })} data-testid="roto-models">
              <Cpu size={11} className="shrink-0" />{modelLine}
            </div>
          )}
        </div>
        {project?.has_ai_pass && !jobRunning && (
          <button className="text-[11px] px-2.5 py-1 rounded-lg bg-[#27272A] hover:bg-[#3F3F46] text-[#A1A1AA] hover:text-[#FAFAFA] whitespace-nowrap"
                  onClick={() => setRedoPass((r) => !r)} title={t('roto.redoPassTitle')} data-testid="roto-redo-pass">
            {t('roto.redoPass')}
          </button>
        )}
        <div className="flex-1" />
        {pendingCount > 0 && (
          <span className="text-[11px] text-[#C084FC] flex items-center gap-1.5 whitespace-nowrap" data-testid="roto-pending">
            <Loader2 size={12} className="animate-spin" />
            {pendingCount > 1 ? t('roto.pendingQueue', { count: pendingCount - 1 }) : t('roto.pending')}
          </span>
        )}
        {latency && pendingCount === 0 && (
          <span className="text-[11px] text-emerald-300 whitespace-nowrap" data-testid="roto-latency"
                title={t('roto.latencyTitle', { server: latency.server, total: latency.total })}>
            {t('roto.latency', { ms: latency.total, server: latency.server })}
          </span>
        )}
        {project && (
          <span className="text-[11px] text-[#71717A] flex items-center gap-1 whitespace-nowrap" data-testid="roto-saved">
            {pendingCount > 0 ? <Loader2 size={11} className="animate-spin" /> : <CheckCircle2 size={11} className="text-emerald-400" />}
            {pendingCount > 0 ? t('roto.saving') : t('roto.saved')}
          </span>
        )}
        <button className={iconBtn} onClick={() => setShowHelp((s) => !s)} title={t('roto.help')} aria-label={t('roto.help')} data-testid="roto-help">
          <HelpCircle size={18} />
        </button>
        <button className={iconBtn} onClick={toggleFullscreen} title={fullscreen ? t('roto.exitFullscreen') : t('roto.fullscreen')}
                aria-label={fullscreen ? t('roto.exitFullscreen') : t('roto.fullscreen')} data-testid="roto-fullscreen">
          {fullscreen ? <Minimize size={18} /> : <Maximize size={18} />}
        </button>
        <button className={`${iconBtn} hover:text-[#EC4899]`} onClick={close} title={t('roto.closeTitle')} aria-label={t('roto.close')} data-testid="roto-close">
          <X size={20} />
        </button>
      </div>

      {m && (m.sam3_missing || m.slow_machine) && (
        <div className="flex items-center gap-x-6 gap-y-0.5 flex-wrap px-4 py-1.5 bg-amber-500/10 border-b border-amber-500/30">
          {m.sam3_missing && (
            <div className="text-[11px] text-amber-300 flex items-center gap-1.5" data-testid="roto-sam3-missing">
              <AlertTriangle size={11} className="shrink-0" />{t('roto.models.sam3Missing', { model: m.tracker ?? m.clicks })}
            </div>
          )}
          {m.slow_machine && (
            <div className="text-[11px] text-amber-300 flex items-center gap-1.5" data-testid="roto-slow-machine">
              <AlertTriangle size={11} className="shrink-0" />{t('roto.models.slow')}
            </div>
          )}
        </div>
      )}

      {openError && (
        <div className="flex-1 flex items-center justify-center p-6">
          <div className="bg-red-500/10 border-2 border-red-500/40 text-red-200 rounded-xl p-5 max-w-lg text-sm space-y-3">
            <div>{t('roto.openError', { message: openError })}</div>
            <button className="px-4 py-2 rounded-lg bg-[#27272A] hover:bg-[#3F3F46] text-[#FAFAFA] text-xs" onClick={() => setOpenNonce((n) => n + 1)}>
              {t('roto.retry')}
            </button>
          </div>
        </div>
      )}
      {!project && !openError && (
        <div className="flex-1 flex items-center justify-center text-[#A1A1AA] text-sm gap-2">
          <Loader2 size={16} className="animate-spin" />{t('roto.opening')}
        </div>
      )}

      {project && (
        <div className="flex-1 min-h-0 flex">
          {/* viewer + timeline */}
          <div className="flex-1 min-w-0 flex flex-col p-3 gap-2">
            <div className="flex items-center gap-2 flex-wrap">
              <div className="flex items-center bg-[#0F0F11] border border-[#27272A] rounded-lg p-0.5" data-testid="roto-views">
                {ROTO_VIEWS.map((v, k) => (
                  <button key={v} onClick={() => setView(v)} title={viewLabels[v][1]} data-testid={`roto-view-${v}`}
                          className={`px-2.5 py-1 rounded text-xs ${view === v ? 'bg-[#3F3F46] text-white' : 'text-[#A1A1AA] hover:text-[#FAFAFA]'}`}>
                    <span className="text-[#71717A] font-mono mr-1">{k + 1}</span>{viewLabels[v][0]}
                  </button>
                ))}
              </div>
              <button className="px-2.5 py-1 rounded-lg bg-[#0F0F11] border border-[#27272A] text-xs text-[#A1A1AA] hover:text-[#FAFAFA] flex items-center gap-1.5"
                      onClick={() => setFitSignal((n) => n + 1)} title={t('roto.fitTitle')} data-testid="roto-fit">
                <Maximize2 size={12} />{t('roto.fit')} <span className="text-[#71717A] font-mono">0</span>
              </button>
              {notice && <span className="text-[11px] text-[#A1A1AA]">{notice}</span>}
            </div>

            <div className="relative flex-1 min-h-0 rounded-xl overflow-hidden border-2 border-[#A855F7]/30">
              <RotoEditorCanvas
                project={project} frame={frame} view={view} activeLayer={layerId || null} solo={solo}
                tool={tool} brushMode={brushMode} altHeld={altHeld} radius={radius} cache={cache}
                marks={visibleMarks} pendingStrokes={visibleStrokes}
                pickMode={pickMode} editDisabled={editDisabled} playing={playing}
                spaceRef={spaceRef} fitSignal={fitSignal}
                onStroke={onStroke} onClickPoint={onClickPoint} onPick={onPick}
              />

              {/* first AI pass call to action */}
              {showFirstPass && !armClick && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/55 pointer-events-none">
                  <div className="pointer-events-auto bg-[#18181B]/95 border-2 border-[#A855F7]/50 rounded-2xl p-6 w-[min(560px,92%)] shadow-2xl shadow-[#A855F7]/30 space-y-4"
                       data-testid="roto-first-pass">
                    <div>
                      <div className="text-lg font-bold">{t('roto.first.title')}</div>
                      <div className="text-xs text-[#A1A1AA] mt-1">{t(redoPass ? 'roto.redoPassHint' : 'roto.first.subtitle')}</div>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      {([
                        ['all', Users, t('roto.target.all'), t('roto.target.allHint')],
                        ['main', User, t('roto.target.main'), t('roto.target.mainHint')],
                        ['click', MousePointerClick, t('roto.target.click'), t('roto.target.clickHint')],
                      ] as const).map(([key, Icon, label, hint]) => (
                        <button key={key} onClick={() => setTarget(key)} data-testid={`roto-target-${key}`} aria-pressed={target === key}
                                className={`rounded-xl p-3 text-left border-2 transition-all ${target === key
                                  ? 'border-[#A855F7] bg-[#A855F7]/15' : 'border-[#27272A] bg-[#0F0F11] hover:border-[#3F3F46]'}`}>
                          <Icon size={18} className={target === key ? 'text-[#C084FC]' : 'text-[#71717A]'} />
                          <div className="text-xs font-semibold mt-1.5">{label}</div>
                          <div className="text-[10px] text-[#71717A] mt-0.5">{hint}</div>
                        </button>
                      ))}
                    </div>
                    <button onClick={onDetectButton} data-testid="roto-detect"
                            className="w-full px-5 py-3 rounded-xl bg-gradient-to-r from-[#A855F7] to-[#EC4899] text-white text-base font-bold hover:shadow-lg hover:shadow-[#A855F7]/50 flex items-center justify-center gap-2">
                      <Sparkles size={18} />{t('roto.detect')}
                    </button>
                  </div>
                </div>
              )}

              {armClick && !jobRunning && (
                <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-[#A855F7]/90 text-white text-xs font-semibold px-4 py-2 rounded-full shadow-lg flex items-center gap-3"
                     data-testid="roto-click-prompt">
                  <MousePointerClick size={14} />{t('roto.detectClickPrompt')}
                  <button className="underline opacity-80 hover:opacity-100" onClick={() => setArmClick(false)}>{t('roto.job.cancel')}</button>
                </div>
              )}

              {layerPick && !jobRunning && (
                <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-[#18181B]/95 border-2 border-[#A855F7]/60 rounded-xl px-4 py-3 shadow-xl w-[min(520px,92%)] space-y-2"
                     data-testid="roto-layer-pick">
                  <div className="text-sm font-semibold flex items-center gap-2">
                    {layerPickLayer && <span className="w-3 h-3 rounded-full" style={{ backgroundColor: layerPickLayer.color }} />}
                    {t('roto.layerPick.title', { name: layerPickLayer?.name ?? '' })}
                  </div>
                  <div className="text-[11px] text-[#A1A1AA]">{t('roto.layerPick.hint')}</div>
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] text-[#71717A] flex-1">{t('roto.layerPick.points', { count: layerPick.points.length })}</span>
                    <button className="px-3 py-1.5 rounded-lg bg-[#27272A] hover:bg-[#3F3F46] text-xs text-[#A1A1AA] disabled:opacity-40"
                            disabled={!layerPick.points.length} onClick={() => setLayerPick({ ...layerPick, points: [] })}>
                      {t('roto.layerPick.clear')}
                    </button>
                    <button className="px-3 py-1.5 rounded-lg bg-[#27272A] hover:bg-[#3F3F46] text-xs text-[#A1A1AA]" onClick={() => setLayerPick(null)}>
                      {t('roto.layerPick.cancel')}
                    </button>
                    <button className="px-4 py-1.5 rounded-lg bg-gradient-to-r from-[#A855F7] to-[#EC4899] text-white text-xs font-bold disabled:opacity-40 flex items-center gap-1.5"
                            disabled={!layerPick.points.some((q) => q.positive)} onClick={detectLayer} data-testid="roto-detect-layer">
                      <Sparkles size={13} />{t('roto.detect')}
                    </button>
                  </div>
                </div>
              )}

              {/* running job */}
              {jobRunning && job && (
                <div className="absolute inset-x-0 bottom-0 bg-black/80 px-4 py-3 space-y-1.5" data-testid="roto-progress">
                  <div className="flex items-center gap-3 text-xs">
                    <Loader2 size={14} className="animate-spin text-[#C084FC]" />
                    <span className="font-semibold">{jobTitle}</span>
                    <span className="text-[#A1A1AA] truncate flex-1">{JOB_MESSAGES[job.message] ? t(JOB_MESSAGES[job.message]) : job.message}</span>
                    <span className="font-mono">{Math.round(job.percent)}%</span>
                    <button className="px-3 py-1 rounded-lg bg-[#27272A] hover:bg-[#3F3F46] text-[#FAFAFA]" onClick={cancelJob} data-testid="roto-cancel">
                      {t('roto.job.cancel')}
                    </button>
                  </div>
                  <div className="h-2 bg-[#27272A] rounded">
                    <div className="h-full bg-gradient-to-r from-[#A855F7] to-[#EC4899] rounded transition-all" style={{ width: `${clamp(job.percent, 0, 100)}%` }} />
                  </div>
                </div>
              )}

              {error && (
                <div className="absolute top-3 right-3 max-w-md bg-red-500/15 border-2 border-red-500/50 text-red-100 rounded-xl px-3 py-2 text-xs flex items-start gap-2 shadow-lg"
                     data-testid="roto-error">
                  <AlertTriangle size={14} className="shrink-0 mt-0.5" />
                  <span className="flex-1 break-words">{t('roto.error', { message: error })}</span>
                  <button className="opacity-70 hover:opacity-100" onClick={() => setError(null)} aria-label={t('roto.dismiss')}><X size={14} /></button>
                </div>
              )}
            </div>

            <RotoTimeline
              nFrames={project.n_frames} frame={frame} keyframes={lockedNow} suspects={suspects} pendingCount={pending.length}
              playing={playing} onSeek={(f) => { setPlaying(false); setFrame(f); }} onTogglePlay={togglePlay}
              onNextSuspect={goNextSuspect} onPrevSuspect={goPrevSuspect}
            />
          </div>

          {/* side panel */}
          <div className="w-[300px] shrink-0 border-l-2 border-[#27272A] bg-[#18181B]/60 overflow-y-auto p-3 space-y-3">
            <ToolsSection
              tool={tool} setTool={setTool} brushMode={brushMode} setBrushMode={setBrushMode} altHeld={altHeld}
              radius={radius} setRadius={setRadius}
              canUndo={project.can_undo} canRedo={project.can_redo}
              onUndo={() => undoRedo('undo')} onRedo={() => undoRedo('redo')} disabled={editDisabled}
            />
            <PropagateSection onPropagate={propagate} locked={isLocked} onToggleLock={toggleLock} disabled={editDisabled} />
            <LayersSection
              layers={project.layers} active={layerId || null} onSelect={(id) => setActiveLayer(id)}
              onAdd={() => void addLayer()} onDelete={(l) => void deleteLayer(l)} solo={solo} setSolo={setSolo} disabled={jobRunning}
            />
            <EdgesSection settings={settings} onChange={changeSettings} onReset={() => changeSettings(DEFAULT_EDGE_SETTINGS)} />
            <ExportSection
              opts={exportOpts} setOpts={setExportOpts} onExport={doExport}
              disabled={jobRunning || !project.has_ai_pass}
              disabledReason={jobRunning ? t('roto.busyJob') : !project.has_ai_pass ? t('roto.export.needPass') : null}
              files={exportResult?.files ?? null} addedToSelection={!!exportResult?.added}
            />
          </div>
        </div>
      )}

      {showHelp && <ShortcutsOverlay onClose={() => setShowHelp(false)} />}
    </div>
  );

  return createPortal(content, document.body);
}

function ShortcutsOverlay({ onClose }: { onClose: () => void }) {
  const t = useT();
  const rows: [string, string][] = [
    ['B', t('roto.keys.brush')],
    ['E', t('roto.keys.eraser')],
    ['C', t('roto.keys.click')],
    ['Alt', t('roto.keys.alt')],
    ['[ / ]', t('roto.keys.size')],
    ['Ctrl+Z', t('roto.keys.undo')],
    [t('roto.keys.redoKeys'), t('roto.keys.redo')],
    ['1 – 5', t('roto.keys.views')],
    ['0', t('roto.keys.fit')],
    [t('roto.keys.wheel'), t('roto.keys.zoom')],
    [t('roto.keys.spaceDrag'), t('roto.keys.pan')],
    [t('roto.keys.space'), t('roto.keys.play')],
    ['← / →', t('roto.keys.frames')],
    [t('roto.keys.homeEnd'), t('roto.keys.ends')],
    [t('roto.keys.shiftN'), t('roto.keys.suspects')],
    ['P', t('roto.keys.propagate')],
    ['V', t('roto.keys.validate')],
    ['F', t('roto.keys.fullscreen')],
    ['?', t('roto.keys.help')],
    ['Esc', t('roto.keys.close')],
  ];
  return (
    <div className="absolute inset-0 z-10 bg-black/70 flex items-center justify-center" onClick={onClose} data-testid="roto-help-overlay">
      <div className="bg-[#18181B] border-2 border-[#27272A] rounded-2xl p-5 w-[min(520px,92%)] max-h-[85%] overflow-y-auto shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-3">
          <div className="text-base font-bold">{t('roto.keys.title')}</div>
          <button className="p-1.5 rounded-lg hover:bg-[#27272A] text-[#A1A1AA]" onClick={onClose} aria-label={t('roto.close')}><X size={16} /></button>
        </div>
        <table className="w-full text-xs">
          <tbody>
            {rows.map(([k, label]) => (
              <tr key={k} className="border-t border-[#27272A]">
                <td className="py-1.5 pr-4 whitespace-nowrap">
                  <kbd className="bg-[#27272A] px-1.5 py-0.5 rounded text-[11px] border border-[#3F3F46] font-mono">{k}</kbd>
                </td>
                <td className="py-1.5 text-[#D4D4D8]">{label}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
