import { Loader2 } from 'lucide-react';
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { rotoApi, type RotoProject, type RotoStroke } from '../api/roto';
import type { RotoImageCache } from './rotoImages';
import {
  fitXform, frameToScreen, matteVersion, preloadOrder, screenToFrame, settingsKey, thinPoints, zoomAround,
  type RotoView, type ViewXform,
} from './rotoUtils';

export type RotoTool = 'brush' | 'eraser' | 'click';

export interface RotoClickMark { frame: number; layer: string; x: number; y: number; positive: boolean }
export interface RotoPendingStroke { id: number; frame: number; stroke: RotoStroke }

interface MatteSpec { url: string; color: string }
interface ViewerUrls { frame: string; mattes: MatteSpec[] }

const FALLBACK_TINT = '#ef4444';
const MODE_COLOR: Record<RotoStroke['mode'], string> = {
  add: 'rgba(74, 222, 128, 0.5)',
  remove: 'rgba(248, 113, 113, 0.5)',
  erase: 'rgba(96, 165, 250, 0.45)',
};
const MODE_RING: Record<RotoStroke['mode'], string> = { add: '#4ADE80', remove: '#F87171', erase: '#60A5FA' };

/** URLs the viewer needs for frame i (frame image + the matte(s) of the view). */
export function viewerUrls(p: RotoProject, i: number, view: RotoView, activeLayer: string | null, solo: boolean): ViewerUrls {
  const frame = rotoApi.frameUrl(p.id, i);
  if (view === 'original') return { frame, mattes: [] };
  // Before the first AI pass only painted frames have a matte (others 404).
  if (!p.has_ai_pass && p.frame_versions?.[String(i)] == null) return { frame, mattes: [] };
  const v = matteVersion(p, i);
  const sk = settingsKey(p.settings);
  const layerColor = (id: string | null) => p.layers.find((l) => l.id === id)?.color ?? FALLBACK_TINT;
  if (view === 'overlay' && !solo && p.layers.length > 1) {
    return { frame, mattes: p.layers.map((l) => ({ url: rotoApi.matteUrl(p.id, i, v, l.id, sk), color: l.color || FALLBACK_TINT })) };
  }
  const layer = solo ? activeLayer : null;
  const color = solo ? layerColor(activeLayer) : (p.layers[0]?.color ?? FALLBACK_TINT);
  return { frame, mattes: [{ url: rotoApi.matteUrl(p.id, i, v, layer, sk), color }] };
}

interface Props {
  project: RotoProject;
  frame: number;
  view: RotoView;
  activeLayer: string | null;
  solo: boolean;
  tool: RotoTool;
  brushMode: 'add' | 'remove';
  altHeld: boolean;
  radius: number;
  cache: RotoImageCache;
  marks: RotoClickMark[];
  pendingStrokes: RotoPendingStroke[];
  /** Clicks pick a character for detection instead of editing. */
  pickMode: boolean;
  editDisabled: boolean;
  playing: boolean;
  /** Shared with the editor's keyboard handler: Space held / used for panning. */
  spaceRef: React.MutableRefObject<{ down: boolean; used: boolean }>;
  /** Increment to fit the image to the window. */
  fitSignal: number;
  onStroke: (stroke: RotoStroke) => void;
  onClickPoint: (x: number, y: number, positive: boolean) => void;
  onPick: (x: number, y: number, positive: boolean) => void;
}

type Painted = { fimg: CanvasImageSource | null; mattes: { img: HTMLCanvasElement | null; color: string }[] };

/** Main viewer: composites the working frame and the matte client-side on a
 * canvas, with zoom/pan, and turns mouse gestures into brush strokes / clicks
 * in working-frame pixels. A second canvas on top draws the live stroke, the
 * brush cursor and the click points without recompositing the frame. */
export default function RotoEditorCanvas(props: Props) {
  const {
    project, frame, view, activeLayer, solo, tool, brushMode, altHeld, radius, cache, marks, pendingStrokes,
    pickMode, editDisabled, playing, spaceRef, fitSignal, onStroke, onClickPoint, onPick,
  } = props;
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const baseRef = useRef<HTMLCanvasElement | null>(null);
  const topRef = useRef<HTMLCanvasElement | null>(null);
  const scratchRef = useRef<HTMLCanvasElement | null>(null);
  const checkerRef = useRef<CanvasPattern | null>(null);
  const xfRef = useRef<ViewXform>({ scale: 1, tx: 0, ty: 0 });
  const autoFitRef = useRef(true);
  const sizeRef = useRef({ w: 0, h: 0, dpr: 1 });
  const paintedRef = useRef<Painted>({ fimg: null, mattes: [] });
  const renderToken = useRef(0);
  const liveRef = useRef<{ points: [number, number][]; mode: RotoStroke['mode'] } | null>(null);
  const hoverRef = useRef<[number, number] | null>(null);
  const panRef = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Latest props for the imperative drawing code.
  const live = useRef(props);
  live.current = props;

  const W = project.width;
  const H = project.height;

  // ── drawing ──────────────────────────────────────────────────────────────
  const paintBase = useCallback(() => {
    const c = baseRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const { dpr } = sizeRef.current;
    const xf = xfRef.current;
    const p = live.current;
    const { fimg, mattes } = paintedRef.current;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#09090B';
    ctx.fillRect(0, 0, c.width, c.height);
    if (!fimg) return;
    const w = p.project.width;
    const h = p.project.height;
    ctx.setTransform(dpr * xf.scale, 0, 0, dpr * xf.scale, dpr * xf.tx, dpr * xf.ty);
    ctx.imageSmoothingEnabled = xf.scale < 2;

    let scratch = scratchRef.current;
    if (!scratch) { scratch = document.createElement('canvas'); scratchRef.current = scratch; }
    if (scratch.width !== w || scratch.height !== h) { scratch.width = w; scratch.height = h; }
    const sctx = scratch.getContext('2d')!;

    if (p.view === 'original' || p.view === 'overlay') {
      ctx.drawImage(fimg, 0, 0, w, h);
      if (p.view === 'overlay') {
        for (const m of mattes) {
          if (!m.img) continue;
          sctx.globalCompositeOperation = 'source-over';
          sctx.clearRect(0, 0, w, h);
          sctx.drawImage(m.img, 0, 0, w, h);
          sctx.globalCompositeOperation = 'source-in';
          sctx.fillStyle = m.color;
          sctx.fillRect(0, 0, w, h);
          sctx.globalCompositeOperation = 'source-over';
          ctx.globalAlpha = 0.5;
          ctx.drawImage(scratch, 0, 0, w, h);
          ctx.globalAlpha = 1;
        }
      }
    } else {
      if (p.view === 'black') ctx.fillStyle = '#000000';
      else if (p.view === 'white') ctx.fillStyle = '#FFFFFF';
      else {
        if (!checkerRef.current) {
          const t = document.createElement('canvas');
          t.width = 32; t.height = 32;
          const tc = t.getContext('2d')!;
          tc.fillStyle = '#FFFFFF'; tc.fillRect(0, 0, 32, 32);
          tc.fillStyle = '#C8C8C8'; tc.fillRect(0, 0, 16, 16); tc.fillRect(16, 16, 16, 16);
          checkerRef.current = ctx.createPattern(t, 'repeat');
        }
        ctx.fillStyle = checkerRef.current ?? '#C8C8C8';
      }
      ctx.fillRect(0, 0, w, h);
      const m = mattes[0];
      if (m && m.img) {
        sctx.globalCompositeOperation = 'source-over';
        sctx.clearRect(0, 0, w, h);
        sctx.drawImage(fimg, 0, 0, w, h);
        sctx.globalCompositeOperation = 'destination-in';
        sctx.drawImage(m.img, 0, 0, w, h);
        sctx.globalCompositeOperation = 'source-over';
        ctx.drawImage(scratch, 0, 0, w, h);
      }
    }
    // thin frame border so the edges of the shot stay visible on black
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.strokeStyle = 'rgba(168, 85, 247, 0.45)';
    ctx.lineWidth = 1;
    ctx.strokeRect(dpr * xf.tx - 0.5, dpr * xf.ty - 0.5, dpr * w * xf.scale + 1, dpr * h * xf.scale + 1);
  }, []);

  const paintTop = useCallback(() => {
    const c = topRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const { dpr } = sizeRef.current;
    const xf = xfRef.current;
    const p = live.current;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const drawStroke = (pts: [number, number][], r: number, mode: RotoStroke['mode'], alpha = 1) => {
      if (!pts.length) return;
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = MODE_COLOR[mode];
      ctx.fillStyle = MODE_COLOR[mode];
      ctx.lineWidth = Math.max(1, 2 * r * xf.scale);
      if (pts.length === 1) {
        const [sx, sy] = frameToScreen(pts[0][0], pts[0][1], xf);
        ctx.beginPath(); ctx.arc(sx, sy, Math.max(0.5, r * xf.scale), 0, Math.PI * 2); ctx.fill();
      } else {
        ctx.beginPath();
        pts.forEach(([x, y], k) => {
          const [sx, sy] = frameToScreen(x, y, xf);
          if (k === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
        });
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    };

    for (const ps of p.pendingStrokes) drawStroke(ps.stroke.points, ps.stroke.radius, ps.stroke.mode, 0.7);
    const lv = liveRef.current;
    if (lv) drawStroke(lv.points, p.radius, lv.mode);

    for (const m of p.marks) {
      const [sx, sy] = frameToScreen(m.x, m.y, xf);
      ctx.beginPath();
      ctx.arc(sx, sy, 6, 0, Math.PI * 2);
      ctx.fillStyle = m.positive ? '#22C55E' : '#EF4444';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#FFFFFF';
      ctx.stroke();
    }

    // brush cursor
    const hv = hoverRef.current;
    const wrap = wrapRef.current;
    const panning = !!panRef.current;
    let cursor = 'default';
    if (panning) cursor = 'grabbing';
    else if (p.spaceRef.current.down) cursor = 'grab';
    else if (p.pickMode || p.tool === 'click') cursor = 'crosshair';
    else if (p.editDisabled) cursor = 'not-allowed';
    else cursor = 'none';
    if (wrap && wrap.style.cursor !== cursor) wrap.style.cursor = cursor;
    if (hv && cursor === 'none') {
      const mode: RotoStroke['mode'] = p.tool === 'eraser' ? 'erase' : (p.altHeld || p.brushMode === 'remove') ? 'remove' : 'add';
      const [sx, sy] = frameToScreen(hv[0], hv[1], xf);
      const r = Math.max(2, p.radius * xf.scale);
      ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(0,0,0,0.6)'; ctx.stroke();
      ctx.lineWidth = 1.5; ctx.strokeStyle = MODE_RING[mode]; ctx.stroke();
      ctx.beginPath(); ctx.arc(sx, sy, 1.5, 0, Math.PI * 2); ctx.fillStyle = MODE_RING[mode]; ctx.fill();
    }
  }, []);

  const repaint = useCallback(() => { paintBase(); paintTop(); }, [paintBase, paintTop]);

  // ── sizing ───────────────────────────────────────────────────────────────
  useLayoutEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const apply = () => {
      const r = wrap.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      sizeRef.current = { w: r.width, h: r.height, dpr };
      for (const c of [baseRef.current, topRef.current]) {
        if (!c) continue;
        c.width = Math.max(1, Math.round(r.width * dpr));
        c.height = Math.max(1, Math.round(r.height * dpr));
      }
      if (autoFitRef.current) xfRef.current = fitXform(r.width, r.height, live.current.project.width, live.current.project.height);
      repaint();
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [repaint]);

  useEffect(() => {
    autoFitRef.current = true;
    const { w, h } = sizeRef.current;
    xfRef.current = fitXform(w, h, W, H);
    repaint();
  }, [fitSignal, W, H, repaint]);

  // ── load + composite the current frame ───────────────────────────────────
  const urls = viewerUrls(project, frame, view, activeLayer, solo);
  const urlKey = `${urls.frame}|${urls.mattes.map((m) => m.url + m.color).join('|')}|${view}`;
  useEffect(() => {
    const token = ++renderToken.current;
    const peeked = cache.peekFrame(urls.frame);
    let slowTimer: number | null = null;
    if (peeked === undefined) slowTimer = window.setTimeout(() => { if (token === renderToken.current) setLoading(true); }, 150);
    Promise.all([
      cache.frame(urls.frame),
      Promise.all(urls.mattes.map((m) => cache.matte(m.url).catch(() => null))),
    ]).then(([fimg, mattes]) => {
      if (token !== renderToken.current) return;
      paintedRef.current = { fimg, mattes: mattes.map((img, k) => ({ img, color: urls.mattes[k].color })) };
      setLoadError(null);
      paintBase();
    }).catch((err) => {
      if (token === renderToken.current) setLoadError((err as Error).message);
    }).finally(() => {
      if (slowTimer != null) clearTimeout(slowTimer);
      if (token === renderToken.current) setLoading(false);
    });
    // Preload the neighbours (more ahead while playing).
    const order = preloadOrder(frame, project.n_frames, playing ? 1 : 4, playing ? 12 : 4);
    for (const j of order) {
      const u = viewerUrls(project, j, view, activeLayer, solo);
      cache.frame(u.frame).catch(() => {});
      for (const m of u.mattes) cache.matte(m.url).catch(() => {});
    }
    return () => { if (slowTimer != null) clearTimeout(slowTimer); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlKey, playing]);

  // overlay redraw when its inputs change
  useEffect(() => { paintTop(); }, [marks, pendingStrokes, tool, brushMode, altHeld, radius, pickMode, editDisabled, paintTop]);

  // ── interactions ─────────────────────────────────────────────────────────
  const local = (e: { clientX: number; clientY: number }): [number, number] => {
    const r = wrapRef.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };

  // Wheel zoom around the cursor (native listener: React's is passive).
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const [lx, ly] = local(e);
      const factor = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
      xfRef.current = zoomAround(xfRef.current, lx, ly, factor);
      autoFitRef.current = false;
      repaint();
    };
    wrap.addEventListener('wheel', onWheel, { passive: false });
    return () => wrap.removeEventListener('wheel', onWheel);
  }, [repaint]);

  function startPan(e: React.MouseEvent) {
    const [lx, ly] = local(e);
    panRef.current = { x: lx, y: ly, tx: xfRef.current.tx, ty: xfRef.current.ty };
    autoFitRef.current = false;
    const move = (ev: MouseEvent) => {
      const pr = panRef.current;
      if (!pr) return;
      const [mx, my] = local(ev);
      xfRef.current = { ...xfRef.current, tx: pr.tx + (mx - pr.x), ty: pr.ty + (my - pr.y) };
      repaint();
    };
    const up = () => {
      panRef.current = null;
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
      paintTop();
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
    paintTop();
  }

  function onMouseDown(e: React.MouseEvent) {
    e.preventDefault();
    if (e.button === 1 || (e.button === 0 && spaceRef.current.down)) {
      if (spaceRef.current.down) spaceRef.current.used = true;
      startPan(e);
      return;
    }
    if (e.button !== 0 && e.button !== 2) return;
    const [lx, ly] = local(e);
    const [fx, fy] = screenToFrame(lx, ly, xfRef.current);
    const inside = fx >= 0 && fy >= 0 && fx < W && fy < H;
    const positive = !(e.altKey || e.button === 2);
    if (pickMode) {
      if (inside) onPick(Math.round(fx), Math.round(fy), positive);
      return;
    }
    if (editDisabled) return;
    if (tool === 'click') {
      if (inside) onClickPoint(Math.round(fx), Math.round(fy), positive);
      return;
    }
    if (e.button !== 0) return;
    const mode: RotoStroke['mode'] = tool === 'eraser' ? 'erase' : (e.altKey || brushMode === 'remove') ? 'remove' : 'add';
    liveRef.current = { points: [[fx, fy]], mode };
    paintTop();
    const move = (ev: MouseEvent) => {
      const lv = liveRef.current;
      if (!lv) return;
      const [mx, my] = local(ev);
      const pt = screenToFrame(mx, my, xfRef.current);
      lv.points.push(pt);
      hoverRef.current = pt;
      paintTop();
    };
    const up = () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
      const lv = liveRef.current;
      liveRef.current = null;
      if (lv && lv.points.length) {
        const pts = thinPoints(lv.points.map(([x, y]) => [Math.round(x), Math.round(y)] as [number, number]), 1);
        onStroke({ points: pts, radius: Math.max(1, Math.round(live.current.radius)), mode: lv.mode });
      }
      paintTop();
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  }

  function onMouseMove(e: React.MouseEvent) {
    if (liveRef.current || panRef.current) return;
    const [lx, ly] = local(e);
    hoverRef.current = screenToFrame(lx, ly, xfRef.current);
    paintTop();
  }

  return (
    <div
      ref={wrapRef}
      className="relative w-full h-full overflow-hidden select-none bg-[#09090B]"
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseLeave={() => { hoverRef.current = null; paintTop(); }}
      onContextMenu={(e) => e.preventDefault()}
    >
      <canvas ref={baseRef} data-testid="roto-canvas" className="absolute inset-0 w-full h-full" />
      <canvas ref={topRef} className="absolute inset-0 w-full h-full pointer-events-none" />
      {loading && (
        <div className="absolute top-3 left-3 bg-black/70 text-[#A1A1AA] text-[11px] px-2 py-1 rounded flex items-center gap-1.5 pointer-events-none">
          <Loader2 size={12} className="animate-spin" />
        </div>
      )}
      {loadError && (
        <div className="absolute bottom-3 left-3 bg-red-500/20 border border-red-500/40 text-red-200 text-[11px] font-mono px-2 py-1 rounded pointer-events-none">
          {loadError}
        </div>
      )}
    </div>
  );
}
