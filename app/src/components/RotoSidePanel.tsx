import {
  ArrowLeft, ArrowLeftRight, ArrowRight, Brush, CheckCircle2, Circle, Download, Eraser, FolderOpen, Minus,
  MousePointerClick, Plus, Redo2, RotateCcw, Trash2, Undo2,
} from 'lucide-react';
import type { RotoEdgeSettings, RotoExportFiles, RotoLayer } from '../api/roto';
import { useT } from '../i18n';
import type { RotoTool } from './RotoEditorCanvas';

export const sectionCls = 'bg-[#0F0F11] border-2 border-[#27272A] rounded-xl p-3 space-y-2';
const titleCls = 'text-[10px] uppercase tracking-wider font-bold text-[#71717A]';
const segWrap = 'flex items-center bg-[#18181B] border border-[#27272A] rounded-lg p-0.5';
const segBtn = (on: boolean, accent = 'bg-[#A855F7] text-white') =>
  `flex-1 px-2 py-1.5 rounded text-xs font-semibold flex items-center justify-center gap-1.5 transition-all disabled:opacity-40 ${
    on ? accent : 'text-[#A1A1AA] hover:text-[#FAFAFA]'}`;
const smallBtn = 'px-2.5 py-1.5 rounded-lg bg-[#27272A] hover:bg-[#3F3F46] text-[#D4D4D8] text-xs flex items-center justify-center gap-1.5 disabled:opacity-40 disabled:cursor-not-allowed';

// ── tools ──────────────────────────────────────────────────────────────────
export function ToolsSection(p: {
  tool: RotoTool; setTool: (t: RotoTool) => void;
  brushMode: 'add' | 'remove'; setBrushMode: (m: 'add' | 'remove') => void; altHeld: boolean;
  radius: number; setRadius: (r: number) => void;
  canUndo: boolean; canRedo: boolean; onUndo: () => void; onRedo: () => void;
  disabled: boolean;
}) {
  const t = useT();
  const effMode = p.altHeld ? 'remove' : p.brushMode;
  return (
    <div className={sectionCls}>
      <div className={titleCls}>{t('roto.tools')}</div>
      <div className={segWrap}>
        <button className={segBtn(p.tool === 'brush')} onClick={() => p.setTool('brush')} title={t('roto.tool.brushTitle')} data-testid="roto-tool-brush">
          <Brush size={14} />{t('roto.tool.brush')}
        </button>
        <button className={segBtn(p.tool === 'eraser')} onClick={() => p.setTool('eraser')} title={t('roto.tool.eraserTitle')} data-testid="roto-tool-eraser">
          <Eraser size={14} />{t('roto.tool.eraser')}
        </button>
        <button className={segBtn(p.tool === 'click')} onClick={() => p.setTool('click')} title={t('roto.tool.clickTitle')} data-testid="roto-tool-click">
          <MousePointerClick size={14} />{t('roto.tool.click')}
        </button>
      </div>
      {p.tool === 'brush' && (
        <div className={segWrap}>
          <button className={segBtn(effMode === 'add', 'bg-green-500 text-white')} onClick={() => p.setBrushMode('add')} title={t('roto.brush.addTitle')} data-testid="roto-brush-add">
            <Plus size={14} />{t('roto.brush.add')}
          </button>
          <button className={segBtn(effMode === 'remove', 'bg-red-500 text-white')} onClick={() => p.setBrushMode('remove')} title={t('roto.brush.removeTitle')} data-testid="roto-brush-remove">
            <Minus size={14} />{t('roto.brush.remove')}
          </button>
        </div>
      )}
      {p.tool !== 'click' ? (
        <label className="flex items-center gap-2 text-[11px] text-[#A1A1AA]" title={t('roto.brush.sizeTitle')}>
          <span className="whitespace-nowrap">{t('roto.brush.size')}</span>
          <input type="range" min={1} max={200} value={p.radius} onChange={(e) => p.setRadius(Number(e.target.value))}
                 className="flex-1 accent-[#A855F7]" data-testid="roto-brush-size" />
          <span className="font-mono text-[#FAFAFA] w-8 text-right">{p.radius}</span>
        </label>
      ) : (
        <div className="text-[11px] text-[#A1A1AA] flex items-center gap-2">
          <Circle size={10} className="text-green-400 fill-green-400" />
          <Circle size={10} className="text-red-400 fill-red-400" />
          {t('roto.click.legend')}
        </div>
      )}
      <div className="flex gap-2">
        <button className={`${smallBtn} flex-1`} onClick={p.onUndo} disabled={!p.canUndo || p.disabled} title={t('roto.undo')} data-testid="roto-undo">
          <Undo2 size={14} />{t('roto.undoShort')}
        </button>
        <button className={`${smallBtn} flex-1`} onClick={p.onRedo} disabled={!p.canRedo || p.disabled} title={t('roto.redo')} data-testid="roto-redo">
          <Redo2 size={14} />{t('roto.redoShort')}
        </button>
      </div>
    </div>
  );
}

// ── propagate / validate ───────────────────────────────────────────────────
export function PropagateSection(p: {
  onPropagate: (dir: 'forward' | 'backward' | 'both') => void;
  locked: boolean; onToggleLock: () => void; disabled: boolean; onClearFrame?: () => void;
}) {
  const t = useT();
  const pbtn = 'flex-1 px-2 py-2 rounded-lg bg-gradient-to-r from-[#A855F7]/80 to-[#EC4899]/80 hover:from-[#A855F7] hover:to-[#EC4899] text-white text-xs font-semibold flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed';
  return (
    <div className={sectionCls}>
      <div className={titleCls} title={t('roto.propagateTitle')}>{t('roto.propagate')}</div>
      <div className="flex gap-1.5" title={t('roto.propagateTitle')}>
        <button className={pbtn} disabled={p.disabled} onClick={() => p.onPropagate('backward')} aria-label={t('roto.propagate.backward')}
                title={`${t('roto.propagate.backward')} - ${t('roto.propagateTitle')}`} data-testid="roto-propagate-backward">
          <ArrowLeft size={16} />
        </button>
        <button className={pbtn} disabled={p.disabled} onClick={() => p.onPropagate('both')} aria-label={t('roto.propagate.both')}
                title={`${t('roto.propagate.both')} - ${t('roto.propagateTitle')}`} data-testid="roto-propagate-both">
          <ArrowLeftRight size={16} />
        </button>
        <button className={pbtn} disabled={p.disabled} onClick={() => p.onPropagate('forward')} aria-label={t('roto.propagate.forward')}
                title={`${t('roto.propagate.forward')} - ${t('roto.propagateTitle')}`} data-testid="roto-propagate-forward">
          <ArrowRight size={16} />
        </button>
      </div>
      <p className="text-[10px] text-[#71717A] leading-snug">{t('roto.propagateTitle')}</p>
      <button
        className={`w-full px-3 py-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 border-2 disabled:opacity-40 ${
          p.locked ? 'bg-yellow-400/15 border-yellow-400/60 text-yellow-200' : 'bg-[#27272A] border-transparent hover:bg-[#3F3F46] text-[#D4D4D8]'}`}
        onClick={p.onToggleLock} disabled={p.disabled} title={t('roto.validateTitle')} data-testid="roto-validate" aria-pressed={p.locked}
      >
        <CheckCircle2 size={14} />{p.locked ? t('roto.validated') : t('roto.validate')} <span className="text-[#71717A] font-mono">V</span>
      </button>
      {p.onClearFrame && (
        <button
          className="w-full px-3 py-1.5 rounded-lg text-[11px] font-semibold flex items-center justify-center gap-1.5 bg-[#27272A] hover:bg-[#3F3F46] text-[#A1A1AA] hover:text-[#FAFAFA] disabled:opacity-40"
          onClick={p.onClearFrame} disabled={p.disabled} title={t('roto.clearFrameTitle')} data-testid="roto-clear-frame"
        >
          {t('roto.clearFrame')} <span className="text-[#71717A] font-mono">Suppr</span>
        </button>
      )}
    </div>
  );
}

// ── characters (layers) ────────────────────────────────────────────────────
export function LayersSection(p: {
  layers: RotoLayer[]; active: string | null; onSelect: (id: string) => void;
  onAdd: () => void; onDelete: (l: RotoLayer) => void; solo: boolean; setSolo: (b: boolean) => void; disabled: boolean;
}) {
  const t = useT();
  return (
    <div className={sectionCls}>
      <div className="flex items-center justify-between">
        <div className={titleCls}>{t('roto.layers')}</div>
        <button className="text-[11px] font-semibold text-[#C084FC] hover:text-[#E9D5FF] disabled:opacity-40" onClick={p.onAdd}
                disabled={p.disabled} title={t('roto.layers.addTitle')} data-testid="roto-layer-add">
          {t('roto.layers.add')}
        </button>
      </div>
      <div className="space-y-1" data-testid="roto-layers">
        {p.layers.map((l) => {
          const on = l.id === p.active;
          return (
            <div key={l.id}
                 className={`flex items-center gap-2 px-2 py-1.5 rounded-lg cursor-pointer border ${on ? 'bg-[#27272A] border-[#A855F7]/60' : 'border-transparent hover:bg-[#18181B]'}`}
                 onClick={() => p.onSelect(l.id)} title={on ? t('roto.layers.active') : undefined}
                 data-testid="roto-layer" data-active={on ? 'true' : 'false'}>
              <span className="w-3 h-3 rounded-full border border-white/40 shrink-0" style={{ backgroundColor: l.color }} />
              <span className={`text-xs truncate flex-1 ${on ? 'text-[#FAFAFA] font-semibold' : 'text-[#A1A1AA]'}`}>{l.name}</span>
              {p.layers.length > 1 && (
                <button className="p-1 rounded hover:bg-red-500/20 text-[#71717A] hover:text-red-300 disabled:opacity-40" disabled={p.disabled}
                        onClick={(e) => { e.stopPropagation(); p.onDelete(l); }} title={t('roto.layers.delete')} aria-label={t('roto.layers.delete')}>
                  <Trash2 size={12} />
                </button>
              )}
            </div>
          );
        })}
      </div>
      {p.layers.length > 1 && (
        <label className="flex items-center gap-2 text-[11px] text-[#A1A1AA] cursor-pointer" title={t('roto.soloLayerTitle')}>
          <input type="checkbox" checked={p.solo} onChange={(e) => p.setSolo(e.target.checked)} className="accent-[#A855F7]" />
          {t('roto.soloLayer')}
        </label>
      )}
    </div>
  );
}

// ── edges ──────────────────────────────────────────────────────────────────
function Slider(p: { label: string; title?: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void; fmt: (v: number) => string; testid: string }) {
  return (
    <label className="block text-[11px] text-[#A1A1AA]" title={p.title}>
      <div className="flex items-center justify-between mb-0.5">
        <span>{p.label}</span>
        <span className="font-mono text-[#FAFAFA]">{p.fmt(p.value)}</span>
      </div>
      <input type="range" min={p.min} max={p.max} step={p.step} value={p.value}
             onChange={(e) => p.onChange(Number(e.target.value))} className="w-full accent-[#A855F7]" data-testid={p.testid} />
    </label>
  );
}

export function EdgesSection(p: { settings: RotoEdgeSettings; onChange: (s: RotoEdgeSettings) => void; onReset: () => void }) {
  const t = useT();
  const s = p.settings;
  const signed = (v: number, d = 2) => `${v > 0 ? '+' : ''}${v.toFixed(d)}`;
  return (
    <div className={sectionCls}>
      <div className="flex items-center justify-between">
        <div className={titleCls}>{t('roto.edges')}</div>
        <button className="text-[11px] text-[#A1A1AA] hover:text-[#FAFAFA] flex items-center gap-1" onClick={p.onReset} data-testid="roto-edges-reset">
          <RotateCcw size={11} />{t('roto.edges.reset')}
        </button>
      </div>
      <label className="flex items-center gap-2 text-xs text-[#D4D4D8] cursor-pointer" title={t('roto.edges.keepInkTitle')}>
        <input type="checkbox" checked={s.keep_ink} onChange={(e) => p.onChange({ ...s, keep_ink: e.target.checked })}
               className="accent-[#A855F7]" data-testid="roto-edges-keepink" />
        {t('roto.edges.keepInk')}
      </label>
      <Slider label={t('roto.edges.softness')} min={-1} max={1} step={0.05} value={s.softness}
              onChange={(v) => p.onChange({ ...s, softness: v })} fmt={(v) => signed(v)} testid="roto-edges-softness" />
      <Slider label={t('roto.edges.grow')} min={-6} max={6} step={0.5} value={s.grow_px}
              onChange={(v) => p.onChange({ ...s, grow_px: v })} fmt={(v) => `${signed(v, 1)} px`} testid="roto-edges-grow" />
      <Slider label={t('roto.edges.temporal')} title={t('roto.edges.temporalTitle')} min={0} max={1} step={0.05} value={s.temporal}
              onChange={(v) => p.onChange({ ...s, temporal: v })} fmt={(v) => v.toFixed(2)} testid="roto-edges-temporal" />
    </div>
  );
}

// ── export ─────────────────────────────────────────────────────────────────
export interface RotoExportOptions { prores: boolean; png: boolean; matte: boolean; add_to_selection: boolean }

export function ExportSection(p: {
  opts: RotoExportOptions; setOpts: (o: RotoExportOptions) => void; onExport: () => void;
  disabled: boolean; disabledReason: string | null; files: RotoExportFiles | null; addedToSelection: boolean;
}) {
  const t = useT();
  const o = p.opts;
  const none = !o.prores && !o.png && !o.matte;
  const check = (key: keyof RotoExportOptions, label: string) => (
    <label className="flex items-start gap-2 text-[11px] text-[#D4D4D8] cursor-pointer leading-snug">
      <input type="checkbox" checked={o[key]} onChange={(e) => p.setOpts({ ...o, [key]: e.target.checked })}
             className="accent-[#A855F7] mt-0.5" data-testid={`roto-export-${key}`} />
      <span>{label}</span>
    </label>
  );
  const reveal = window.amvBridge?.reveal;
  const entries = p.files ? (Object.entries(p.files).filter(([, v]) => !!v) as [string, string][]) : [];
  return (
    <div className={sectionCls}>
      <div className={titleCls}>{t('roto.export')}</div>
      {check('prores', t('roto.export.prores'))}
      {check('png', t('roto.export.png'))}
      {check('matte', t('roto.export.matte'))}
      {check('add_to_selection', t('roto.export.selection'))}
      <button
        className="w-full px-4 py-2.5 rounded-xl bg-gradient-to-r from-green-500 to-emerald-500 text-white text-sm font-semibold hover:shadow-lg hover:shadow-green-500/40 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        onClick={p.onExport} disabled={p.disabled || none}
        title={none ? t('roto.export.none') : (p.disabledReason ?? undefined)} data-testid="roto-export"
      >
        <Download size={15} />{t('roto.export')}
      </button>
      {entries.length > 0 && (
        <div className="space-y-1.5 pt-1" data-testid="roto-export-done">
          <div className="text-[11px] text-emerald-300 font-semibold">{t('roto.export.done')}</div>
          {entries.map(([k, path]) => (
            <div key={k} className="flex items-center gap-1.5 text-[10px]">
              <span className="text-[#A1A1AA] shrink-0 w-10">{t(`roto.export.kind.${k}`)}</span>
              <span className="font-mono text-[#D4D4D8] truncate flex-1" title={path}>{path}</span>
              {reveal && (
                <button className="p-1 rounded hover:bg-[#27272A] text-[#A1A1AA] hover:text-[#FAFAFA]" onClick={() => reveal(path)}
                        title={t('roto.export.reveal')} aria-label={t('roto.export.reveal')}>
                  <FolderOpen size={12} />
                </button>
              )}
            </div>
          ))}
          {p.addedToSelection && <div className="text-[10px] text-[#A1A1AA]">{t('roto.export.addedToSelection')}</div>}
        </div>
      )}
    </div>
  );
}
