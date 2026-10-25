import { AlertTriangle, Check, Cpu, Download, Gauge, Loader2, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useT } from '../i18n';
import {
  finishApi, formatBytes, formatDuration, requiredModel, subscribeFinish,
  type FinishCapabilities, type FinishEstimate, type FinishKind, type FinishSource,
} from '../api/finish';

interface FinishDialogProps {
  kind: FinishKind;
  sources: FinishSource[];
  caps: FinishCapabilities | null;
  onCapsChange: () => void;
  onClose: () => void;
}

const DEFAULT_PARAMS: Record<FinishKind, Record<string, unknown>> = {
  upscale: { model: 'realesr-animevideov3', scale: 2, tile: 'auto' },
  interpolate: { mode: 'slowmo', factor: 2, target_fps: 60, skip_cuts: true, audio: 'mute' },
  depth: { format: 'png16', smoothing: 0.5, invert: false },
  matte: { codec: 'prores_4444_alpha', stabilize: true },
};

/** Parameters + 5-frame speed estimate + queueing for one finishing kind. */
export default function FinishDialog({ kind, sources, caps, onCapsChange, onClose }: FinishDialogProps) {
  const t = useT();
  const [params, setParams] = useState<Record<string, unknown>>(() => ({ ...DEFAULT_PARAMS[kind] }));
  const [estimate, setEstimate] = useState<FinishEstimate | null>(null);
  const [estimating, setEstimating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [queued, setQueued] = useState<number | null>(null);
  const [dlPct, setDlPct] = useState<number | null>(null);

  const set = (k: string, v: unknown) => { setParams((p) => ({ ...p, [k]: v })); setEstimate(null); };

  const modelId = requiredModel(kind, params);
  const model = useMemo(() => caps?.models.find((m) => m.id === modelId) ?? null, [caps, modelId]);
  const modelReady = !modelId || !!model?.installed;
  const totalSeconds = useMemo(
    () => sources.reduce((acc, s) => acc + Math.max(0, ((s.end_ms ?? 0) - (s.start_ms ?? 0)) / 1000), 0),
    [sources],
  );

  // Follow the model download started from this dialog.
  useEffect(() => subscribeFinish((e) => {
    if (e.type !== 'finish_model' || e.model_id !== modelId) return;
    if (e.done) {
      setDlPct(null);
      if (e.ok === false) setError(t('finish.models.error', { message: e.message ?? '' }));
      onCapsChange();
    } else {
      setDlPct(e.percent ?? 0);
    }
  }), [modelId, onCapsChange, t]);

  // Escape closes the dialog without closing the MiniEditor underneath.
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.stopPropagation(); onClose(); }
    };
    window.addEventListener('keydown', h, true);
    return () => window.removeEventListener('keydown', h, true);
  }, [onClose]);

  async function download() {
    if (!modelId) return;
    setError(null);
    setDlPct(0);
    try { await finishApi.downloadModel(modelId); } catch (err) { setDlPct(null); setError((err as Error).message); }
  }

  async function runEstimate() {
    setEstimating(true);
    setError(null);
    try {
      setEstimate(await finishApi.estimate(kind, sources[0], params));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setEstimating(false);
    }
  }

  async function submit() {
    setSubmitting(true);
    setError(null);
    try {
      const r = await finishApi.createJobs(kind, sources, params);
      setQueued(r.jobs.length);
      window.dispatchEvent(new CustomEvent('amv-finish-jobs-changed', { detail: { open: true } }));
      setTimeout(onClose, 1400);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  const select = 'bg-[#0F0F11] border-2 border-[#27272A] rounded-lg px-3 py-2 text-[#FAFAFA] text-sm focus:outline-none focus:border-[#8B5CF6]';
  const row = (label: string, control: React.ReactNode, hint?: string) => (
    <div className="flex items-start justify-between gap-4 py-2">
      <div className="min-w-0">
        <div className="text-sm text-[#FAFAFA]">{label}</div>
        {hint && <div className="text-[11px] text-[#71717A] mt-0.5">{hint}</div>}
      </div>
      <div className="flex-shrink-0">{control}</div>
    </div>
  );
  const chips = (key: string, options: { v: unknown; label: string }[]) => (
    <div className="flex bg-[#0F0F11] border-2 border-[#27272A] rounded-lg p-0.5">
      {options.map((o) => (
        <button
          key={String(o.v)}
          onClick={() => set(key, o.v)}
          className={`px-3 py-1 rounded text-xs font-semibold ${params[key] === o.v ? 'bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white' : 'text-[#A1A1AA] hover:text-[#FAFAFA]'}`}
        >{o.label}</button>
      ))}
    </div>
  );

  return (
    <div className="fixed inset-0 z-[70] bg-black/70 backdrop-blur-sm flex items-center justify-center p-6" onClick={onClose}>
      <div className="w-full max-w-lg bg-[#18181B] border-2 border-[#27272A] rounded-2xl shadow-2xl shadow-[#8B5CF6]/20" onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-5 py-4 border-b-2 border-[#27272A]">
          <div>
            <div className="text-[#FAFAFA] font-semibold">{t('finish.dialog.title', { kind: t(`finish.kind.${kind}`) })}</div>
            <div className="text-xs text-[#71717A] mt-0.5">
              {sources.length === 1
                ? t('finish.dialog.clipOne', { seconds: totalSeconds.toFixed(1) })
                : t('finish.dialog.clips', { n: sources.length })}
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-[#27272A] text-[#A1A1AA]"><X size={16} /></button>
        </div>

        <div className="px-5 py-3 divide-y divide-[#27272A]">
          {kind === 'upscale' && (
            <>
              {row(t('finish.param.model'), (
                <select value={String(params.model)} onChange={(e) => set('model', e.target.value)} className={select}>
                  <option value="realesr-animevideov3">{t('finish.param.model.fast')}</option>
                  <option value="realesrgan-x4plus-anime-6b">{t('finish.param.model.quality')}</option>
                </select>
              ))}
              {row(t('finish.param.scale'), chips('scale', [{ v: 2, label: 'x2' }, { v: 4, label: 'x4' }]))}
              {row(t('finish.param.tile'), (
                <select value={String(params.tile)} onChange={(e) => set('tile', e.target.value === 'auto' ? 'auto' : Number(e.target.value))} className={select}>
                  <option value="auto">{t('finish.param.tile.auto')}</option>
                  <option value="128">128 px</option>
                  <option value="256">256 px</option>
                  <option value="512">512 px</option>
                  <option value="0">—</option>
                </select>
              ), t('finish.param.tile.hint'))}
            </>
          )}
          {kind === 'interpolate' && (
            <>
              {row(t('finish.param.mode'), chips('mode', [
                { v: 'slowmo', label: t('finish.param.mode.slowmo') },
                { v: 'fps', label: t('finish.param.mode.fps') },
              ]))}
              {params.mode === 'slowmo'
                ? (
                  <>
                    {row(t('finish.param.factor'), chips('factor', [{ v: 2, label: 'x2' }, { v: 4, label: 'x4' }, { v: 8, label: 'x8' }]))}
                    {row(t('finish.param.audio'), chips('audio', [
                      { v: 'mute', label: t('finish.param.audio.mute') },
                      { v: 'stretch', label: t('finish.param.audio.stretch') },
                    ]))}
                  </>
                )
                : row(t('finish.param.targetFps'), chips('target_fps', [{ v: 48, label: '48' }, { v: 50, label: '50' }, { v: 60, label: '60' }]))}
              {row(t('finish.param.skipCuts'), (
                <input type="checkbox" checked={!!params.skip_cuts} onChange={(e) => set('skip_cuts', e.target.checked)} className="w-4 h-4 accent-[#8B5CF6]" />
              ), t('finish.param.skipCuts.hint'))}
            </>
          )}
          {kind === 'depth' && (
            <>
              {row(t('finish.param.format'), (
                <select value={String(params.format)} onChange={(e) => set('format', e.target.value)} className={select}>
                  <option value="png16">{t('finish.param.format.png16')}</option>
                  <option value="prores">{t('finish.param.format.prores')}</option>
                  <option value="h264">{t('finish.param.format.h264')}</option>
                </select>
              ))}
              {row(t('finish.param.smoothing'), (
                <div className="flex items-center gap-2">
                  <input type="range" min={0} max={100} step={5} value={Math.round(Number(params.smoothing) * 100)}
                    onChange={(e) => set('smoothing', Number(e.target.value) / 100)} className="w-32 accent-[#8B5CF6]" />
                  <span className="text-xs font-mono text-[#A1A1AA] w-8">{Math.round(Number(params.smoothing) * 100)}</span>
                </div>
              ))}
              {row(t('finish.param.invert'), (
                <input type="checkbox" checked={!!params.invert} onChange={(e) => set('invert', e.target.checked)} className="w-4 h-4 accent-[#8B5CF6]" />
              ))}
            </>
          )}
          {kind === 'matte' && (
            <>
              {row(t('finish.param.codec'), (
                <select value={String(params.codec)} onChange={(e) => set('codec', e.target.value)} className={select}>
                  <option value="prores_4444_alpha">ProRes 4444 (.mov)</option>
                  <option value="vp9_alpha">VP9 alpha (.webm)</option>
                </select>
              ))}
              {row(t('finish.param.stabilize'), (
                <input type="checkbox" checked={!!params.stabilize} onChange={(e) => set('stabilize', e.target.checked)} className="w-4 h-4 accent-[#8B5CF6]" />
              ))}
            </>
          )}
        </div>

        <div className="px-5 pb-4 space-y-2">
          {caps && !caps.torch && (
            <div className="flex items-start gap-2 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg p-2.5">
              <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" /> {t('finish.noTorch')}
            </div>
          )}
          {caps?.cpu_only && (
            <div className="flex items-start gap-2 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded-lg p-2.5">
              <Cpu size={14} className="flex-shrink-0 mt-0.5" /> {t('finish.cpuWarning')}
            </div>
          )}
          {modelId && !modelReady && (
            <div className="flex items-center gap-3 text-xs text-[#A1A1AA] bg-[#0F0F11] border border-[#27272A] rounded-lg p-2.5">
              <span className="flex-1">{t('finish.modelMissing', { model: model?.label ?? modelId, size: formatBytes(model?.size ?? 0) })}</span>
              {dlPct !== null || model?.downloading
                ? <span className="flex items-center gap-1.5 text-[#8B5CF6]"><Loader2 size={13} className="animate-spin" />{t('finish.downloading', { pct: dlPct ?? model?.download_percent ?? 0 })}</span>
                : <button onClick={download} className="px-3 py-1.5 rounded-lg bg-[#27272A] hover:bg-[#3F3F46] text-[#FAFAFA] flex items-center gap-1.5"><Download size={13} />{t('finish.download')}</button>}
            </div>
          )}
          {estimate && (
            <div className="flex items-center gap-2 text-xs text-[#A1A1AA]">
              <Gauge size={14} className="text-[#8B5CF6]" />
              {t(sources.length > 1 ? 'finish.estimate.resultMany' : 'finish.estimate.result', {
                total: formatDuration(estimate.estimated_seconds),
                perFrame: estimate.seconds_per_frame.toFixed(2),
                device: estimate.device_name || estimate.device,
              })}
            </div>
          )}
          {error && <div className="text-xs text-red-300 bg-red-500/10 border border-red-500/30 rounded-lg p-2.5 font-mono break-words">{error}</div>}
          {queued !== null && (
            <div className="flex items-center gap-2 text-xs text-emerald-400"><Check size={14} />{t('finish.dialog.queued', { n: queued })}</div>
          )}
          <div className="text-[11px] text-[#71717A]">{t('finish.dialog.output')}</div>
        </div>

        <div className="flex items-center justify-end gap-2 px-5 py-4 border-t-2 border-[#27272A]">
          <button
            onClick={runEstimate}
            disabled={estimating || !modelReady || !caps?.torch}
            className="px-4 py-2 rounded-xl bg-[#27272A] hover:bg-[#3F3F46] text-[#A1A1AA] text-sm flex items-center gap-2 disabled:opacity-40"
          >
            {estimating ? <Loader2 size={14} className="animate-spin" /> : <Gauge size={14} />}
            {estimating ? t('finish.estimate.running') : t('finish.estimate.run')}
          </button>
          <button
            onClick={submit}
            disabled={submitting || !modelReady || !caps?.torch || queued !== null}
            className="px-5 py-2 rounded-xl bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white text-sm font-semibold disabled:opacity-40 flex items-center gap-2"
          >
            {submitting && <Loader2 size={14} className="animate-spin" />}
            {submitting ? t('finish.dialog.starting') : t('finish.dialog.start')}
          </button>
        </div>
      </div>
    </div>
  );
}
