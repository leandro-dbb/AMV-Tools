// Export the edit as a timeline for Premiere / AE / Resolve / Vegas / FCP, or
// only the beat + passage markers to edit by hand.
import { useState } from 'react';
import { AlertTriangle, ChevronRight, Clapperboard, Flag, FolderOpen, Loader2 } from 'lucide-react';
import { musicApi } from '../api/music';
import type { MusicAnalysis, MusicExportResult, MusicMarkerFormat, MusicTimelineFormat, ProposedClip } from '../api/music';
import { useI18n } from '../i18n';
import { dirname, exportExtension, safeFilename, warningKey } from './musicUtils';

interface Props {
  analysis: MusicAnalysis;
  clips: ProposedClip[] | null;
}

const TIMELINE_FORMATS: MusicTimelineFormat[] = ['fcp7xml', 'fcpxml', 'edl', 'otio'];
const MARKER_FORMATS: MusicMarkerFormat[] = ['fcp7xml', 'fcpxml', 'edl', 'otio', 'csv'];

const selectCls = 'bg-[#0F0F11] border border-[#27272A] rounded-lg px-3 py-2 text-sm text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6] w-full';
const inputCls = 'bg-[#0F0F11] border border-[#27272A] rounded-lg px-3 py-2 text-xs font-mono text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6] w-full';

async function pickOutput(defaultName: string, format: MusicMarkerFormat, fallback: string, label: string): Promise<string | null> {
  const ext = exportExtension(format);
  if (window.amvBridge?.saveFileDialog) {
    return window.amvBridge.saveFileDialog({
      defaultPath: `${defaultName}.${ext}`,
      filters: [{ name: label, extensions: [ext] }],
    });
  }
  const p = fallback.trim();
  return p || null;
}

function withExt(path: string, format: MusicMarkerFormat): string {
  const ext = exportExtension(format);
  return /\.[a-z0-9]{2,6}$/i.test(path) ? path.replace(/\.[a-z0-9]{2,6}$/i, `.${ext}`) : `${path}.${ext}`;
}

export default function MusicExportPanel({ analysis, clips }: Props) {
  const { t, lang } = useI18n();
  const hasBridge = !!window.amvBridge?.saveFileDialog;
  const stem = safeFilename(analysis.name, 'amv-music');
  const baseDir = dirname(analysis.path);
  const sep = baseDir.includes('\\') ? '\\' : '/';
  const defaultFallback = (suffix: string, f: MusicMarkerFormat) =>
    `${baseDir ? baseDir + sep : ''}${stem}${suffix}.${exportExtension(f)}`;

  const [tlFormat, setTlFormat] = useState<MusicTimelineFormat>('fcp7xml');
  const [beatMarkers, setBeatMarkers] = useState(true);
  const [sourceAudio, setSourceAudio] = useState(false);
  const [tlPath, setTlPath] = useState(() => defaultFallback(' - AMV', 'fcp7xml'));
  const [mkFormat, setMkFormat] = useState<MusicMarkerFormat>('csv');
  const [mkBeats, setMkBeats] = useState(true);
  const [mkPath, setMkPath] = useState(() => defaultFallback(' - markers', 'csv'));

  const [busy, setBusy] = useState<'timeline' | 'markers' | null>(null);
  const [result, setResult] = useState<MusicExportResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canTimeline = !!clips && clips.length > 0;

  async function exportTimeline() {
    if (!clips || busy) return;
    setError(null); setResult(null);
    const out = await pickOutput(`${stem} - AMV`, tlFormat, tlPath, t(`music.export.format.${tlFormat}`));
    if (!out) return;
    setBusy('timeline');
    try {
      const r = await musicApi.exportTimeline({
        hash: analysis.hash, clips, format: tlFormat, output_path: out,
        beat_markers: beatMarkers, include_source_audio: sourceAudio, lang,
      });
      setResult(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function exportMarkers() {
    if (busy) return;
    setError(null); setResult(null);
    const out = await pickOutput(`${stem} - markers`, mkFormat, mkPath, t(`music.export.format.${mkFormat}`));
    if (!out) return;
    setBusy('markers');
    try {
      const r = await musicApi.exportMarkers({ hash: analysis.hash, format: mkFormat, output_path: out, beat_markers: mkBeats, lang });
      setResult(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-4 flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-gradient-to-br from-[#8B5CF6] to-[#EC4899]">
          <Clapperboard size={16} className="text-white" />
        </div>
        <div className="text-[#FAFAFA] font-bold">{t('music.export.title')}</div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* timeline */}
        <div className="flex flex-col gap-2.5 bg-[#0F0F11]/60 border border-[#27272A] rounded-lg p-3">
          <div className="text-sm font-semibold text-[#FAFAFA]">{t('music.export.timeline')}</div>
          <div className="text-[11px] text-[#71717A] -mt-1.5">{t('music.export.timelineHint')}</div>
          <label className="flex flex-col gap-1">
            <span className="text-[11px] uppercase tracking-wider text-[#71717A]">{t('music.export.format')}</span>
            <select value={tlFormat} onChange={(e) => { const f = e.target.value as MusicTimelineFormat; setTlFormat(f); setTlPath((p) => withExt(p, f)); }} className={selectCls}>
              {TIMELINE_FORMATS.map((f) => <option key={f} value={f}>{t(`music.export.format.${f}`)}</option>)}
            </select>
          </label>
          <Check checked={beatMarkers} onChange={setBeatMarkers} label={t('music.export.beatMarkers')} />
          <Check checked={sourceAudio} onChange={setSourceAudio} label={t('music.export.sourceAudio')} />
          {!hasBridge && (
            <label className="flex flex-col gap-1">
              <span className="text-[11px] uppercase tracking-wider text-[#71717A]">{t('music.export.outputPath')}</span>
              <input value={tlPath} onChange={(e) => setTlPath(e.target.value)} className={inputCls} />
            </label>
          )}
          {!canTimeline && <div className="text-[11px] text-amber-300/90">{t('music.export.needEdit')}</div>}
          <button type="button" onClick={exportTimeline} disabled={!canTimeline || !!busy}
                  className="mt-auto flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-bold bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white disabled:opacity-40 disabled:cursor-not-allowed">
            {busy === 'timeline' ? <Loader2 size={14} className="animate-spin" /> : <Clapperboard size={14} />}
            {busy === 'timeline' ? t('music.export.exporting') : t('music.export.timeline')}
          </button>
        </div>

        {/* markers only */}
        <div className="flex flex-col gap-2.5 bg-[#0F0F11]/60 border border-[#27272A] rounded-lg p-3">
          <div className="text-sm font-semibold text-[#FAFAFA]">{t('music.export.markers')}</div>
          <div className="text-[11px] text-[#71717A] -mt-1.5">{t('music.export.markersHint')}</div>
          <label className="flex flex-col gap-1">
            <span className="text-[11px] uppercase tracking-wider text-[#71717A]">{t('music.export.format')}</span>
            <select value={mkFormat} onChange={(e) => { const f = e.target.value as MusicMarkerFormat; setMkFormat(f); setMkPath((p) => withExt(p, f)); }} className={selectCls}>
              {MARKER_FORMATS.map((f) => <option key={f} value={f}>{t(`music.export.format.${f}`)}</option>)}
            </select>
          </label>
          <Check checked={mkBeats} onChange={setMkBeats} label={t('music.export.beatMarkers')} />
          {!hasBridge && (
            <label className="flex flex-col gap-1">
              <span className="text-[11px] uppercase tracking-wider text-[#71717A]">{t('music.export.outputPath')}</span>
              <input value={mkPath} onChange={(e) => setMkPath(e.target.value)} className={inputCls} />
            </label>
          )}
          <button type="button" onClick={exportMarkers} disabled={!!busy}
                  className="mt-auto flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold bg-[#27272A] hover:bg-[#3F3F46] text-[#FAFAFA] disabled:opacity-40 disabled:cursor-not-allowed">
            {busy === 'markers' ? <Loader2 size={14} className="animate-spin" /> : <Flag size={14} />}
            {busy === 'markers' ? t('music.export.exporting') : t('music.export.markers')}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/40 text-red-300 text-xs rounded-lg px-3 py-2 break-words">
          {t('music.export.failed', { message: error })}
        </div>
      )}

      {result && (
        <div className="bg-emerald-500/10 border border-emerald-500/40 text-emerald-300 text-xs rounded-lg px-3 py-2 flex flex-col gap-1">
          <div className="break-all">{t('music.export.saved', { path: result.output })}</div>
          <div className="text-emerald-200/80">
            {result.clip_count != null
              ? t('music.export.counts', { clips: result.clip_count, markers: result.marker_count, fps: result.sequence_fps })
              : t('music.export.markerCount', { markers: result.marker_count })}
          </div>
          {(result.warnings ?? []).length > 0 && (
            <ul className="text-amber-200 list-none space-y-0.5 mt-1">
              {result.warnings.map((w, i) => (
                <li key={i} className="flex gap-1.5"><AlertTriangle size={12} className="shrink-0 mt-0.5" />{t(warningKey(w).key, warningKey(w).vars)}</li>
              ))}
            </ul>
          )}
          {window.amvBridge?.reveal && (
            <button type="button" onClick={() => window.amvBridge?.reveal(result.output)}
                    className="self-start mt-1 inline-flex items-center gap-1 underline hover:text-emerald-200">
              <FolderOpen size={12} /> {t('music.export.reveal')} <ChevronRight size={12} />
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function Check({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="flex items-center gap-2 text-xs text-[#A1A1AA] cursor-pointer">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="accent-[#8B5CF6]" />
      <span>{label}</span>
    </label>
  );
}
