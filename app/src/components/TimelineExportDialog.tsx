import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Check, ChevronRight, Clapperboard, Loader2, X } from 'lucide-react';
import { api } from '../api/client';
import type {
  TimelineExportResult,
  TimelineFormat,
  TimelineOptions,
  TimelinePreview,
  TimelineSelection,
} from '../api/types';
import { useT } from '../i18n';

export interface TimelineExportDialogProps {
  selection: TimelineSelection;
  defaultName?: string;
  scopeLabel?: string;
  onClose: () => void;
  onExported?: (result: TimelineExportResult) => void;
}

interface Option { value: string; label?: string; labelKey?: string }

const FPS_OPTIONS: Option[] = [
  { value: 'first', labelKey: 'timeline.dialog.firstClip' },
  { value: '24000/1001', label: '23.976' },
  { value: '24', label: '24' },
  { value: '25', label: '25' },
  { value: '30000/1001', label: '29.97' },
  { value: '30', label: '30' },
  { value: '60000/1001', label: '59.94' },
  { value: '60', label: '60' },
];

const RESOLUTION_OPTIONS: Option[] = [
  { value: 'first', labelKey: 'timeline.dialog.firstClip' },
  { value: '1280x720', label: '1280×720' },
  { value: '1920x1080', label: '1920×1080' },
  { value: '2560x1440', label: '2560×1440' },
  { value: '3840x2160', label: '3840×2160' },
];

const FORMAT_I18N: Record<TimelineFormat['id'], string> = {
  fcp7xml: 'timeline.format.fcp7xml',
  fcpxml: 'timeline.format.fcpxml',
  edl: 'timeline.format.edl',
  otio: 'timeline.format.otio',
};

function fmtDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${m}:${String(s).padStart(2, '0')}`;
}

function safeFilename(name: string): string {
  return name.replace(/[\\/:*?"<>|]+/g, '_').trim() || 'amv-timeline';
}

function missingSources(message: string): string[] | null {
  const start = message.indexOf('{');
  if (start < 0) return null;
  try {
    const payload = JSON.parse(message.slice(start));
    const detail = payload?.detail;
    if (detail && detail.error === 'source_missing' && Array.isArray(detail.paths)) {
      return detail.paths.map(String);
    }
  } catch {}
  return null;
}

export default function TimelineExportDialog({
  selection,
  defaultName,
  scopeLabel,
  onClose,
  onExported,
}: TimelineExportDialogProps) {
  const t = useT();
  const [formats, setFormats] = useState<TimelineFormat[]>([]);
  const [format, setFormat] = useState<TimelineFormat['id']>('fcp7xml');
  const [title, setTitle] = useState(defaultName ?? 'AMV timeline');
  const [sequenceFps, setSequenceFps] = useState('first');
  const [resolution, setResolution] = useState('first');
  const [layout, setLayout] = useState<'sequential' | 'gaps'>('sequential');
  const [gapFrames, setGapFrames] = useState(0);
  const [handlesFrames, setHandlesFrames] = useState(0);
  const [dropFrame, setDropFrame] = useState(false);
  const [markers, setMarkers] = useState(true);
  const [includeAudio, setIncludeAudio] = useState(true);
  const [collectMedia, setCollectMedia] = useState(false);
  const [relativePaths, setRelativePaths] = useState(false);
  const [preview, setPreview] = useState<TimelinePreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [result, setResult] = useState<TimelineExportResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selectionKey = useMemo(() => JSON.stringify(selection), [selection]);
  const options = useMemo<TimelineOptions>(() => ({
    sequence_fps: sequenceFps,
    resolution,
    layout,
    gap_frames: layout === 'gaps' ? gapFrames : 0,
    handles_frames: handlesFrames,
    drop_frame: dropFrame,
    markers,
    include_audio: includeAudio,
    collect_media: collectMedia,
    path_mode: relativePaths || collectMedia ? 'relative' : 'absolute',
    title: title.trim() || 'AMV timeline',
  }), [sequenceFps, resolution, layout, gapFrames, handlesFrames, dropFrame,
       markers, includeAudio, collectMedia, relativePaths, title]);

  useEffect(() => {
    api.timelineFormats().then((r) => {
      setFormats(r.formats);
      if (r.formats.length > 0 && !r.formats.some((f) => f.id === format)) {
        setFormat(r.formats[0].id);
      }
    }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let alive = true;
    setPreviewing(true);
    const timer = window.setTimeout(() => {
      api.timelinePreview({ ...selection, options })
        .then((r) => {
          if (!alive) return;
          setPreview(r);
          setPreviewError(null);
        })
        .catch((e) => {
          if (!alive) return;
          const message = (e as Error).message;
          const missing = missingSources(message);
          setPreview(null);
          setPreviewError(missing
            ? t('timeline.dialog.missingSources', { paths: missing.join(', ') })
            : message);
        })
        .finally(() => { if (alive) setPreviewing(false); });
    }, 180);
    return () => { alive = false; window.clearTimeout(timer); };
    // selectionKey keeps the effect stable when the parent builds the
    // selection inline on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectionKey, options]);

  const selectedFormat = formats.find((f) => f.id === format);
  const extension = selectedFormat?.extension ?? '.xml';

  async function doExport() {
    if (exporting) return;
    setError(null);
    const stem = safeFilename(title || defaultName || 'amv-timeline');
    const target = await window.amvBridge?.saveFileDialog({
      defaultPath: `${stem}${extension}`,
      filters: [{ name: selectedFormat?.label ?? 'Timeline', extensions: [extension.replace('.', '')] }],
    });
    if (!target) return;
    setExporting(true);
    try {
      const r = await api.timelineExport({
        ...selection,
        format,
        output_path: target,
        options,
      });
      setResult(r);
      onExported?.(r);
    } catch (e) {
      const message = (e as Error).message;
      const missing = missingSources(message);
      setError(missing
        ? t('timeline.dialog.missingSources', { paths: missing.join(', ') })
        : message);
    } finally {
      setExporting(false);
    }
  }

  const effectiveRate = sequenceFps === 'first' ? preview?.sequence_fps : sequenceFps;
  const dropFrameAvailable = effectiveRate === '30000/1001' || effectiveRate === '60000/1001';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
         onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="w-[720px] max-w-full max-h-[92vh] overflow-y-auto bg-[#18181B] border-2 border-[#3F3F46] rounded-2xl shadow-2xl">
        <div className="flex items-center gap-3 px-5 py-4 border-b border-[#27272A]">
          <div className="p-2 rounded-lg bg-gradient-to-br from-[#8B5CF6] to-[#EC4899]">
            <Clapperboard size={18} className="text-white" />
          </div>
          <div className="flex-1">
            <div className="text-[#FAFAFA] font-bold">{t('timeline.dialog.title')}</div>
            <div className="text-xs text-[#71717A]">{t('timeline.dialog.subtitle')}</div>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg text-[#71717A] hover:text-[#FAFAFA] hover:bg-[#27272A]">
            <X size={16} />
          </button>
        </div>

        <div className="p-5 flex flex-col gap-4">
          {scopeLabel && (
            <div className="text-xs text-[#A1A1AA] bg-[#0F0F11] border border-[#27272A] rounded-lg px-3 py-2">
              {scopeLabel}
            </div>
          )}

          {/* format picker */}
          <label className="flex flex-col gap-1.5">
            <span className="text-xs uppercase tracking-wider text-[#71717A]">{t('timeline.dialog.format')}</span>
            <select
              value={format}
              onChange={(e) => setFormat(e.target.value as TimelineFormat['id'])}
              className="bg-[#0F0F11] border border-[#8B5CF6]/40 rounded-lg px-3 py-2 text-sm text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]"
            >
              {formats.map((f) => (
                <option key={f.id} value={f.id}>{t(FORMAT_I18N[f.id])}</option>
              ))}
            </select>
            <span className="text-[11px] text-[#71717A]">{t('timeline.dialog.formatHint')}</span>
          </label>

          {/* preview */}
          <div className="flex items-center gap-3 bg-[#0F0F11] border border-[#27272A] rounded-lg px-4 py-3">
            {previewing ? (
              <Loader2 size={16} className="animate-spin text-[#8B5CF6]" />
            ) : previewError ? (
              <AlertTriangle size={16} className="text-amber-400" />
            ) : (
              <Check size={16} className="text-emerald-400" />
            )}
            <div className="flex-1 min-w-0">
              {previewError ? (
                <div className="text-xs text-amber-300 break-words">{previewError}</div>
              ) : preview ? (
                <div className="flex items-center gap-3 text-sm text-[#FAFAFA]">
                  <span className="font-semibold">{t('timeline.dialog.clips', { count: preview.clip_count })}</span>
                  <span className="text-[#3F3F46]">•</span>
                  <span>{t('timeline.dialog.duration', { duration: fmtDuration(preview.duration_seconds) })}</span>
                  <span className="text-[#3F3F46]">•</span>
                  <span className="font-mono text-xs text-[#A1A1AA]">{preview.sequence_fps} fps · {preview.width}×{preview.height}</span>
                </div>
              ) : (
                <div className="text-xs text-[#71717A]">—</div>
              )}
            </div>
          </div>

          {preview && preview.warnings.length > 0 && (
            <div className="bg-amber-500/10 border border-amber-500/40 rounded-lg px-3 py-2">
              <div className="text-[11px] uppercase tracking-wider text-amber-300 mb-1">{t('timeline.dialog.warnings')}</div>
              <ul className="text-xs text-amber-200 list-disc pl-4 space-y-0.5">
                {preview.warnings.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            </div>
          )}

          {/* options */}
          <div className="grid grid-cols-2 gap-3">
            <label className="flex flex-col gap-1">
              <span className="text-[11px] uppercase tracking-wider text-[#71717A]">{t('timeline.dialog.titleField')}</span>
              <input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="bg-[#0F0F11] border border-[#27272A] rounded-lg px-3 py-2 text-sm text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[11px] uppercase tracking-wider text-[#71717A]">{t('timeline.dialog.sequenceFps')}</span>
              <select
                value={sequenceFps}
                onChange={(e) => setSequenceFps(e.target.value)}
                className="bg-[#0F0F11] border border-[#27272A] rounded-lg px-3 py-2 text-sm text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]"
              >
                {FPS_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.labelKey ? t(o.labelKey) : o.label}</option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[11px] uppercase tracking-wider text-[#71717A]">{t('timeline.dialog.resolution')}</span>
              <select
                value={resolution}
                onChange={(e) => setResolution(e.target.value)}
                className="bg-[#0F0F11] border border-[#27272A] rounded-lg px-3 py-2 text-sm text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]"
              >
                {RESOLUTION_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.labelKey ? t(o.labelKey) : o.label}</option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[11px] uppercase tracking-wider text-[#71717A]">{t('timeline.dialog.layout')}</span>
              <select
                value={layout}
                onChange={(e) => setLayout(e.target.value as 'sequential' | 'gaps')}
                className="bg-[#0F0F11] border border-[#27272A] rounded-lg px-3 py-2 text-sm text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]"
              >
                <option value="sequential">{t('timeline.dialog.layoutSequential')}</option>
                <option value="gaps">{t('timeline.dialog.layoutGaps')}</option>
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[11px] uppercase tracking-wider text-[#71717A]">{t('timeline.dialog.gapFrames')}</span>
              <input
                type="number" min={0}
                value={gapFrames}
                disabled={layout !== 'gaps'}
                onChange={(e) => setGapFrames(Math.max(0, Number(e.target.value) || 0))}
                className="bg-[#0F0F11] border border-[#27272A] rounded-lg px-3 py-2 text-sm text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6] disabled:opacity-40"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[11px] uppercase tracking-wider text-[#71717A]">{t('timeline.dialog.handles')}</span>
              <input
                type="number" min={0}
                value={handlesFrames}
                onChange={(e) => setHandlesFrames(Math.max(0, Number(e.target.value) || 0))}
                className="bg-[#0F0F11] border border-[#27272A] rounded-lg px-3 py-2 text-sm text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]"
              />
            </label>
          </div>

          <div className="grid grid-cols-2 gap-x-6 gap-y-2">
            <Toggle checked={markers} onChange={setMarkers} label={t('timeline.dialog.markers')} />
            <Toggle checked={includeAudio} onChange={setIncludeAudio} label={t('timeline.dialog.audio')} />
            <Toggle checked={dropFrame} onChange={setDropFrame} label={t('timeline.dialog.dropFrame')} disabled={!dropFrameAvailable} />
            <Toggle checked={relativePaths} onChange={setRelativePaths} label={t('timeline.dialog.relativePaths')} />
            <Toggle checked={collectMedia} onChange={setCollectMedia} label={t('timeline.dialog.collect')} />
          </div>
          {collectMedia && (
            <div className="text-[11px] text-[#71717A] -mt-2">{t('timeline.dialog.collectHint')}</div>
          )}
          {dropFrame && dropFrameAvailable && (
            <div className="text-[11px] text-[#71717A] -mt-2">{t('timeline.dialog.dropFrameHint')}</div>
          )}

          {error && (
            <div className="bg-red-500/10 border border-red-500/40 text-red-300 text-xs rounded-lg px-3 py-2 break-words">
              {t('timeline.dialog.failed', { message: error })}
            </div>
          )}

          {result && (
            <div className="bg-emerald-500/10 border border-emerald-500/40 text-emerald-300 text-xs rounded-lg px-3 py-2">
              <div className="break-all">{t('timeline.dialog.saved', { path: result.output })}</div>
              <button
                onClick={() => api.revealOutput(result.output)}
                className="mt-1 inline-flex items-center gap-1 underline hover:text-emerald-200"
              >
                {t('timeline.dialog.reveal')} <ChevronRight size={12} />
              </button>
            </div>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 px-5 py-4 border-t border-[#27272A] bg-[#0F0F11]/40">
          <button onClick={onClose} className="px-4 py-2 rounded-lg text-sm text-[#A1A1AA] hover:text-[#FAFAFA] hover:bg-[#27272A]">
            {/* After a successful export there is nothing left to cancel. */}
            {result ? t('common.close') : t('timeline.dialog.cancel')}
          </button>
          <button
            onClick={doExport}
            disabled={exporting || !!previewError || !preview || preview.clip_count === 0}
            className="flex items-center gap-2 px-5 py-2 rounded-xl text-sm font-bold bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {exporting ? <Loader2 size={14} className="animate-spin" /> : <Clapperboard size={14} />}
            {exporting ? t('timeline.dialog.exporting') : t('timeline.dialog.export')}
          </button>
        </div>
      </div>
    </div>
  );
}

function Toggle({ checked, onChange, label, disabled }: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <label className={`flex items-center gap-2 text-xs text-[#A1A1AA] ${disabled ? 'opacity-40' : 'cursor-pointer'}`}>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        className="accent-[#8B5CF6]"
      />
      <span>{label}</span>
    </label>
  );
}
