// Details of the selected shot: what it is, where it sits on the music, its
// alternatives (click = swap), lock and "regenerate this passage".
import { ChevronLeft, ChevronRight, ExternalLink, Loader2, Lock, Play, RefreshCw, Unlock, X } from 'lucide-react';
import { api } from '../api/client';
import type { ClipAlternative, MusicSection, ProposedClip } from '../api/music';
import { useT } from '../i18n';
import { formatTime, hexToRgba, sectionColor, sectionLabelKey } from './musicUtils';

interface Props {
  clip: ProposedClip;
  section: MusicSection | null;
  total: number;
  busy: boolean;
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
  onPickAlternative: (alt: ClipAlternative) => void;
  onToggleLock: () => void;
  onRegenerateSection: () => void;
  onPlayHere: () => void;
  onOpenEditor?: () => void;
}

function pct(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? '—' : `${Math.round(v * 100)}%`;
}

function IntensityBar({ value, target }: { value: number | null; target?: number }) {
  const v = value == null ? 0 : Math.min(1, Math.max(0, value));
  const hue = Math.round(50 - 50 * v);
  return (
    <div className="relative h-1.5 w-full bg-[#27272A] rounded-full overflow-visible">
      <div className="h-full rounded-full" style={{ width: `${v * 100}%`, background: value == null ? 'transparent' : `hsl(${hue} 95% 55%)` }} />
      {target != null && (
        <div className="absolute -top-1 -bottom-1 w-0.5 bg-[#FAFAFA]" style={{ left: `${Math.min(1, Math.max(0, target)) * 100}%` }} />
      )}
    </div>
  );
}

export default function MusicClipPanel({
  clip, section, total, busy, onClose, onPrev, onNext, onPickAlternative,
  onToggleLock, onRegenerateSection, onPlayHere, onOpenEditor,
}: Props) {
  const t = useT();
  const col = sectionColor(section?.type);
  const alts = clip.alternatives ?? [];

  return (
    <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#3F3F46] rounded-xl p-4 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <button type="button" onClick={onPrev} disabled={clip.slot <= 0} title={t('music.clip.prev')}
                className="p-1 rounded-md text-[#A1A1AA] hover:text-[#FAFAFA] hover:bg-[#27272A] disabled:opacity-30">
          <ChevronLeft size={16} />
        </button>
        <div className="text-[#FAFAFA] font-bold text-sm">{t('music.clip.title', { slot: clip.slot + 1 })}<span className="text-[#71717A] font-normal"> / {total}</span></div>
        <button type="button" onClick={onNext} disabled={clip.slot >= total - 1} title={t('music.clip.next')}
                className="p-1 rounded-md text-[#A1A1AA] hover:text-[#FAFAFA] hover:bg-[#27272A] disabled:opacity-30">
          <ChevronRight size={16} />
        </button>
        {section && (
          <span className="text-[10px] font-bold uppercase tracking-wide px-2 py-0.5 rounded" style={{ background: hexToRgba(col, 0.3), color: '#FAFAFA' }}>
            {t(sectionLabelKey(section.type))}
          </span>
        )}
        {clip.locked && (
          <span className="inline-flex items-center gap-1 text-[11px] text-[#FBBF24]"><Lock size={11} /> {t('music.clip.locked')}</span>
        )}
        <span className="flex-1" />
        <button type="button" onClick={onClose} title={t('music.clip.close')} className="p-1.5 rounded-lg text-[#71717A] hover:text-[#FAFAFA] hover:bg-[#27272A]">
          <X size={15} />
        </button>
      </div>

      <div className="flex gap-4 flex-wrap">
        {/* the shot */}
        <div className="w-[260px] shrink-0 flex flex-col gap-2">
          <div className="aspect-video w-full rounded-lg overflow-hidden bg-[#27272A]" style={{ boxShadow: `inset 0 -3px 0 ${col}` }}>
            <img src={api.thumbnailUrl(clip.scene_id)} alt="" className="w-full h-full object-cover"
                 onError={(e) => { (e.currentTarget as HTMLImageElement).style.visibility = 'hidden'; }} />
          </div>
          <div className="text-sm text-[#FAFAFA] font-semibold truncate" title={clip.video_name}>{clip.video_name || '—'}</div>
          {clip.group && <div className="text-xs text-[#A1A1AA] -mt-1.5 truncate">{clip.group}</div>}
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
            <dt className="text-[#71717A]">{t('music.clip.source')}</dt>
            <dd className="text-[#FAFAFA] font-mono">{formatTime(clip.src_in_ms, { tenths: true })} → {formatTime(clip.src_out_ms, { tenths: true })}</dd>
            <dt className="text-[#71717A]">{t('music.clip.placement')}</dt>
            <dd className="text-[#FAFAFA] font-mono">
              {formatTime(clip.start_ms, { tenths: true })} → {formatTime(clip.end_ms, { tenths: true })}
              <span className="text-[#71717A] font-sans"> · {t('music.clip.beats', { count: clip.beats })}</span>
            </dd>
            <dt className="text-[#71717A]">{t('music.clip.intensity')}</dt>
            <dd className="text-[#FAFAFA] flex flex-col gap-1">
              <span>
                {clip.intensity == null ? t('music.clip.unknown') : pct(clip.intensity)}
                <span className="text-[#71717A]"> · {t('music.clip.wanted', { value: pct(clip.target) })}</span>
              </span>
              <IntensityBar value={clip.intensity} target={clip.target} />
            </dd>
          </dl>
        </div>

        {/* alternatives */}
        <div className="flex-1 min-w-[260px] flex flex-col gap-2">
          <div className="flex items-baseline gap-2">
            <span className="text-[11px] uppercase tracking-wider text-[#71717A]">{t('music.clip.alternatives')}</span>
            {alts.length > 0 && <span className="text-[11px] text-[#52525B]">{t('music.clip.alternativesHint')}</span>}
          </div>
          {alts.length === 0 ? (
            <div className="text-xs text-[#71717A]">{t('music.clip.alternativesNone')}</div>
          ) : (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(120px,1fr))] gap-2 max-h-[220px] overflow-y-auto pr-1">
              {alts.map((a, i) => (
                <button
                  key={`${a.scene_id}-${a.src_in_ms}-${i}`}
                  type="button"
                  onClick={() => onPickAlternative(a)}
                  title={`${a.video_name} · ${formatTime(a.src_in_ms, { tenths: true })}`}
                  className="group text-left rounded-lg overflow-hidden bg-[#0F0F11] border border-[#27272A] hover:border-[#8B5CF6] transition-colors"
                >
                  <div className="aspect-video bg-[#27272A] overflow-hidden">
                    <img src={api.thumbnailUrl(a.scene_id)} alt="" loading="lazy" className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                         onError={(e) => { (e.currentTarget as HTMLImageElement).style.visibility = 'hidden'; }} />
                  </div>
                  <div className="px-1.5 py-1 flex flex-col gap-1">
                    <span className="text-[10px] text-[#A1A1AA] truncate">{a.video_name}</span>
                    <IntensityBar value={a.intensity} target={clip.target} />
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* actions */}
        <div className="w-[200px] shrink-0 flex flex-col gap-2">
          <button type="button" onClick={onToggleLock}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold border transition-colors ${
                    clip.locked ? 'bg-[#FBBF24]/15 border-[#FBBF24]/60 text-[#FBBF24]' : 'bg-[#0F0F11] border-[#27272A] text-[#FAFAFA] hover:border-[#3F3F46]'
                  }`}>
            {clip.locked ? <Unlock size={13} /> : <Lock size={13} />}
            {clip.locked ? t('music.clip.unlock') : t('music.clip.lock')}
          </button>
          {clip.locked && <div className="text-[11px] text-[#71717A] -mt-1">{t('music.clip.lockedHint')}</div>}
          <button type="button" onClick={onRegenerateSection} disabled={busy}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold bg-[#0F0F11] border border-[#27272A] text-[#FAFAFA] hover:border-[#8B5CF6] disabled:opacity-40">
            {busy ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
            {t('music.clip.regenerateSection')}
          </button>
          <button type="button" onClick={onPlayHere}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold bg-[#0F0F11] border border-[#27272A] text-[#FAFAFA] hover:border-[#3F3F46]">
            <Play size={13} /> {t('music.clip.playHere')}
          </button>
          {onOpenEditor && (
            <button type="button" onClick={onOpenEditor}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold bg-[#0F0F11] border border-[#27272A] text-[#FAFAFA] hover:border-[#3F3F46]">
              <ExternalLink size={13} /> {t('music.clip.openEditor')}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
