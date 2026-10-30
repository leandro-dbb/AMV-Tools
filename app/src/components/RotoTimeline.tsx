import { ChevronFirst, ChevronLast, ChevronLeft, ChevronRight, Pause, Play, SkipBack, SkipForward } from 'lucide-react';
import { useRef } from 'react';
import type { RotoSuspect } from '../api/roto';
import { useT } from '../i18n';
import { REASON_KEYS, timelineFrameAt } from './rotoUtils';

interface Props {
  nFrames: number;
  frame: number;
  keyframes: number[];
  suspects: RotoSuspect[];
  pendingCount: number;
  playing: boolean;
  onSeek: (frame: number) => void;
  onTogglePlay: () => void;
  onNextSuspect: () => void;
  onPrevSuspect: () => void;
}

/** Bar the width of the shot, one cell per frame: current frame cursor, the
 * frames the user fixed (yellow diamonds) and the frames to check (red, more
 * opaque = more likely wrong). Click or drag to scrub. */
export default function RotoTimeline({
  nFrames, frame, keyframes, suspects, pendingCount, playing, onSeek, onTogglePlay, onNextSuspect, onPrevSuspect,
}: Props) {
  const t = useT();
  const barRef = useRef<HTMLDivElement | null>(null);
  const n = Math.max(1, nFrames);
  const cellPct = 100 / n;

  function seekAt(clientX: number) {
    const el = barRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    onSeek(timelineFrameAt(clientX - r.left, r.width, n));
  }

  function onPointerDown(e: React.PointerEvent) {
    if (e.button !== 0) return;
    e.preventDefault();
    seekAt(e.clientX);
    const move = (ev: PointerEvent) => seekAt(ev.clientX);
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  }

  const btn = 'p-1.5 rounded-lg hover:bg-[#27272A] text-[#A1A1AA] hover:text-[#FAFAFA] disabled:opacity-30';
  const suspectsLabel = pendingCount === 0
    ? t('roto.suspectsNone')
    : pendingCount === 1 ? t('roto.suspectsLeftOne') : t('roto.suspectsLeft', { count: pendingCount });

  return (
    <div className="bg-[#0F0F11] border-2 border-[#27272A] rounded-xl px-3 py-2 space-y-2">
      <div className="flex items-center gap-1 flex-wrap">
        <button className={btn} onClick={() => onSeek(0)} title={t('roto.firstFrame')} aria-label={t('roto.firstFrame')}><ChevronFirst size={16} /></button>
        <button className={btn} onClick={() => onSeek(Math.max(0, frame - 1))} title={t('roto.prevFrame')} aria-label={t('roto.prevFrame')}><ChevronLeft size={16} /></button>
        <button
          className="p-1.5 rounded-lg bg-[#27272A] hover:bg-[#3F3F46] text-[#FAFAFA]"
          onClick={onTogglePlay}
          title={playing ? t('roto.pause') : t('roto.play')}
          aria-label={playing ? t('roto.pause') : t('roto.play')}
          data-testid="roto-play"
        >
          {playing ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <button className={btn} onClick={() => onSeek(Math.min(n - 1, frame + 1))} title={t('roto.nextFrame')} aria-label={t('roto.nextFrame')}><ChevronRight size={16} /></button>
        <button className={btn} onClick={() => onSeek(n - 1)} title={t('roto.lastFrame')} aria-label={t('roto.lastFrame')}><ChevronLast size={16} /></button>
        <span className="ml-2 text-[#FAFAFA] text-xs font-mono whitespace-nowrap" data-testid="roto-frame-label">
          {t('roto.frameLabel', { current: frame + 1, total: nFrames })}
        </span>
        <div className="flex-1" />
        <span className={`text-xs whitespace-nowrap ${pendingCount ? 'text-red-300' : 'text-emerald-300'}`} data-testid="roto-suspects-left">
          {suspectsLabel}
        </span>
        <button className={btn} onClick={onPrevSuspect} disabled={!suspects.length} title={t('roto.prevSuspect')} aria-label={t('roto.prevSuspect')} data-testid="roto-prev-suspect">
          <SkipBack size={15} />
        </button>
        <button
          className="px-2.5 py-1 rounded-lg bg-red-500/15 hover:bg-red-500/30 border border-red-500/40 text-red-200 text-xs flex items-center gap-1 disabled:opacity-30"
          onClick={onNextSuspect} disabled={!suspects.length} title={t('roto.nextSuspect')} data-testid="roto-next-suspect"
        >
          <SkipForward size={14} /> N
        </button>
      </div>

      <div
        ref={barRef}
        data-testid="roto-timeline"
        className="relative h-9 rounded-md bg-[#18181B] border border-[#27272A] cursor-pointer overflow-hidden touch-none"
        onPointerDown={onPointerDown}
        style={nFrames <= 400 ? {
          backgroundImage: `repeating-linear-gradient(to right, transparent 0, transparent calc(${cellPct}% - 1px), rgba(63,63,70,0.6) calc(${cellPct}% - 1px), rgba(63,63,70,0.6) ${cellPct}%)`,
        } : undefined}
      >
        {suspects.map((s) => (
          <div
            key={`s${s.frame}`}
            className="absolute top-0 bottom-0 bg-red-500"
            style={{
              left: `${s.frame * cellPct}%`,
              width: `max(3px, ${cellPct}%)`,
              opacity: 0.25 + 0.75 * Math.max(0, Math.min(1, s.score)),
            }}
            title={t('roto.suspectTitle', { frame: s.frame + 1, reasons: s.reasons.map((r) => t(REASON_KEYS[r] ?? r)).join(', ') })}
            data-testid="roto-suspect"
            data-frame={s.frame}
          />
        ))}
        {keyframes.map((k) => (
          <div
            key={`k${k}`}
            className="absolute bottom-1 w-2.5 h-2.5 -ml-[5px] rotate-45 bg-yellow-400 border border-black/60"
            style={{ left: `${(k + 0.5) * cellPct}%` }}
            title={t('roto.keyframeTitle', { frame: k + 1 })}
            data-testid="roto-keyframe"
            data-frame={k}
          />
        ))}
        <div
          className="absolute top-0 bottom-0 bg-white/25 border-x-2 border-white pointer-events-none"
          style={{ left: `${frame * cellPct}%`, width: `max(2px, ${cellPct}%)` }}
        />
      </div>
      <div className="flex items-center gap-4 text-[10px] text-[#71717A]">
        <span className="flex items-center gap-1.5"><span className="inline-block w-2 h-2 rotate-45 bg-yellow-400" />{t('roto.legend.keyframe')}</span>
        <span className="flex items-center gap-1.5"><span className="inline-block w-2.5 h-2.5 bg-red-500/80 rounded-sm" />{t('roto.legend.suspect')}</span>
      </div>
    </div>
  );
}
