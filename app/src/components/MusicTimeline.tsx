// Shared time axis: passages row, waveform canvas and edit strip scroll and
// zoom together; one playhead runs across all three.
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Tag } from 'lucide-react';
import type { MusicAnalysis, ProposedClip } from '../api/music';
import { useT } from '../i18n';
import MusicWaveform from './MusicWaveform';
import MusicEditStrip from './MusicEditStrip';
import { formatTime, hexToRgba, pxPerMs, sectionColor, sectionLabelKey } from './musicUtils';

interface Props {
  analysis: MusicAnalysis;
  clips: ProposedClip[] | null;
  zoom: number;
  currentMs: number;
  playing: boolean;
  /** Live media time, read every animation frame while playing. */
  getTimeMs: () => number;
  onSeek: (ms: number) => void;
  selectedSlot: number | null;
  onSelectClip: (slot: number) => void;
  selectedSection: number | null;
  onSelectSection: (index: number) => void;
  sectionQueries: Record<string, string>;
  emptyStrip?: React.ReactNode;
}

const SECTION_ROW_H = 22;
const WAVE_H = 132;
const STRIP_H = 56;

export default function MusicTimeline({
  analysis, clips, zoom, currentMs, playing, getTimeMs, onSeek,
  selectedSlot, onSelectClip, selectedSection, onSelectSection, sectionQueries, emptyStrip,
}: Props) {
  const t = useT();
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [scrollEl, setScrollEl] = useState<HTMLDivElement | null>(null);
  const [viewW, setViewW] = useState(0);
  const playheadRef = useRef<HTMLDivElement | null>(null);
  const duration = Math.max(1, analysis.duration_ms || 0);
  const ppm = pxPerMs(viewW, zoom, duration);
  const contentW = Math.max(viewW, Math.round(duration * ppm));
  const prevPpm = useRef(ppm);

  // measure the viewport
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    setScrollEl(el);
    const measure = () => setViewW(el.clientWidth);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // keep the centre of the view in place when zooming / resizing
  useLayoutEffect(() => {
    const el = scrollRef.current;
    const old = prevPpm.current;
    prevPpm.current = ppm;
    if (!el || !(old > 0) || !(ppm > 0) || old === ppm) return;
    const centerMs = (el.scrollLeft + el.clientWidth / 2) / old;
    el.scrollLeft = Math.max(0, centerMs * ppm - el.clientWidth / 2);
  }, [ppm]);

  // playhead: rAF while playing, otherwise follow currentMs
  useEffect(() => {
    const place = (ms: number, follow: boolean) => {
      const ph = playheadRef.current;
      if (!ph || !(ppm > 0)) return;
      const x = ms * ppm;
      ph.style.transform = `translateX(${x}px)`;
      const el = scrollRef.current;
      if (follow && el && (x < el.scrollLeft || x > el.scrollLeft + el.clientWidth - 24)) {
        el.scrollLeft = Math.max(0, x - el.clientWidth * 0.1);
      }
    };
    if (!playing) { place(currentMs, false); return; }
    let raf = 0;
    const tick = () => { place(getTimeMs(), true); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, currentMs, ppm, getTimeMs]);

  const sectionTypes = useMemo(() => {
    const m: Record<number, string> = {};
    for (const s of analysis.sections ?? []) m[s.index] = s.type;
    return m;
  }, [analysis.sections]);

  const playingSlot = useMemo(() => {
    if (!clips) return null;
    const c = clips.find((x) => currentMs >= x.start_ms && currentMs < x.end_ms);
    return c ? c.slot : null;
  }, [clips, currentMs]);

  return (
    <div ref={scrollRef} className="relative overflow-x-auto overflow-y-hidden rounded-lg border border-[#27272A] bg-[#0F0F11] select-none">
      <div className="relative" style={{ width: contentW || '100%' }}>
        {/* passages row */}
        <div className="relative bg-[#18181B]" style={{ height: SECTION_ROW_H }}>
          {(analysis.sections ?? []).map((s) => {
            const left = s.start_ms * ppm;
            const width = Math.max(1, (s.end_ms - s.start_ms) * ppm - 1);
            const col = sectionColor(s.type);
            const sel = selectedSection === s.index;
            const hasQuery = !!sectionQueries[String(s.index)]?.trim();
            const label = t(sectionLabelKey(s.type));
            return (
              <button
                key={s.index}
                type="button"
                onClick={() => onSelectSection(s.index)}
                title={`${label} · ${formatTime(s.start_ms)} – ${formatTime(s.end_ms)}${hasQuery ? ` · ${sectionQueries[String(s.index)]}` : ''}`}
                className="absolute top-0.5 bottom-0.5 rounded-[3px] px-1.5 text-[10px] font-bold uppercase tracking-wide text-left overflow-hidden whitespace-nowrap flex items-center gap-1"
                style={{
                  left, width,
                  background: hexToRgba(col, sel ? 0.75 : 0.35),
                  color: sel ? '#FFFFFF' : '#E4E4E7',
                  boxShadow: sel ? `0 0 0 1px ${col}` : undefined,
                }}
              >
                {hasQuery && width > 24 && <Tag size={9} className="shrink-0" />}
                {width > 34 && <span className="truncate">{label}</span>}
              </button>
            );
          })}
        </div>

        {/* waveform */}
        <div className="relative" style={{ height: WAVE_H }}>
          {viewW > 0 && (
            <MusicWaveform
              analysis={analysis}
              ppm={ppm}
              viewW={viewW}
              height={WAVE_H}
              scrollEl={scrollEl}
              selectedSection={selectedSection}
              dropLabel={t('music.wave.drop')}
              title={t('music.wave.clickToSeek')}
              onSeek={onSeek}
            />
          )}
        </div>

        {/* edit strip */}
        <div className="relative border-t border-[#27272A] bg-[#18181B]/60" style={{ height: STRIP_H + 8 }}>
          {clips && clips.length > 0 ? (
            <div className="pt-1">
              <MusicEditStrip
                clips={clips}
                sectionTypes={sectionTypes}
                ppm={ppm}
                height={STRIP_H}
                selectedSlot={selectedSlot}
                playingSlot={playingSlot}
                onSelect={onSelectClip}
              />
            </div>
          ) : (
            <div className="sticky left-0 h-full flex items-center px-4 text-xs text-[#71717A]" style={{ width: viewW || '100%' }}>
              {emptyStrip}
            </div>
          )}
        </div>

        {/* playhead */}
        <div
          ref={playheadRef}
          className="absolute top-0 bottom-0 left-0 w-0.5 bg-[#FAFAFA] pointer-events-none z-20 shadow-[0_0_6px_rgba(255,255,255,0.6)]"
          style={{ transform: `translateX(${currentMs * ppm}px)` }}
        />
      </div>
    </div>
  );
}
