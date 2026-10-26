// Canvas waveform of the analysed music: section bands, peaks, energy curve,
// beat ticks, drop flags and a time ruler. Only the visible window is drawn
// (the canvas is viewport-sized and follows the horizontal scroll), so it
// stays cheap at 8x zoom with thousands of beats.
import { useCallback, useEffect, useRef } from 'react';
import type { MusicAnalysis } from '../api/music';
import { formatTime, hexToRgba, sectionColor, visibleRange, waveformPeak } from './musicUtils';

interface Props {
  analysis: MusicAnalysis;
  /** Pixels per ms (shared with the edit strip). */
  ppm: number;
  /** Viewport width in CSS px. */
  viewW: number;
  height: number;
  scrollEl: HTMLDivElement | null;
  selectedSection: number | null;
  dropLabel: string;
  title?: string;
  onSeek: (ms: number) => void;
}

const RULER_H = 14;
const TICK_STEPS_S = [1, 2, 5, 10, 15, 30, 60, 120, 300];

export default function MusicWaveform({ analysis, ppm, viewW, height, scrollEl, selectedSection, dropLabel, title, onSeek }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);

  const draw = useCallback(() => {
    rafRef.current = null;
    const canvas = canvasRef.current;
    if (!canvas || !(viewW > 0) || !(ppm > 0)) return;
    const scrollLeft = scrollEl?.scrollLeft ?? 0;
    canvas.style.left = `${scrollLeft}px`;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(viewW);
    const H = height;
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      canvas.style.width = `${W}px`;
      canvas.style.height = `${H}px`;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const duration = Math.max(1, analysis.duration_ms || 0);
    const t0 = scrollLeft / ppm;
    const t1 = (scrollLeft + W) / ppm;
    const xOf = (ms: number) => ms * ppm - scrollLeft;
    const top = RULER_H;
    const waveH = H - RULER_H;
    const mid = top + waveH / 2;

    // background
    ctx.fillStyle = '#0F0F11';
    ctx.fillRect(0, 0, W, H);

    // section bands
    for (const s of analysis.sections ?? []) {
      if (s.end_ms < t0 || s.start_ms > t1) continue;
      const x0 = Math.max(0, xOf(s.start_ms));
      const x1 = Math.min(W, xOf(s.end_ms));
      const col = sectionColor(s.type);
      const sel = selectedSection === s.index;
      ctx.fillStyle = hexToRgba(col, sel ? 0.32 : 0.16);
      ctx.fillRect(x0, top, x1 - x0, waveH);
      ctx.fillStyle = hexToRgba(col, 0.9);
      ctx.fillRect(x0, top, x1 - x0, 2);
      if (sel) {
        ctx.strokeStyle = hexToRgba(col, 0.9);
        ctx.lineWidth = 1;
        ctx.strokeRect(x0 + 0.5, top + 0.5, Math.max(0, x1 - x0 - 1), waveH - 1);
      }
    }

    // ruler
    ctx.fillStyle = '#18181B';
    ctx.fillRect(0, 0, W, RULER_H);
    const pxPerSec = ppm * 1000;
    const step = TICK_STEPS_S.find((s) => s * pxPerSec >= 64) ?? 600;
    ctx.fillStyle = '#71717A';
    ctx.strokeStyle = '#3F3F46';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 1;
    const firstTick = Math.floor(t0 / 1000 / step) * step;
    for (let s = firstTick; s * 1000 <= t1; s += step) {
      const x = Math.round(xOf(s * 1000)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, RULER_H - 5);
      ctx.lineTo(x, RULER_H);
      ctx.stroke();
      ctx.fillText(formatTime(s * 1000), x + 3, RULER_H / 2);
    }

    // waveform peaks (mirrored)
    const wf = analysis.waveform ?? [];
    if (wf.length > 0) {
      ctx.fillStyle = 'rgba(228,228,231,0.55)';
      const amp = waveH / 2 - 4;
      for (let x = 0; x < W; x++) {
        const a = (x + scrollLeft) / ppm;
        if (a > duration) break;
        const p = waveformPeak(wf, duration, a, (x + 1 + scrollLeft) / ppm);
        const h = Math.max(0.5, p * amp);
        ctx.fillRect(x, mid - h, 1, h * 2);
      }
    }

    // beats: thin ticks at the bottom, bar starts a bit stronger
    const beats = analysis.beats_ms ?? [];
    const downs = analysis.downbeats_ms ?? [];
    const [b0, b1] = visibleRange(beats, t0, t1);
    const beatSpacing = beats.length > 1 ? ((beats[beats.length - 1] - beats[0]) / (beats.length - 1)) * ppm : 99;
    if (beatSpacing >= 3) {
      ctx.fillStyle = 'rgba(250,250,250,0.35)';
      for (let i = b0; i < b1; i++) {
        ctx.fillRect(Math.round(xOf(beats[i])), H - 6, 1, 6);
      }
    }
    const [d0, d1] = visibleRange(downs, t0, t1);
    const downSpacing = beatSpacing * Math.max(1, analysis.meter || 4);
    for (let i = d0; i < d1; i++) {
      const x = Math.round(xOf(downs[i]));
      ctx.fillStyle = 'rgba(250,250,250,0.75)';
      ctx.fillRect(x, H - 11, 1, 11);
      if (downSpacing >= 14) {
        ctx.fillStyle = 'rgba(250,250,250,0.06)';
        ctx.fillRect(x, top, 1, waveH - 11);
      }
    }

    // energy curve
    const ev = analysis.energy?.values ?? [];
    const stepMs = analysis.energy?.step_ms ?? 0;
    if (ev.length > 1 && stepMs > 0) {
      const i0 = Math.max(0, Math.floor(t0 / stepMs) - 1);
      const i1 = Math.min(ev.length - 1, Math.ceil(t1 / stepMs) + 1);
      const yOf = (v: number) => top + 4 + (1 - Math.min(1, Math.max(0, v))) * (waveH - 8);
      ctx.beginPath();
      for (let i = i0; i <= i1; i++) {
        const x = xOf(i * stepMs);
        const y = yOf(ev[i]);
        if (i === i0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = 'rgba(251,191,36,0.95)';
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.stroke();
      ctx.lineTo(xOf(i1 * stepMs), H);
      ctx.lineTo(xOf(i0 * stepMs), H);
      ctx.closePath();
      ctx.fillStyle = 'rgba(251,191,36,0.08)';
      ctx.fill();
    }

    // drops: red line + flag
    const drops = analysis.drops_ms ?? [];
    const [r0, r1] = visibleRange(drops, t0 - 60 / ppm, t1);
    ctx.font = 'bold 9px Inter, system-ui, sans-serif';
    for (let i = r0; i < r1; i++) {
      const x = Math.round(xOf(drops[i])) + 0.5;
      ctx.strokeStyle = '#EF4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, H);
      ctx.stroke();
      const tw = ctx.measureText(dropLabel).width + 8;
      ctx.fillStyle = '#EF4444';
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x + tw, top);
      ctx.lineTo(x + tw - 4, top + 7);
      ctx.lineTo(x + tw, top + 14);
      ctx.lineTo(x, top + 14);
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = '#FFFFFF';
      ctx.textBaseline = 'middle';
      ctx.fillText(dropLabel, x + 3, top + 7.5);
    }
  }, [analysis, ppm, viewW, height, scrollEl, selectedSection, dropLabel]);

  const schedule = useCallback(() => {
    if (rafRef.current == null) rafRef.current = requestAnimationFrame(draw);
  }, [draw]);

  useEffect(() => { draw(); }, [draw]);

  useEffect(() => {
    if (!scrollEl) return;
    scrollEl.addEventListener('scroll', schedule, { passive: true });
    return () => scrollEl.removeEventListener('scroll', schedule);
  }, [scrollEl, schedule]);

  useEffect(() => () => { if (rafRef.current != null) cancelAnimationFrame(rafRef.current); }, []);

  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!(ppm > 0)) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left + (scrollEl?.scrollLeft ?? 0);
    onSeek(Math.min(Math.max(0, x / ppm), analysis.duration_ms || 0));
  };

  return (
    <canvas
      ref={canvasRef}
      onClick={onClick}
      title={title}
      className="absolute top-0 cursor-pointer block"
      style={{ width: viewW, height }}
    />
  );
}
