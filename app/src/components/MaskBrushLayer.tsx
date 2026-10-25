import { useRef, useState } from 'react';
import type { BrushStroke } from '../api/mask';

interface MaskBrushLayerProps {
  /** Natural frame size (stroke coordinates are sent in frame pixels). */
  frameW: number;
  frameH: number;
  mode: 'add' | 'remove';
  /** Brush radius in frame pixels. */
  radius: number;
  disabled?: boolean;
  onStroke: (stroke: BrushStroke) => void;
}

/** Transparent paint layer over the Mask Mode canvas. Draws the stroke being
 * painted locally (green = add, red = remove) and hands the finished stroke,
 * in frame-pixel coordinates, to ``onStroke``. */
export default function MaskBrushLayer({ frameW, frameH, mode, radius, disabled, onStroke }: MaskBrushLayerProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [live, setLive] = useState<[number, number][]>([]);
  const [hover, setHover] = useState<[number, number] | null>(null);

  function toFrame(e: { clientX: number; clientY: number }): [number, number] {
    const r = ref.current!.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * frameW, ((e.clientY - r.top) / r.height) * frameH];
  }

  function down(e: React.MouseEvent) {
    if (disabled || e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    const pts: [number, number][] = [toFrame(e)];
    setLive(pts);
    const move = (ev: MouseEvent) => {
      pts.push(toFrame(ev));
      setLive([...pts]);
    };
    const up = () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
      setLive([]);
      onStroke({ points: pts.map(([x, y]) => [Math.round(x), Math.round(y)]), radius, mode });
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  }

  const color = mode === 'add' ? 'rgba(74, 222, 128, 0.55)' : 'rgba(248, 113, 113, 0.55)';
  const sw = (radius * 2 * 100) / frameW;  // stroke width in viewBox units (% of width)
  return (
    <div
      ref={ref}
      className={`absolute inset-0 ${disabled ? 'cursor-not-allowed' : 'cursor-none'}`}
      onMouseDown={down}
      onMouseMove={(e) => setHover(toFrame(e))}
      onMouseLeave={() => setHover(null)}
    >
      <svg viewBox={`0 0 100 ${(frameH / frameW) * 100}`} preserveAspectRatio="none" className="w-full h-full pointer-events-none">
        {live.length > 0 && (
          <polyline
            points={live.map(([x, y]) => `${(x / frameW) * 100},${(y / frameW) * 100}`).join(' ')}
            fill="none" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round"
          />
        )}
        {hover && !disabled && (
          <circle cx={(hover[0] / frameW) * 100} cy={(hover[1] / frameW) * 100} r={sw / 2}
                  fill="none" stroke={mode === 'add' ? '#4ADE80' : '#F87171'} strokeWidth={0.25} />
        )}
      </svg>
    </div>
  );
}
