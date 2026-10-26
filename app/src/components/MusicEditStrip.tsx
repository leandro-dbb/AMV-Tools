// One block per proposed shot, on the same time axis as the waveform.
import { memo } from 'react';
import { Lock } from 'lucide-react';
import { api } from '../api/client';
import type { ProposedClip } from '../api/music';
import { formatTime, sectionColor } from './musicUtils';

interface Props {
  clips: ProposedClip[];
  sectionTypes: Record<number, string>;
  ppm: number;
  height: number;
  selectedSlot: number | null;
  playingSlot: number | null;
  onSelect: (slot: number) => void;
}

function MusicEditStrip({ clips, sectionTypes, ppm, height, selectedSlot, playingSlot, onSelect }: Props) {
  return (
    <div className="relative" style={{ height }}>
      {clips.map((c) => {
        const left = c.start_ms * ppm;
        const width = Math.max(2, (c.end_ms - c.start_ms) * ppm - 1);
        const sel = c.slot === selectedSlot;
        const playing = c.slot === playingSlot;
        const col = sectionColor(sectionTypes[c.section_index]);
        return (
          <button
            key={c.slot}
            type="button"
            onClick={() => onSelect(c.slot)}
            title={`${c.video_name} · ${formatTime(c.src_in_ms, { tenths: true })}`}
            className={`absolute top-0 overflow-hidden rounded-[3px] bg-[#27272A] transition-shadow ${
              sel ? 'ring-2 ring-[#FAFAFA] z-10' : playing ? 'ring-1 ring-[#A1A1AA]' : 'hover:ring-1 hover:ring-[#71717A]'
            }`}
            style={{ left, width, height, borderBottom: `3px solid ${col}` }}
          >
            {width >= 6 && (
              <img
                src={api.thumbnailUrl(c.scene_id)}
                alt=""
                loading="lazy"
                draggable={false}
                className="w-full h-full object-cover pointer-events-none select-none"
                onError={(e) => { (e.currentTarget as HTMLImageElement).style.visibility = 'hidden'; }}
              />
            )}
            {c.locked && (
              <span className="absolute top-0.5 left-0.5 bg-black/70 rounded p-0.5 text-[#FBBF24]">
                <Lock size={10} />
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

export default memo(MusicEditStrip);
