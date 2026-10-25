import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  Activity, ArrowDown, ArrowLeft, ArrowRight, ArrowUp, CircleDot, Copy, Loader2, Plus, UserPlus, ZoomIn, ZoomOut,
} from 'lucide-react';
import { api } from '../api/client';
import type { Character, MotionType, SceneResult } from '../api/types';
import { useT } from '../i18n';
import { requestSimilar } from '../searchplus';

const MOTION_ICONS: Record<MotionType, typeof ArrowLeft> = {
  static: CircleDot,
  pan_left: ArrowLeft,
  pan_right: ArrowRight,
  tilt_up: ArrowUp,
  tilt_down: ArrowDown,
  zoom_in: ZoomIn,
  zoom_out: ZoomOut,
  shake: Activity,
  unknown: CircleDot,
};

export function MotionIcon({ type, size = 12 }: { type: MotionType; size?: number }) {
  const Icon = MOTION_ICONS[type] ?? CircleDot;
  return <Icon size={size} />;
}

// Hue goes from calm yellow to hot red as intensity rises.
function intensityColor(v: number): string {
  const hue = Math.round(50 - 50 * Math.min(Math.max(v, 0), 1));
  return `hsl(${hue} 95% 55%)`;
}

/** Motion chip (top-left), palette dots and intensity bar (bottom) laid over
 * a scene thumbnail. Parent must be `relative`. */
export function SceneBadges({ scene }: { scene: SceneResult }) {
  const t = useT();
  const m = scene.metrics;
  if (!m) return null;
  const intensity = m.intensity ?? 0;
  const motion = m.motion_type;
  return (
    <>
      {motion && (
        <div
          className="absolute top-2 left-2 flex items-center gap-1 bg-black/70 backdrop-blur-sm text-[#FAFAFA] px-1.5 py-0.5 rounded-md text-[10px] font-semibold pointer-events-auto"
          title={t('sp.badge.motion', { motion: t(`sp.motion.${motion}`), value: Math.round((m.motion_magnitude ?? 0) * 100) })}
        >
          <MotionIcon type={motion} size={11} />
          {motion !== 'static' && <span className="hidden xl:inline">{t(`sp.motion.${motion}`)}</span>}
        </div>
      )}
      {m.palette.length > 0 && (
        <div className="absolute bottom-2.5 right-2 flex gap-0.5 pointer-events-none">
          {m.palette.map((c, i) => (
            <span key={i} className="w-2.5 h-2.5 rounded-full border border-black/60" style={{ background: c }} />
          ))}
        </div>
      )}
      <div className="absolute bottom-0 left-0 right-0 h-1 bg-black/50" title={t('sp.badge.intensity', { value: Math.round(intensity * 100) })}>
        <div className="h-full" style={{ width: `${Math.round(intensity * 100)}%`, background: intensityColor(intensity) }} />
      </div>
    </>
  );
}

export function CharacterChips({ scene, max = 2 }: { scene: SceneResult; max?: number }) {
  if (!scene.characters?.length) return null;
  return (
    <div className="flex gap-1 flex-wrap">
      {scene.characters.slice(0, max).map((c) => (
        <span key={c.id} className="bg-[#EC4899]/80 text-white px-2 py-0.5 rounded-full text-[10px] font-semibold" title={`${Math.round(c.score * 100)}%`}>
          {c.name}
        </span>
      ))}
    </div>
  );
}

/** "Similar shots" button — works from any tab (App switches to Search). */
export function SimilarButton({ sceneId, className = '', label = false }: { sceneId: number; className?: string; label?: boolean }) {
  const t = useT();
  return (
    <button
      type="button"
      title={t('sp.action.similarTitle')}
      onClick={(e) => { e.stopPropagation(); requestSimilar(sceneId); }}
      className={className || 'p-1.5 rounded-md bg-black/70 text-white/80 hover:text-white hover:bg-[#8B5CF6]'}
    >
      <span className="flex items-center gap-1.5"><Copy size={label ? 14 : 12} />{label && t('sp.action.similar')}</span>
    </button>
  );
}

/** Small popover: add the scene as a positive example of a character
 * (existing one, or a new one created on the fly). */
export function AddToCharacterButton({ sceneId, className = '' }: { sceneId: number; className?: string }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [chars, setChars] = useState<Character[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState<string | null>(null);
  const [newName, setNewName] = useState('');
  const [pos, setPos] = useState<{ top: number; left: number }>({ top: 0, left: 0 });
  const btnRef = useRef<HTMLButtonElement>(null);
  const popRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    api.listCharacters().then((r) => setChars(r.characters)).catch(() => setChars([]));
    const close = (e: MouseEvent) => {
      const n = e.target as Node;
      if (!popRef.current?.contains(n) && !btnRef.current?.contains(n)) setOpen(false);
    };
    window.addEventListener('mousedown', close);
    return () => window.removeEventListener('mousedown', close);
  }, [open]);

  function toggleOpen() {
    // The popover is portalled to <body>: cards clip overflow and scale on
    // hover, which would cut or offset an in-place dropdown.
    const r = btnRef.current?.getBoundingClientRect();
    if (r) setPos({ top: r.bottom + 4, left: Math.max(8, Math.min(r.right - 224, window.innerWidth - 232)) });
    setOpen((o) => !o);
  }

  async function add(c: Character) {
    setBusy(true);
    try {
      await api.addCharacterExample(c.id, { scene_id: sceneId, polarity: 1 });
      setDone(t('sp.action.added', { name: c.name }));
      setTimeout(() => { setOpen(false); setDone(null); }, 900);
    } catch (err) {
      setDone((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function create() {
    const name = newName.trim();
    if (!name) return;
    setBusy(true);
    try {
      const c = await api.createCharacter({ name, examples: [{ scene_id: sceneId }] });
      setDone(t('sp.action.added', { name: c.name }));
      setNewName('');
      setTimeout(() => { setOpen(false); setDone(null); }, 900);
    } catch (err) {
      setDone((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="relative" onClick={(e) => e.stopPropagation()}>
      <button
        ref={btnRef}
        type="button"
        title={t('sp.action.addToCharacter')}
        onClick={toggleOpen}
        className={className || 'p-1.5 rounded-md bg-black/70 text-white/80 hover:text-white hover:bg-[#EC4899]'}
      >
        <UserPlus size={12} />
      </button>
      {open && createPortal(
        <div ref={popRef} onClick={(e) => e.stopPropagation()} style={{ top: pos.top, left: pos.left }}
             className="fixed z-[60] w-56 bg-[#18181B] border border-[#27272A] rounded-xl shadow-2xl p-2 text-left">
          <div className="text-[10px] uppercase tracking-wider text-[#71717A] px-1 mb-1">{t('sp.action.addToCharacter')}</div>
          {chars === null && <Loader2 size={14} className="animate-spin text-[#8B5CF6] m-2" />}
          <div className="max-h-48 overflow-auto">
            {chars?.map((c) => (
              <button key={c.id} disabled={busy} onClick={() => add(c)}
                      className="w-full text-left px-2 py-1.5 rounded-md text-xs text-[#FAFAFA] hover:bg-[#27272A] flex items-center justify-between">
                <span className="truncate">{c.name}</span>
                {c.group_name && <span className="text-[10px] text-[#71717A] truncate ml-2">{c.group_name}</span>}
              </button>
            ))}
          </div>
          <div className="flex gap-1 mt-1 border-t border-[#27272A] pt-2">
            <input value={newName} onChange={(e) => setNewName(e.target.value)}
                   onKeyDown={(e) => { if (e.key === 'Enter') create(); }}
                   placeholder={t('sp.action.newCharacter')}
                   className="flex-1 min-w-0 bg-[#0F0F11] border border-[#27272A] rounded-md px-2 py-1 text-xs text-[#FAFAFA] focus:outline-none focus:border-[#EC4899]" />
            <button disabled={busy || !newName.trim()} onClick={create}
                    className="p-1.5 rounded-md bg-[#EC4899] text-white disabled:opacity-30"><Plus size={12} /></button>
          </div>
          {done && <div className="text-[11px] text-[#8B5CF6] mt-1 px-1">{done}</div>}
        </div>,
        document.body,
      )}
    </div>
  );
}
