import { ChevronDown, Layers, Maximize2, Scissors, Timer, Wand2 } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useT } from '../i18n';
import { finishApi, type FinishCapabilities, type FinishKind, type FinishSource } from '../api/finish';
import FinishDialog from './FinishDialog';

const ENTRIES: { kind: FinishKind; icon: typeof Wand2 }[] = [
  { kind: 'upscale', icon: Maximize2 },
  { kind: 'interpolate', icon: Timer },
  { kind: 'depth', icon: Layers },
  { kind: 'matte', icon: Scissors },
];

/** "Finishing" dropdown: Upscale / Slow-mo-60 fps / Depth / Cutout for one
 * clip (MiniEditor, trimmed) or a batch (selects library). */
export default function FinishMenu({ sources, className }: { sources: FinishSource[]; className?: string }) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [kind, setKind] = useState<FinishKind | null>(null);
  const [caps, setCaps] = useState<FinishCapabilities | null>(null);
  const ref = useRef<HTMLDivElement | null>(null);

  const refreshCaps = useCallback(() => {
    finishApi.capabilities().then(setCaps).catch(() => setCaps(null));
  }, []);

  useEffect(() => { if (open || kind) refreshCaps(); }, [open, kind, refreshCaps]);

  useEffect(() => {
    if (!open) return;
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    window.addEventListener('mousedown', h);
    return () => window.removeEventListener('mousedown', h);
  }, [open]);

  const disabled = sources.length === 0;
  return (
    <div ref={ref} className={`relative ${className ?? ''}`}>
      <button
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        title={sources.length > 1 ? t('finish.menuTitleMany', { n: sources.length }) : t('finish.menuTitle')}
        className="px-4 py-2.5 rounded-xl text-sm font-semibold flex items-center gap-2 bg-[#27272A] hover:bg-[#3F3F46] text-[#FAFAFA] border-2 border-[#8B5CF6]/40 hover:border-[#8B5CF6] disabled:opacity-40 whitespace-nowrap"
      >
        <Wand2 size={16} className="text-[#A78BFA]" />
        {t('finish.menu')}
        <ChevronDown size={14} className="text-[#A1A1AA]" />
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-2 w-80 z-[60] bg-[#18181B] border-2 border-[#27272A] rounded-xl shadow-2xl shadow-black/60 p-1.5">
          {ENTRIES.map(({ kind: k, icon: Icon }) => {
            const off = k === 'matte' && caps !== null && !caps.mask_enabled;
            return (
              <button
                key={k}
                disabled={off}
                onClick={() => { setOpen(false); setKind(k); }}
                title={off ? t('finish.kind.matte.disabled') : undefined}
                className="w-full text-left px-3 py-2.5 rounded-lg hover:bg-[#27272A] flex items-start gap-3 disabled:opacity-40 disabled:hover:bg-transparent"
              >
                <Icon size={16} className="text-[#A78BFA] mt-0.5 flex-shrink-0" />
                <span>
                  <span className="block text-sm text-[#FAFAFA] font-medium">{t(`finish.kind.${k}`)}</span>
                  <span className="block text-[11px] text-[#71717A]">{off ? t('finish.kind.matte.disabled') : t(`finish.kind.${k}.desc`)}</span>
                </span>
              </button>
            );
          })}
        </div>
      )}
      {kind && (
        <FinishDialog kind={kind} sources={sources} caps={caps} onCapsChange={refreshCaps} onClose={() => setKind(null)} />
      )}
    </div>
  );
}
