import { useState } from 'react';
import { ChevronRight, FolderPlus, Search, Tag, Sparkles, X } from 'lucide-react';
import { useT } from '../i18n';

interface Step {
  icon: typeof FolderPlus;
  title: string;
  body: string;
}

// title/body hold i18n keys; resolved with t() at render time.
const STEPS: Step[] = [
  {
    icon: FolderPlus,
    title: 'tutorial.step1.title',
    body: 'tutorial.step1.body',
  },
  {
    icon: Search,
    title: 'tutorial.step2.title',
    body: 'tutorial.step2.body',
  },
  {
    icon: Tag,
    title: 'tutorial.step3.title',
    body: 'tutorial.step3.body',
  },
  {
    icon: Sparkles,
    title: 'tutorial.step4.title',
    body: 'tutorial.step4.body',
  },
];

export default function TutorialOverlay({ onDone }: { onDone: () => void }) {
  const t = useT();
  const [step, setStep] = useState(0);
  const S = STEPS[step];
  const Icon = S.icon;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-[60] p-8">
      <div className="bg-gradient-to-br from-[#18181B] via-[#0F0F11] to-[#18181B] border-2 border-[#8B5CF6]/40 rounded-2xl p-8 max-w-xl w-full shadow-2xl shadow-[#8B5CF6]/30 relative">
        <button onClick={onDone} className="absolute top-4 right-4 text-[#71717A] hover:text-[#FAFAFA]">
          <X size={20} />
        </button>

        <div className="flex items-center gap-3 mb-4">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] rounded-lg blur opacity-60"></div>
            <div className="relative bg-gradient-to-br from-[#8B5CF6] to-[#EC4899] p-3 rounded-xl">
              <Icon size={24} className="text-white" />
            </div>
          </div>
          <div className="text-[#71717A] text-xs uppercase tracking-wider">{t('tutorial.gettingStarted')}</div>
        </div>

        <h2 className="text-2xl font-bold mb-2">{t(S.title)}</h2>
        <p className="text-[#A1A1AA] mb-6">{t(S.body)}</p>

        <div className="flex items-center justify-between">
          <div className="flex gap-1.5">
            {STEPS.map((_, i) => (
              <div key={i} className={`w-2 h-2 rounded-full transition-all ${i === step ? 'bg-[#8B5CF6] w-6' : 'bg-[#27272A]'}`} />
            ))}
          </div>
          <div className="flex items-center gap-2">
            <button onClick={onDone} className="text-[#71717A] hover:text-[#FAFAFA] text-sm">{t('tutorial.skip')}</button>
            {step < STEPS.length - 1 ? (
              <button onClick={() => setStep(step + 1)} className="bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white px-5 py-2 rounded-lg text-sm font-semibold flex items-center gap-2">
                {t('tutorial.next')} <ChevronRight size={14} />
              </button>
            ) : (
              <button onClick={onDone} className="bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white px-5 py-2 rounded-lg text-sm font-semibold">
                {t('tutorial.letsGo')}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
