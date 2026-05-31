import { useState } from 'react';
import { ChevronRight, FolderPlus, Search, Tag, Sparkles, X } from 'lucide-react';

interface Step {
  icon: typeof FolderPlus;
  title: string;
  body: string;
}

const STEPS: Step[] = [
  {
    icon: FolderPlus,
    title: '1. Add your library',
    body: 'Open Settings → Databases and drop a folder of anime episodes into the index queue. AMV Tools will detect cuts, sub-segment long scenes, and run wd-tagger + SigLIP 2.',
  },
  {
    icon: Search,
    title: '2. Search in plain English',
    body: 'Type "Gojo combat" or "Charizard fire breath" in the Search tab. Drop an image to find similar scenes. Tune the threshold if you want stricter or looser matches.',
  },
  {
    icon: Tag,
    title: '3. Browse by tag',
    body: 'In the Tags tab, pick one or several videos and a tag (e.g. "fighting") to see every occurrence with hover preview. Multi-select and bulk-export in one click.',
  },
  {
    icon: Sparkles,
    title: '4. Edit and export',
    body: 'Click any scene to open the mini-editor. J/K/L scrub, I/O set in-out, ← → walk the list. Hit Export scene when you\'re done — frame-accurate via ffmpeg.',
  },
];

export default function TutorialOverlay({ onDone }: { onDone: () => void }) {
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
          <div className="text-[#71717A] text-xs uppercase tracking-wider">Getting started</div>
        </div>

        <h2 className="text-2xl font-bold mb-2">{S.title}</h2>
        <p className="text-[#A1A1AA] mb-6">{S.body}</p>

        <div className="flex items-center justify-between">
          <div className="flex gap-1.5">
            {STEPS.map((_, i) => (
              <div key={i} className={`w-2 h-2 rounded-full transition-all ${i === step ? 'bg-[#8B5CF6] w-6' : 'bg-[#27272A]'}`} />
            ))}
          </div>
          <div className="flex items-center gap-2">
            <button onClick={onDone} className="text-[#71717A] hover:text-[#FAFAFA] text-sm">Skip</button>
            {step < STEPS.length - 1 ? (
              <button onClick={() => setStep(step + 1)} className="bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white px-5 py-2 rounded-lg text-sm font-semibold flex items-center gap-2">
                Next <ChevronRight size={14} />
              </button>
            ) : (
              <button onClick={onDone} className="bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white px-5 py-2 rounded-lg text-sm font-semibold">
                Let's go
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
