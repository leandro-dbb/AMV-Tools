import { useEffect, useRef, useState } from 'react';
import { Cpu, Sparkles, Check, AlertCircle, Loader2, Zap, Target } from 'lucide-react';
import { api, platformIs } from '../api/client';
import { useI18n } from '../i18n';

type GpuBackend = 'cu130' | 'cu130-trt' | 'cu126' | 'dml' | 'xpu' | 'rocm' | 'mps' | 'cpu';

interface Option {
  id: GpuBackend;
  title: string;
  subtitle: string;
  recommended?: boolean;
  badge?: string;
  detail: string;
  platforms?: ('win32' | 'linux' | 'darwin')[];
}

// title/subtitle/detail/badge hold i18n keys; resolved with t() at render time.
const ALL_OPTIONS: Option[] = [
  { id: 'cu130-trt', title: 'onboarding.opt.cu130-trt.title', subtitle: 'onboarding.opt.cu130-trt.subtitle', badge: 'onboarding.badge.gpuEverywhere', detail: 'onboarding.opt.cu130-trt.detail', platforms: ['win32', 'linux'] },
  { id: 'cu130', title: 'onboarding.opt.cu130.title', subtitle: 'onboarding.opt.cu130.subtitle', recommended: true, badge: 'onboarding.badge.gpuEverywhere', detail: 'onboarding.opt.cu130.detail', platforms: ['win32', 'linux'] },
  { id: 'cu126', title: 'onboarding.opt.cu126.title', subtitle: 'onboarding.opt.cu126.subtitle', detail: 'onboarding.opt.cu126.detail', platforms: ['win32', 'linux'] },
  // `recommended` doesn't clash with cu130's: the two are never visible on
  // the same platform.
  { id: 'mps', title: 'onboarding.opt.mps.title', subtitle: 'onboarding.opt.mps.subtitle', recommended: true, badge: 'onboarding.badge.gpuEverywhere', detail: 'onboarding.opt.mps.detail', platforms: ['darwin'] },
  { id: 'dml', title: 'onboarding.opt.dml.title', subtitle: 'onboarding.opt.dml.subtitle', detail: 'onboarding.opt.dml.detail', platforms: ['win32'] },
  { id: 'rocm', title: 'onboarding.opt.rocm.title', subtitle: 'onboarding.opt.rocm.subtitle', detail: 'onboarding.opt.rocm.detail', platforms: ['linux'] },
  { id: 'xpu', title: 'onboarding.opt.xpu.title', subtitle: 'onboarding.opt.xpu.subtitle', detail: 'onboarding.opt.xpu.detail', platforms: ['win32', 'linux'] },
  { id: 'cpu', title: 'onboarding.opt.cpu.title', subtitle: 'onboarding.opt.cpu.subtitle', detail: 'onboarding.opt.cpu.detail' },
];

export default function OnboardingScreen({ onDone }: { onDone: () => void }) {
  const { lang, setLang, t } = useI18n();
  const [phase, setPhase] = useState<'choose' | 'installing' | 'done' | 'error'>('choose');
  const [selected, setSelected] = useState<GpuBackend>(platformIs('darwin') ? 'mps' : 'cu130');
  const [profile, setProfile] = useState<'fast' | 'quality'>('quality');
  const [log, setLog] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [pythonVer, setPythonVer] = useState<string>('');
  const logRef = useRef<HTMLPreElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    api.status().then((s) => setPythonVer(s.python_version)).catch(() => {});
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  const visibleOptions = ALL_OPTIONS.filter((o) => {
    if (!o.platforms) return true;
    return o.platforms.some((p) => platformIs(p));
  });

  async function startInstall() {
    setPhase('installing');
    setLog('');
    setError('');

    try {
      wsRef.current = api.progressSocket();
      wsRef.current.onmessage = (msg) => {
        try {
          const evt = JSON.parse(msg.data);
          if (evt.type === 'install') {
            if (evt.line) setLog((l) => l + evt.line);
          }
        } catch {}
      };
    } catch (err) {
      // proceed without live stream
    }

    try {
      const res = await api.setupGpu(selected);
      if (!log) setLog(res.log);
      if (res.ok) {
        // Apply the speed/quality preference chosen on the first screen.
        // Non-fatal: worst case the user keeps the shipped defaults.
        try {
          const s = await api.getSettings();
          await api.saveSettings({
            indexing: {
              ...s.indexing,
              mode: profile === 'fast' ? 'fast' : 'balanced',
              sub_segmentation: profile === 'quality',
            },
          });
        } catch {}
        setPhase('done');
        setTimeout(() => { wsRef.current?.close(); onDone(); }, 1200);
      } else {
        setPhase('error');
        setError(res.hint || t('onboarding.errorFallback'));
      }
    } catch (err) {
      setPhase('error');
      setError((err as Error).message);
    }
  }

  return (
    <div className="w-screen h-screen bg-gradient-to-br from-[#0F0F11] via-[#18181B] to-[#0F0F11] text-[#FAFAFA] flex items-center justify-center p-8 overflow-auto" style={{ fontFamily: 'Inter, Space Grotesk, system-ui, sans-serif' }}>
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-[#8B5CF6]/10 rounded-full blur-[150px] pointer-events-none"></div>

      <div className="relative max-w-4xl w-full">
        <div className="absolute top-0 right-0 flex gap-1.5 z-10">
          {(['en', 'fr'] as const).map((l) => (
            <button
              key={l}
              onClick={() => setLang(l)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold border-2 transition-all ${
                lang === l
                  ? 'border-[#8B5CF6] bg-[#8B5CF6]/10 text-[#FAFAFA]'
                  : 'border-[#27272A] text-[#71717A] hover:text-[#FAFAFA] hover:border-[#3f3f46]'
              }`}
            >
              {l === 'en' ? 'English' : 'Français'}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3 mb-2">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] rounded-lg blur-lg opacity-50"></div>
            <div className="relative bg-gradient-to-br from-[#8B5CF6] to-[#EC4899] p-3 rounded-xl">
              <Sparkles size={24} className="text-white" />
            </div>
          </div>
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-[#FAFAFA] to-[#A1A1AA] bg-clip-text text-transparent">{t('onboarding.title')}</h1>
            <div className="text-[#71717A] text-sm uppercase tracking-wider">{t('onboarding.tagline')}</div>
          </div>
        </div>

        <p className="text-[#A1A1AA] mb-8 max-w-2xl">
          {t('onboarding.intro')}{' '}
          {t('onboarding.introChangePrefix')}<span className="text-[#FAFAFA]">{t('onboarding.introSettingsPath')}</span>.
        </p>

        {phase === 'choose' && (
          <>
            <div className="grid grid-cols-2 gap-3 mb-6">
              {visibleOptions.map((opt) => {
                const isSel = selected === opt.id;
                return (
                  <button key={opt.id} onClick={() => setSelected(opt.id)}
                    className={`relative text-left p-5 rounded-xl border-2 transition-all ${isSel ? 'border-[#8B5CF6] bg-gradient-to-br from-[#8B5CF6]/10 to-[#EC4899]/10 shadow-lg shadow-[#8B5CF6]/20' : 'border-[#27272A] bg-[#18181B]/50 hover:border-[#3f3f46]'}`}>
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <Cpu size={16} className={isSel ? 'text-[#8B5CF6]' : 'text-[#71717A]'} />
                          <span className="font-semibold text-[#FAFAFA]">{t(opt.title)}</span>
                          {opt.recommended && <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#8B5CF6]/20 text-[#8B5CF6] font-bold uppercase tracking-wider">{t('common.recommended')}</span>}
                          {opt.badge && <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#EC4899]/20 text-[#EC4899] font-bold uppercase tracking-wider">{t(opt.badge)}</span>}
                        </div>
                        <div className="text-xs text-[#71717A] mb-2">{t(opt.subtitle)}</div>
                        <div className="text-xs text-[#A1A1AA]">{t(opt.detail)}</div>
                      </div>
                      {isSel && <Check size={20} className="text-[#8B5CF6] flex-shrink-0" />}
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="mb-2 font-semibold text-[#FAFAFA]">{t('onboarding.profile.title')}</div>
            <div className="grid grid-cols-2 gap-3 mb-2">
              {([
                { id: 'fast' as const, icon: Zap, title: 'onboarding.profile.fast.title', detail: 'onboarding.profile.fast.detail' },
                { id: 'quality' as const, icon: Target, title: 'onboarding.profile.quality.title', detail: 'onboarding.profile.quality.detail', recommended: true },
              ]).map((p) => {
                const isSel = profile === p.id;
                const Icon = p.icon;
                return (
                  <button key={p.id} onClick={() => setProfile(p.id)}
                    className={`relative text-left p-5 rounded-xl border-2 transition-all ${isSel ? 'border-[#EC4899] bg-gradient-to-br from-[#EC4899]/10 to-[#8B5CF6]/10 shadow-lg shadow-[#EC4899]/20' : 'border-[#27272A] bg-[#18181B]/50 hover:border-[#3f3f46]'}`}>
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <Icon size={16} className={isSel ? 'text-[#EC4899]' : 'text-[#71717A]'} />
                          <span className="font-semibold text-[#FAFAFA]">{t(p.title)}</span>
                          {p.recommended && <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#EC4899]/20 text-[#EC4899] font-bold uppercase tracking-wider">{t('common.recommended')}</span>}
                        </div>
                        <div className="text-xs text-[#A1A1AA]">{t(p.detail)}</div>
                      </div>
                      {isSel && <Check size={20} className="text-[#EC4899] flex-shrink-0" />}
                    </div>
                  </button>
                );
              })}
            </div>
            <div className="text-xs text-[#71717A] mb-6">{t('onboarding.profile.changeLater')}</div>

            <div className="flex justify-end gap-3">
              <button onClick={startInstall} className="bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-[#FAFAFA] px-8 py-3 rounded-xl text-sm font-semibold hover:shadow-lg hover:shadow-[#8B5CF6]/50 transition-all">
                {t('onboarding.installBtn')}
              </button>
            </div>
          </>
        )}

        {phase === 'installing' && (
          <div className="bg-[#18181B] border-2 border-[#27272A] rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <Loader2 size={20} className="text-[#8B5CF6] animate-spin" />
              <div>
                <div className="font-semibold">{t('onboarding.installingTitle', { backend: selected.toUpperCase() })}</div>
                <div className="text-xs text-[#71717A]">{t('onboarding.installingHint')}</div>
              </div>
            </div>
            <pre ref={logRef} className="bg-[#0F0F11] border border-[#27272A] rounded-lg p-3 text-xs text-[#A1A1AA] font-mono max-h-[400px] overflow-auto whitespace-pre-wrap">
              {log || t('onboarding.logPlaceholder')}
            </pre>
          </div>
        )}

        {phase === 'done' && (
          <div className="bg-gradient-to-br from-[#8B5CF6]/20 to-[#EC4899]/20 border-2 border-[#8B5CF6]/50 rounded-xl p-6 flex items-center gap-3">
            <Check size={24} className="text-[#8B5CF6]" />
            <div>
              <div className="font-semibold">{t('onboarding.doneTitle')}</div>
              <div className="text-xs text-[#A1A1AA]">{t('onboarding.doneHint')}</div>
            </div>
          </div>
        )}

        {phase === 'error' && (
          <div className="bg-red-500/10 border-2 border-red-500/50 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-3">
              <AlertCircle size={20} className="text-red-400" />
              <div className="font-semibold text-red-300">{t('onboarding.errorTitle')}</div>
            </div>
            <pre className="bg-[#0F0F11] border border-[#27272A] rounded-lg p-3 text-xs text-[#A1A1AA] font-mono max-h-[300px] overflow-auto whitespace-pre-wrap mb-3">{log}</pre>
            <div className="text-xs text-red-400 mb-3">{error}</div>
            <button onClick={() => setPhase('choose')} className="bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-5 py-2 rounded-lg text-sm">{t('onboarding.tryAnother')}</button>
          </div>
        )}

        <div className="mt-6 text-xs text-[#71717A] text-center">
          {pythonVer ? `Python ${pythonVer} - ` : ''}{t('onboarding.version')}
        </div>
      </div>
    </div>
  );
}
