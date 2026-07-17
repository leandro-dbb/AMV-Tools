import { useEffect, useRef, useState } from 'react';
import { Cpu, Zap, Target, Layers, FileVideo, Palette, Download, Settings2, FolderOpen, Loader2, Trash2, Upload, AlertTriangle, Clapperboard, RotateCcw } from 'lucide-react';
import { api } from '../api/client';
import type { AppSettings } from '../api/types';
import { DERUSH_ACTIONS, DERUSH_DEFAULT_KEYS, displayKey } from '../derushKeys';
import { useI18n, useT, type Lang } from '../i18n';

const sections = [
  { name: 'Indexing', icon: Cpu },
  { name: 'Models', icon: Zap },
  { name: 'Search', icon: Target },
  { name: 'Editor', icon: Layers },
  { name: 'Derush', icon: Clapperboard },
  { name: 'Export', icon: FileVideo },
  { name: 'Interface', icon: Palette },
  { name: 'Updates', icon: Download },
  { name: 'Advanced', icon: Settings2 },
] as const;

type SectionName = typeof sections[number]['name'];

export default function SettingsTab({ runtimeDevice, gpuName, vramGb }: { runtimeDevice?: string; gpuName?: string; vramGb?: number | null }) {
  const t = useT();
  const [activeSection, setActiveSection] = useState<SectionName>('Indexing');
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getSettings().then(setSettings).catch((e) => setError(e.message));
  }, []);

  async function patch<K extends keyof AppSettings>(section: K, partial: Partial<AppSettings[K]>) {
    if (!settings) return;
    const next = { ...settings, [section]: { ...settings[section], ...partial } };
    setSettings(next);
    setSaving(true);
    try {
      const saved = await api.saveSettings({ [section]: next[section] } as Partial<AppSettings>);
      setSettings(saved);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="flex h-full">
      <div className="w-64 bg-gradient-to-b from-[#18181B] to-[#0F0F11] border-r border-[#27272A] p-5">
        <div className="flex flex-col gap-2">
          {sections.map((section) => {
            const Icon = section.icon;
            const isActive = activeSection === section.name;
            return (
              <button key={section.name} onClick={() => setActiveSection(section.name)} className={`relative text-left px-4 py-3 rounded-xl text-sm transition-all flex items-center gap-3 ${isActive ? 'bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-[#FAFAFA] shadow-lg shadow-[#8B5CF6]/30' : 'text-[#A1A1AA] hover:bg-[#27272A] hover:text-[#FAFAFA]'}`}>
                {isActive && <div className="absolute inset-0 bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] rounded-xl blur opacity-50"></div>}
                <Icon size={18} className="relative z-10" />
                <span className="relative z-10 font-medium">{t(`settings.section.${section.name.toLowerCase()}`)}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex-1 p-8 overflow-auto relative">
        <div className="absolute top-0 right-0 w-[400px] h-[400px] bg-[#8B5CF6]/5 rounded-full blur-[100px] pointer-events-none"></div>

        <div className="relative z-10 mb-8 flex items-end justify-between">
          <div>
            <h2 className="text-3xl font-bold bg-gradient-to-r from-[#FAFAFA] to-[#A1A1AA] bg-clip-text text-transparent">{t(`settings.section.${activeSection.toLowerCase()}`)}</h2>
            <div className="h-1 w-20 bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] rounded-full mt-2"></div>
          </div>
          {saving && <div className="text-xs text-[#71717A] flex items-center gap-1"><Loader2 size={12} className="animate-spin" /> {t('common.saving')}</div>}
        </div>

        {error && <div className="relative z-10 mb-4 bg-red-500/10 border border-red-500/40 text-red-300 text-sm rounded-lg px-4 py-2">{error}</div>}

        {!settings ? (
          <div className="relative z-10 text-[#71717A] flex items-center gap-2"><Loader2 size={14} className="animate-spin" /> {t('settings.loading')}</div>
        ) : (
          <div className="flex flex-col gap-4 max-w-4xl relative z-10">
            {activeSection === 'Indexing' && <IndexingSection settings={settings} patch={patch} runtimeDevice={runtimeDevice} gpuName={gpuName} vramGb={vramGb} />}
            {activeSection === 'Models' && <ModelsSection settings={settings} patch={patch} runtimeDevice={runtimeDevice} />}
            {activeSection === 'Search' && <SearchSection settings={settings} patch={patch} />}
            {activeSection === 'Editor' && <EditorSection settings={settings} patch={patch} />}
            {activeSection === 'Derush' && <DerushSection settings={settings} patch={patch} />}
            {activeSection === 'Export' && <ExportSection settings={settings} patch={patch} />}
            {activeSection === 'Interface' && <InterfaceSection settings={settings} patch={patch} />}
            {activeSection === 'Updates' && <UpdatesSection />}
            {activeSection === 'Advanced' && <AdvancedSection settings={settings} setSettings={setSettings} />}
          </div>
        )}
      </div>
    </div>
  );
}

type Patch = <K extends keyof AppSettings>(section: K, partial: Partial<AppSettings[K]>) => void;

function Row({ label, hint, children }: { label: string; hint: string; children: React.ReactNode }) {
  return (
    <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5 hover:border-[#8B5CF6]/30 transition-all">
      <div className="flex items-center justify-between">
        <div className="flex flex-col flex-1">
          <span className="text-[#FAFAFA] font-semibold mb-1">{label}</span>
          <span className="text-[#71717A] text-sm">{hint}</span>
        </div>
        {children}
      </div>
    </div>
  );
}

function Toggle({ on, onChange }: { on: boolean; onChange: (v: boolean) => void }) {
  return (
    <button onClick={() => onChange(!on)} className={`relative inline-flex h-7 w-14 items-center rounded-full transition-all ${on ? 'bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] shadow-lg shadow-[#8B5CF6]/30' : 'bg-[#27272A]'}`}>
      <span className={`inline-block h-5 w-5 transform rounded-full bg-white transition-transform shadow-lg ${on ? 'translate-x-8' : 'translate-x-1'}`} />
    </button>
  );
}

function Slider({ value, min, max, step, onChange }: { value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="flex items-center gap-4">
      <div className="relative w-40 h-2 bg-gradient-to-r from-[#27272A] via-[#8B5CF6]/20 to-[#8B5CF6]/40 rounded-full overflow-hidden">
        <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="absolute inset-0 w-full opacity-0 cursor-pointer" />
        <div className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-[#8B5CF6] rounded-full border-2 border-[#FAFAFA] shadow-lg shadow-[#8B5CF6]/50 pointer-events-none" style={{ left: `${pct}%` }}></div>
      </div>
      <span className="text-[#8B5CF6] font-mono font-bold w-12 text-right">{value}</span>
    </div>
  );
}

// Rough VRAM → batch-size heuristic for SigLIP + wd-tagger inference at the
// default 128-patch budget. Deliberately conservative: an OOM mid-index costs
// far more time than a slightly under-filled GPU.
function recommendedBatch(vramGb: number): number {
  if (vramGb >= 24) return 32;
  if (vramGb >= 16) return 24;
  if (vramGb >= 12) return 16;
  if (vramGb >= 10) return 12;
  if (vramGb >= 8) return 8;
  if (vramGb >= 6) return 6;
  return 4;
}

function IndexingSection({ settings, patch, runtimeDevice, gpuName, vramGb }: { settings: AppSettings; patch: Patch; runtimeDevice?: string; gpuName?: string; vramGb?: number | null }) {
  const t = useT();
  const s = settings.indexing;
  const isCpu = s.device === 'cpu';
  const cpuRuntime = isCpu || runtimeDevice === 'cpu';
  const rec = !cpuRuntime && typeof vramGb === 'number' && vramGb > 0 ? recommendedBatch(vramGb) : null;
  const batchHint = `${t('settings.batch.hint')} ${
    cpuRuntime
      ? t('settings.batch.rec.cpu')
      : rec !== null
        ? t('settings.batch.rec', { gpu: gpuName ?? 'GPU', vram: vramGb!, rec })
        : ''
  }`.trim();
  return (
    <>
      <Row label={t('settings.indexing.accel.label')} hint={t('settings.indexing.accel.hint')}>
        <div className="flex gap-2">
          <button
            onClick={() => patch('indexing', { device: 'auto' })}
            className={`px-5 py-2 rounded-lg text-sm font-semibold flex items-center gap-2 transition-all ${
              !isCpu
                ? 'bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white shadow-lg shadow-[#8B5CF6]/30'
                : 'bg-[#27272A] text-[#71717A] hover:text-[#FAFAFA]'
            }`}
          >
            <Zap size={14} /> GPU
          </button>
          <button
            onClick={() => patch('indexing', { device: 'cpu' })}
            className={`px-5 py-2 rounded-lg text-sm font-semibold flex items-center gap-2 transition-all ${
              isCpu
                ? 'bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white shadow-lg shadow-[#8B5CF6]/30'
                : 'bg-[#27272A] text-[#71717A] hover:text-[#FAFAFA]'
            }`}
          >
            <Cpu size={14} /> CPU
          </button>
        </div>
      </Row>
      <Row label={t('settings.indexing.device.label')} hint={t('settings.indexing.device.hint')}>
        <select value={s.device} onChange={(e) => patch('indexing', { device: e.target.value })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value="auto">{t('settings.indexing.device.auto')}</option>
          <option value="cuda">CUDA (NVIDIA)</option>
          <option value="dml">{t('settings.indexing.device.dml')}</option>
          <option value="xpu">Intel XPU</option>
          <option value="mps">Apple MPS</option>
          <option value="cpu">CPU</option>
        </select>
      </Row>
      <Row label={t('settings.batch.label')} hint={batchHint}>
        <div className="flex items-center gap-3">
          {rec !== null && rec !== s.batch_size && (
            <button
              onClick={() => patch('indexing', { batch_size: rec })}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-[#8B5CF6]/20 text-[#A78BFA] border border-[#8B5CF6]/40 hover:bg-[#8B5CF6]/30 whitespace-nowrap"
            >
              {t('settings.batch.apply')} {rec}
            </button>
          )}
          <Slider value={s.batch_size} min={1} max={64} step={1} onChange={(v) => patch('indexing', { batch_size: v })} />
        </div>
      </Row>
      <Row label={t('settings.indexing.mode.label')} hint={t('settings.indexing.mode.hint')}>
        <div className="flex gap-2">
          {(['fast', 'balanced', 'accurate'] as const).map((m) => (
            <button key={m} onClick={() => patch('indexing', { mode: m })} className={`px-4 py-2 rounded-lg text-sm capitalize ${s.mode === m ? 'bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white shadow-lg shadow-[#8B5CF6]/30' : 'bg-[#27272A] text-[#71717A] hover:text-[#FAFAFA]'}`}>{m}</button>
          ))}
        </div>
      </Row>
      <Row label={t('settings.subseg.label')} hint={s.sub_segmentation && s.mode === 'fast' ? t('settings.subseg.fastWarning') : t('settings.subseg.hint')}>
        <Toggle on={s.sub_segmentation} onChange={(v) => patch('indexing', { sub_segmentation: v })} />
      </Row>
      <Row label={t('settings.subseg.threshold.label')} hint={t('settings.subseg.threshold.hint')}>
        <Slider value={Math.round(s.sub_segmentation_threshold * 100)} min={20} max={50} step={1} onChange={(v) => patch('indexing', { sub_segmentation_threshold: v / 100 })} />
      </Row>
      <Row label={t('settings.indexing.proxies.label')} hint={t('settings.indexing.proxies.hint')}>
        <Toggle on={s.generate_proxies} onChange={(v) => patch('indexing', { generate_proxies: v })} />
      </Row>
      <Row label={t('settings.indexing.proxyQuality.label')} hint={t('settings.indexing.proxyQuality.hint')}>
        <select value={s.proxy_quality} onChange={(e) => patch('indexing', { proxy_quality: e.target.value as any })} className="bg-[#0F0F11] border-2 border-[#EC4899]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#EC4899]">
          <option value="low">{t('settings.indexing.proxyQuality.low')}</option>
          <option value="medium">{t('settings.indexing.proxyQuality.medium')}</option>
          <option value="high">{t('settings.indexing.proxyQuality.high')}</option>
        </select>
      </Row>
      <Row label={t('settings.indexing.autoSkip.label')} hint={t('settings.indexing.autoSkip.hint')}>
        <Toggle on={s.auto_skip_indexed} onChange={(v) => patch('indexing', { auto_skip_indexed: v })} />
      </Row>
      <Row label={t('settings.indexing.sceneThreshold.label')} hint={t('settings.indexing.sceneThreshold.hint')}>
        <Slider value={Math.round(s.scene_detect_threshold * 100)} min={10} max={60} step={1} onChange={(v) => patch('indexing', { scene_detect_threshold: v / 100 })} />
      </Row>
    </>
  );
}

function ModelsSection({ settings, patch, runtimeDevice }: { settings: AppSettings; patch: Patch; runtimeDevice?: string }) {
  const t = useT();
  const s = settings.models;
  const sam2Available = runtimeDevice !== 'dml';
  return (
    <>
      <Row label={t('settings.models.wd.label')} hint={t('settings.models.wd.hint')}>
        <select value={s.wd_tagger_variant} onChange={(e) => patch('models', { wd_tagger_variant: e.target.value })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value="SmilingWolf/wd-vit-tagger-v3">{t('settings.models.wd.opt.default')}</option>
          <option value="SmilingWolf/wd-vit-large-tagger-v3">{t('settings.models.wd.opt.better')}</option>
          <option value="SmilingWolf/wd-eva02-large-tagger-v3">{t('settings.models.wd.opt.best')}</option>
        </select>
      </Row>
      <Row label={t('settings.models.siglip.label')} hint={t('settings.models.siglip.hint')}>
        <select value={s.siglip_variant} onChange={(e) => patch('models', { siglip_variant: e.target.value })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value="google/siglip2-base-patch16-naflex">{t('settings.models.siglip.opt.base')}</option>
          <option value="google/siglip2-so400m-patch16-naflex">{t('settings.models.siglip.opt.so400m')}</option>
        </select>
      </Row>
      <Row label={t('settings.models.patches.label')} hint={t('settings.models.patches.hint')}>
        <select value={s.siglip_max_num_patches} onChange={(e) => patch('models', { siglip_max_num_patches: Number(e.target.value) })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value={64}>{t('settings.models.patches.opt.64')}</option>
          <option value={128}>{t('settings.models.patches.opt.128')}</option>
          <option value={192}>192</option>
          <option value={256}>{t('settings.models.patches.opt.256')}</option>
        </select>
      </Row>
      <Row label={t('settings.models.tagThreshold.label')} hint={t('settings.models.tagThreshold.hint')}>
        <Slider value={Math.round(s.default_tag_threshold * 100)} min={30} max={70} step={1} onChange={(v) => patch('models', { default_tag_threshold: v / 100 })} />
      </Row>
      <Row label={t('settings.models.vramOffload.label')} hint={t('settings.models.vramOffload.hint')}>
        <select value={s.vram_idle_offload} onChange={(e) => patch('models', { vram_idle_offload: Number(e.target.value) })} className="bg-[#0F0F11] border-2 border-[#EC4899]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#EC4899]">
          <option value={0}>{t('common.off')}</option>
          <option value={30}>30 s</option>
          <option value={60}>60 s</option>
          <option value={300}>5 min</option>
        </select>
      </Row>
      <Row label={t('settings.models.tensorrt.label')} hint={t('settings.models.tensorrt.hint')}>
        <Toggle on={s.use_tensorrt} onChange={(v) => patch('models', { use_tensorrt: v })} />
      </Row>
      <Row label={t('settings.models.mask.label')} hint={t('settings.models.mask.hint')}>
        <Toggle on={!!s.enable_mask} onChange={(v) => patch('models', { enable_mask: v })} />
      </Row>
      <Row label={t('settings.models.maskEngine.label')} hint={t('settings.models.maskEngine.hint')}>
        <select
          value={sam2Available ? (s.mask_engine || 'birefnet') : 'birefnet'}
          onChange={(e) => patch('models', { mask_engine: e.target.value as 'birefnet' | 'sam2' })}
          disabled={!s.enable_mask}
          className="bg-[#0F0F11] border-2 border-[#A855F7]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#A855F7] disabled:opacity-40"
        >
          <option value="birefnet">{t('settings.models.maskEngine.opt.birefnet')}</option>
          <option value="sam2" disabled={!sam2Available}>{t('settings.models.maskEngine.opt.sam2')}</option>
        </select>
      </Row>
      <Row label={t('settings.models.autoModel.label')} hint={t('settings.models.autoModel.hint')}>
        <select
          value={s.birefnet_variant || 'general'}
          onChange={(e) => patch('models', { birefnet_variant: e.target.value as 'general' | 'hr' | 'portrait' | 'anime' })}
          disabled={!s.enable_mask || s.mask_engine !== 'birefnet'}
          className="bg-[#0F0F11] border-2 border-[#A855F7]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#A855F7] disabled:opacity-40"
        >
          <option value="anime">{t('settings.models.autoModel.opt.anime')}</option>
          <option value="general">{t('settings.models.autoModel.opt.general')}</option>
          <option value="hr">{t('settings.models.autoModel.opt.hr')}</option>
          <option value="portrait">{t('settings.models.autoModel.opt.portrait')}</option>
        </select>
      </Row>
      <Row label={t('settings.models.sam2.label')} hint={t('settings.models.sam2.hint')}>
        <select
          value={s.sam2_variant || 'base_plus'}
          onChange={(e) => patch('models', { sam2_variant: e.target.value as 'tiny' | 'small' | 'base_plus' | 'large' })}
          disabled={!s.enable_mask || !sam2Available || s.mask_engine !== 'sam2'}
          className="bg-[#0F0F11] border-2 border-[#A855F7]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#A855F7] disabled:opacity-40"
        >
          <option value="tiny">{t('settings.models.sam2.opt.tiny')}</option>
          <option value="small">{t('settings.models.sam2.opt.small')}</option>
          <option value="base_plus">{t('settings.models.sam2.opt.basePlus')}</option>
          <option value="large">{t('settings.models.sam2.opt.large')}</option>
        </select>
      </Row>
      <Row label={t('settings.models.rotoRes.label')} hint={t('settings.models.rotoRes.hint')}>
        <Slider
          value={s.mask_max_dim ?? 1080}
          min={720}
          max={1440}
          step={120}
          onChange={(v) => patch('models', { mask_max_dim: v })}
        />
      </Row>
      <Row label={t('settings.models.edgeRefine.label')} hint={t('settings.models.edgeRefine.hint')}>
        <Toggle on={s.mask_edge_refine_enabled ?? true} onChange={(v) => patch('models', { mask_edge_refine_enabled: v })} />
      </Row>
      <Row label={t('settings.models.bgAware.label')} hint={t('settings.models.bgAware.hint')}>
        <Toggle on={s.mask_bg_aware_cleanup_enabled ?? true} onChange={(v) => patch('models', { mask_bg_aware_cleanup_enabled: v })} />
      </Row>
      <Row label={t('settings.models.temporal.label')} hint={t('settings.models.temporal.hint')}>
        <Toggle on={s.mask_temporal_smooth_enabled ?? true} onChange={(v) => patch('models', { mask_temporal_smooth_enabled: v })} />
      </Row>
      <Row label={t('settings.models.smoothStrength.label')} hint={t('settings.models.smoothStrength.hint')}>
        <Slider
          value={Math.round((s.mask_temporal_smooth_strength ?? 0.5) * 100)}
          min={10}
          max={100}
          step={5}
          onChange={(v) => patch('models', { mask_temporal_smooth_strength: v / 100 })}
        />
      </Row>
      <Row label={t('settings.models.hardShrink.label')} hint={t('settings.models.hardShrink.hint')}>
        <Slider value={s.mask_shrink_px ?? 0} min={0} max={5} step={1} onChange={(v) => patch('models', { mask_shrink_px: v })} />
      </Row>
      <Row label={t('settings.models.bgSuppress.label')} hint={t('settings.models.bgSuppress.hint')}>
        <Toggle on={s.mask_bg_suppress_enabled ?? false} onChange={(v) => patch('models', { mask_bg_suppress_enabled: v })} />
      </Row>
      <Row label={t('settings.models.bgSuppressThreshold.label')} hint={t('settings.models.bgSuppressThreshold.hint')}>
        <Slider
          value={s.mask_bg_suppress_threshold ?? 25}
          min={10}
          max={50}
          step={1}
          onChange={(v) => patch('models', { mask_bg_suppress_threshold: v })}
        />
      </Row>
      <Row label={t('settings.models.softBg.label')} hint={t('settings.models.softBg.hint')}>
        <Toggle on={s.mask_soft_bg_suppress_enabled ?? false} onChange={(v) => patch('models', { mask_soft_bg_suppress_enabled: v })} />
      </Row>
      <Row label={t('settings.models.softBgThreshold.label')} hint={t('settings.models.softBgThreshold.hint')}>
        <Slider
          value={s.mask_soft_bg_suppress_threshold ?? 25}
          min={8}
          max={40}
          step={1}
          onChange={(v) => patch('models', { mask_soft_bg_suppress_threshold: v })}
        />
      </Row>
      <Row label={t('settings.models.softBand.label')} hint={t('settings.models.softBand.hint')}>
        <Slider
          value={s.mask_soft_bg_suppress_edge_px ?? 16}
          min={2}
          max={24}
          step={1}
          onChange={(v) => patch('models', { mask_soft_bg_suppress_edge_px: v })}
        />
      </Row>
      <Row label={t('settings.models.softShrink.label')} hint={t('settings.models.softShrink.hint')}>
        <Slider
          value={s.mask_soft_shrink_px ?? 0}
          min={0}
          max={3}
          step={1}
          onChange={(v) => patch('models', { mask_soft_shrink_px: v })}
        />
      </Row>
      <Row label={t('settings.models.softLowCut.label')} hint={t('settings.models.softLowCut.hint')}>
        <Slider
          value={Math.round((s.mask_soft_alpha_black ?? 0.0) * 100)}
          min={0}
          max={25}
          step={1}
          onChange={(v) => patch('models', { mask_soft_alpha_black: v / 100 })}
        />
      </Row>
      <Row label={t('settings.models.rgbDecontam.label')} hint={t('settings.models.rgbDecontam.hint')}>
        <Toggle on={s.mask_rgb_decontaminate_enabled ?? false} onChange={(v) => patch('models', { mask_rgb_decontaminate_enabled: v })} />
      </Row>
      <Row label={t('settings.models.hfToken.label')} hint={t('settings.models.hfToken.hint')}>
        <input
          type="password"
          value={s.hf_token}
          onChange={(e) => patch('models', { hf_token: e.target.value })}
          placeholder="hf_..."
          className="bg-[#0F0F11] border-2 border-[#27272A] rounded-lg px-4 py-2.5 text-[#FAFAFA] text-sm font-mono focus:outline-none focus:border-[#8B5CF6] w-72"
        />
      </Row>
    </>
  );
}

function SearchSection({ settings, patch }: { settings: AppSettings; patch: Patch }) {
  const t = useT();
  const s = settings.search;
  return (
    <>
      <Row label={t('settings.search.threshold.label')} hint={t('settings.search.threshold.hint')}>
        <Slider value={Math.round(s.threshold * 100)} min={5} max={50} step={1} onChange={(v) => patch('search', { threshold: v / 100 })} />
      </Row>
      <Row label={t('settings.search.maxResults.label')} hint={t('settings.search.maxResults.hint')}>
        <select value={s.max_results} onChange={(e) => patch('search', { max_results: Number(e.target.value) })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value={50}>50</option>
          <option value={200}>200</option>
          <option value={1000}>1000</option>
          <option value={10000}>{t('settings.search.maxResults.unlimited')}</option>
        </select>
      </Row>
      <Row label={t('settings.search.tagBoost.label')} hint={t('settings.search.tagBoost.hint')}>
        <Slider value={Math.round(s.tag_boost * 100)} min={0} max={30} step={1} onChange={(v) => patch('search', { tag_boost: v / 100 })} />
      </Row>
    </>
  );
}

function EditorSection({ settings, patch }: { settings: AppSettings; patch: Patch }) {
  const t = useT();
  const s = settings.interface;
  return (
    <>
      <Row label={t('settings.editor.hoverDelay.label')} hint={t('settings.editor.hoverDelay.hint')}>
        <select value={s.hover_delay_ms} onChange={(e) => patch('interface', { hover_delay_ms: Number(e.target.value) })} className="bg-[#0F0F11] border-2 border-[#EC4899]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#EC4899]">
          <option value={100}>100 ms</option>
          <option value={200}>200 ms</option>
          <option value={500}>500 ms</option>
        </select>
      </Row>
      <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
        <div className="text-[#FAFAFA] font-semibold mb-2">{t('settings.editor.shortcuts.title')}</div>
        <div className="grid grid-cols-2 gap-2 text-sm font-mono">
          <div className="text-[#71717A]">J / K / L</div><div className="text-[#A1A1AA]">{t('settings.editor.shortcuts.jkl')}</div>
          <div className="text-[#71717A]">, / .</div><div className="text-[#A1A1AA]">{t('settings.editor.shortcuts.step')}</div>
          <div className="text-[#71717A]">I / O</div><div className="text-[#A1A1AA]">{t('settings.editor.shortcuts.inout')}</div>
          <div className="text-[#71717A]">← / →</div><div className="text-[#A1A1AA]">{t('settings.editor.shortcuts.prevnext')}</div>
          <div className="text-[#71717A]">Space</div><div className="text-[#A1A1AA]">{t('settings.editor.shortcuts.space')}</div>
        </div>
      </div>
    </>
  );
}

function DerushSection({ settings, patch }: { settings: AppSettings; patch: Patch }) {
  const t = useT();
  const keys = { ...DERUSH_DEFAULT_KEYS, ...(settings.derush?.keys ?? {}) };
  const setKey = (action: string, key: string) => {
    // Steal the key from any action that already uses it, so two actions can
    // never silently share a binding.
    const next = { ...keys };
    for (const a of Object.keys(next)) {
      if (next[a] === key && a !== action) next[a] = '';
    }
    next[action] = key;
    patch('derush', { keys: next });
  };
  return (
    <>
      <div className="text-sm text-[#71717A] -mb-1">
        {t('settings.derush.intro')}
      </div>
      {DERUSH_ACTIONS.map((a) => (
        <Row key={a.id} label={t(`derush.action.${a.id}.label`)} hint={t(`derush.action.${a.id}.hint`)}>
          <KeyCapture value={keys[a.id] ?? ''} onChange={(k) => setKey(a.id, k)} />
        </Row>
      ))}
      <Row label={t('settings.derush.reset.label')} hint={t('settings.derush.reset.hint')}>
        <button
          onClick={() => patch('derush', { keys: { ...DERUSH_DEFAULT_KEYS } })}
          className="flex items-center gap-2 bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-4 py-2 rounded-lg text-sm"
        >
          <RotateCcw size={14} /> {t('settings.derush.reset.button')}
        </button>
      </Row>
    </>
  );
}

function KeyCapture({ value, onChange }: { value: string; onChange: (key: string) => void }) {
  const t = useT();
  const [listening, setListening] = useState(false);
  return (
    <button
      onClick={() => setListening(true)}
      onKeyDown={(e) => {
        if (!listening) return;
        e.preventDefault();
        e.stopPropagation();
        if (e.key === 'Escape') { setListening(false); return; }
        if (['shift', 'control', 'alt', 'meta'].includes(e.key.toLowerCase())) return;
        onChange(e.key.toLowerCase());
        setListening(false);
      }}
      onBlur={() => setListening(false)}
      className={`min-w-[110px] px-4 py-2 rounded-lg text-sm font-mono font-bold border-2 transition-all ${
        listening
          ? 'border-[#EC4899] bg-[#EC4899]/10 text-[#EC4899] animate-pulse'
          : value
            ? 'border-[#27272A] bg-[#0F0F11] text-[#FAFAFA] hover:border-[#8B5CF6]'
            : 'border-dashed border-[#3F3F46] bg-[#0F0F11] text-[#71717A]'
      }`}
    >
      {listening ? t('settings.key.press') : value ? displayKey(value) : t('settings.key.unbound')}
    </button>
  );
}

function ExportSection({ settings, patch }: { settings: AppSettings; patch: Patch }) {
  const t = useT();
  const s = settings.export;
  return (
    <>
      <Row label={t('settings.export.codec.label')} hint={t('settings.export.codec.hint')}>
        <select value={s.codec} onChange={(e) => patch('export', { codec: e.target.value })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value="h264_nvenc">{t('settings.export.codec.opt.nvenc')}</option>
          <option value="libx264">H.264 (CPU, CRF)</option>
          <option value="libx265">H.265 / HEVC</option>
          <option value="prores_ks">ProRes 422</option>
          <option value="dnxhr">DNxHR</option>
          <option value="libvpx-vp9">VP9</option>
          <option value="libsvtav1">AV1</option>
        </select>
      </Row>
      <Row label={t('settings.export.crf.label')} hint={t('settings.export.crf.hint')}>
        <Slider value={s.crf} min={0} max={51} step={1} onChange={(v) => patch('export', { crf: v })} />
      </Row>
      <Row label={t('settings.export.resolution.label')} hint={t('settings.export.resolution.hint')}>
        <select value={s.resolution} onChange={(e) => patch('export', { resolution: e.target.value })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value="source">{t('settings.export.resolution.source')}</option>
          <option value="1080p">1080p</option>
          <option value="720p">720p</option>
          <option value="480p">480p</option>
        </select>
      </Row>
      <Row label={t('settings.export.audio.label')} hint={t('settings.export.audio.hint')}>
        <select value={s.audio} onChange={(e) => patch('export', { audio: e.target.value })} className="bg-[#0F0F11] border-2 border-[#EC4899]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#EC4899]">
          <option value="copy">{t('settings.export.audio.copy')}</option>
          <option value="encode">{t('settings.export.audio.encode')}</option>
          <option value="mute">{t('settings.export.audio.mute')}</option>
        </select>
      </Row>
      {s.audio === 'encode' && (
        <Row label={t('settings.export.aacBitrate.label')} hint={t('settings.export.aacBitrate.hint')}>
          <select value={s.audio_bitrate_kbps ?? 320} onChange={(e) => patch('export', { audio_bitrate_kbps: Number(e.target.value) })} className="bg-[#0F0F11] border-2 border-[#EC4899]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#EC4899]">
            <option value={192}>192 kbps</option>
            <option value={256}>256 kbps</option>
            <option value={320}>320 kbps</option>
          </select>
        </Row>
      )}
      <Row label={t('settings.export.naming.label')} hint={t('settings.export.naming.hint')}>
        <input type="text" value={s.naming_template} onChange={(e) => patch('export', { naming_template: e.target.value })} className="bg-[#0F0F11] border-2 border-[#27272A] rounded-lg px-4 py-2.5 text-[#FAFAFA] text-sm font-mono focus:outline-none focus:border-[#8B5CF6] w-80" />
      </Row>
      <Row label={t('settings.export.outputFolder.label')} hint={t('settings.export.outputFolder.hint')}>
        <button onClick={async () => {
          const p = await window.amvBridge?.openFileDialog({ directory: true });
          if (p && p[0]) patch('export', { output_folder: p[0] });
        }} className="bg-[#0F0F11] border-2 border-[#27272A] hover:border-[#8B5CF6]/50 rounded-lg px-4 py-2.5 text-[#A1A1AA] text-sm flex items-center gap-2 max-w-[400px] truncate">
          <FolderOpen size={14} />
          <span className="truncate">{s.output_folder || t('settings.export.outputFolder.choose')}</span>
        </button>
      </Row>
      <Row label={t('settings.export.openAfter.label')} hint={t('settings.export.openAfter.hint')}>
        <Toggle on={s.open_folder_after} onChange={(v) => patch('export', { open_folder_after: v })} />
      </Row>
    </>
  );
}

function InterfaceSection({ settings, patch }: { settings: AppSettings; patch: Patch }) {
  const { lang, setLang, t } = useI18n();
  const s = settings.interface;
  return (
    <>
      <Row label={t('settings.language.label')} hint={t('settings.language.hint')}>
        <select value={lang} onChange={(e) => setLang(e.target.value as Lang)} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value="en">English</option>
          <option value="fr">Français</option>
        </select>
      </Row>
      <Row label={t('settings.interface.theme.label')} hint={t('settings.interface.theme.hint')}>
        <select disabled value={s.theme} className="bg-[#0F0F11] border-2 border-[#27272A] rounded-lg px-5 py-2.5 text-[#71717A] opacity-60">
          <option value="dark">{t('settings.interface.theme.dark')}</option>
        </select>
      </Row>
      <Row label={t('settings.interface.thumb.label')} hint={t('settings.interface.thumb.hint')}>
        <select value={s.thumbnail_size} onChange={(e) => patch('interface', { thumbnail_size: e.target.value })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value="small">{t('common.small')}</option>
          <option value="medium">{t('common.medium')}</option>
          <option value="large">{t('common.large')}</option>
        </select>
      </Row>
      <Row label={t('settings.editor.hoverDelay.label')} hint={t('settings.interface.hoverDelay.hint')}>
        <select value={s.hover_delay_ms} onChange={(e) => patch('interface', { hover_delay_ms: Number(e.target.value) })} className="bg-[#0F0F11] border-2 border-[#EC4899]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#EC4899]">
          <option value={100}>100 ms</option>
          <option value={200}>200 ms</option>
          <option value={500}>500 ms</option>
        </select>
      </Row>
    </>
  );
}

function UpdatesSection() {
  const t = useT();
  return (
    <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
      <div className="text-[#FAFAFA] font-semibold mb-1">AMV Tools v0.2.0-alpha</div>
      <div className="text-[#71717A] text-sm">{t('settings.updates.body')}</div>
    </div>
  );
}

function AdvancedSection({ settings, setSettings }: { settings: AppSettings; setSettings: (s: AppSettings) => void }) {
  async function exportJson() {
    const path = await window.amvBridge?.saveFileDialog({
      defaultPath: 'amv-tools-settings.json',
      filters: [{ name: 'JSON', extensions: ['json'] }],
    });
    if (!path) return;
    const blob = new Blob([JSON.stringify(settings, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = path.split(/[\\/]/).pop() || 'amv-tools-settings.json';
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }

  async function importJson(file: File) {
    const text = await file.text();
    const parsed = JSON.parse(text) as AppSettings;
    const saved = await api.importSettings(parsed);
    setSettings(saved);
  }

  const ref = useRef<HTMLInputElement>(null);
  const t = useT();

  return (
    <>
      <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
        <div className="text-[#FAFAFA] font-semibold mb-2">{t('settings.advanced.json.title')}</div>
        <div className="text-[#71717A] text-sm mb-4">{t('settings.advanced.json.hint')}</div>
        <div className="flex gap-3">
          <button onClick={exportJson} className="bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-4 py-2 rounded-lg text-sm flex items-center gap-2"><Download size={14} /> {t('settings.advanced.exportJson')}</button>
          <button onClick={() => ref.current?.click()} className="bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-4 py-2 rounded-lg text-sm flex items-center gap-2"><Upload size={14} /> {t('settings.advanced.importJson')}</button>
          <input ref={ref} type="file" accept=".json" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) importJson(f); }} />
        </div>
      </div>
      <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
        <div className="text-[#FAFAFA] font-semibold mb-1">{t('settings.advanced.file')}</div>
        <div className="text-[#71717A] text-xs font-mono break-all">{`~/.AMVTools/settings.json`} (Windows: %APPDATA%\\AMVTools\\settings.json)</div>
      </div>
      <DangerZone />
    </>
  );
}

function DangerZone() {
  const t = useT();
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const supported = typeof window.amvBridge?.cleanReinstall === 'function';

  async function run() {
    if (!supported) return;
    setBusy(true);
    try {
      await window.amvBridge!.cleanReinstall!();
      // The app is being relaunched — nothing more to do.
    } catch (err) {
      setBusy(false);
      alert(t('settings.danger.alert', { message: (err as Error).message }));
    }
  }

  return (
    <div className="bg-gradient-to-br from-red-500/5 to-[#0F0F11] border-2 border-red-500/40 rounded-xl p-5">
      <div className="flex items-start gap-3 mb-3">
        <AlertTriangle size={20} className="text-red-400 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <div className="text-red-300 font-semibold">{t('settings.danger.title')}</div>
          <div className="text-[#A1A1AA] text-sm mt-1">
            {t('settings.danger.desc')}
          </div>
        </div>
      </div>
      <div className="text-xs text-[#71717A] mb-4 ml-8">
        <div className="font-semibold text-[#A1A1AA] mb-1">{t('settings.danger.deletes')}</div>
        <ul className="list-disc list-inside space-y-0.5">
          <li>{t('settings.danger.del.venv.a')}<span className="font-mono">.venv</span>{t('settings.danger.del.venv.b')}</li>
          <li>{t('settings.danger.del.db.a')}<span className="font-mono">amv_tools.db</span>{t('settings.danger.del.db.b')}</li>
          <li>{t('settings.danger.del.settings')}</li>
        </ul>
        <div className="font-semibold text-[#A1A1AA] mt-2 mb-1">{t('settings.danger.preserved')}</div>
        <ul className="list-disc list-inside space-y-0.5">
          <li>{t('settings.danger.keep.videos')}</li>
          <li>{t('settings.danger.keep.exports')}</li>
          <li>{t('settings.danger.keep.uv')}</li>
        </ul>
      </div>
      {!confirming ? (
        <button
          onClick={() => setConfirming(true)}
          disabled={!supported}
          className="ml-8 bg-red-500/20 hover:bg-red-500/30 border-2 border-red-500/40 text-red-200 px-5 py-2 rounded-lg text-sm font-semibold flex items-center gap-2 disabled:opacity-30 disabled:cursor-not-allowed"
          title={supported ? undefined : t('settings.danger.electronOnly')}
        >
          <Trash2 size={14} /> {t('settings.danger.reinstall')}
        </button>
      ) : (
        <div className="ml-8 flex items-center gap-2">
          <button
            onClick={run}
            disabled={busy}
            className="bg-red-500/30 hover:bg-red-500/40 border-2 border-red-500/60 text-red-100 px-5 py-2 rounded-lg text-sm font-semibold flex items-center gap-2 disabled:opacity-50"
          >
            {busy ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
            {busy ? t('settings.danger.wiping') : t('settings.danger.confirm')}
          </button>
          <button
            onClick={() => setConfirming(false)}
            disabled={busy}
            className="bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-5 py-2 rounded-lg text-sm"
          >
            {t('common.cancel')}
          </button>
        </div>
      )}
    </div>
  );
}
