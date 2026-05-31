import { useEffect, useRef, useState } from 'react';
import { Cpu, Zap, Target, Layers, FileVideo, HardDrive, Palette, Download, Settings2, Plus, FolderOpen, Loader2, Trash2, Upload, RefreshCw, Link, GitMerge, AlertTriangle } from 'lucide-react';
import { api } from '../api/client';
import type { AppSettings, IndexQueueItem } from '../api/types';

const sections = [
  { name: 'Indexing', icon: Cpu },
  { name: 'Models', icon: Zap },
  { name: 'Search', icon: Target },
  { name: 'Editor', icon: Layers },
  { name: 'Export', icon: FileVideo },
  { name: 'Databases', icon: HardDrive },
  { name: 'Interface', icon: Palette },
  { name: 'Updates', icon: Download },
  { name: 'Advanced', icon: Settings2 },
] as const;

type SectionName = typeof sections[number]['name'];

export default function SettingsTab({ runtimeDevice }: { runtimeDevice?: string }) {
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
                <span className="relative z-10 font-medium">{section.name}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex-1 p-8 overflow-auto relative">
        <div className="absolute top-0 right-0 w-[400px] h-[400px] bg-[#8B5CF6]/5 rounded-full blur-[100px] pointer-events-none"></div>

        <div className="relative z-10 mb-8 flex items-end justify-between">
          <div>
            <h2 className="text-3xl font-bold bg-gradient-to-r from-[#FAFAFA] to-[#A1A1AA] bg-clip-text text-transparent">{activeSection}</h2>
            <div className="h-1 w-20 bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] rounded-full mt-2"></div>
          </div>
          {saving && <div className="text-xs text-[#71717A] flex items-center gap-1"><Loader2 size={12} className="animate-spin" /> Saving</div>}
        </div>

        {error && <div className="relative z-10 mb-4 bg-red-500/10 border border-red-500/40 text-red-300 text-sm rounded-lg px-4 py-2">{error}</div>}

        {!settings ? (
          <div className="relative z-10 text-[#71717A] flex items-center gap-2"><Loader2 size={14} className="animate-spin" /> Loading settings...</div>
        ) : (
          <div className="flex flex-col gap-4 max-w-4xl relative z-10">
            {activeSection === 'Indexing' && <IndexingSection settings={settings} patch={patch} />}
            {activeSection === 'Models' && <ModelsSection settings={settings} patch={patch} runtimeDevice={runtimeDevice} />}
            {activeSection === 'Search' && <SearchSection settings={settings} patch={patch} />}
            {activeSection === 'Editor' && <EditorSection settings={settings} patch={patch} />}
            {activeSection === 'Export' && <ExportSection settings={settings} patch={patch} />}
            {activeSection === 'Databases' && <DatabasesSection />}
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

function IndexingSection({ settings, patch }: { settings: AppSettings; patch: Patch }) {
  const s = settings.indexing;
  const isCpu = s.device === 'cpu';
  return (
    <>
      <Row label="Acceleration" hint="Toggle between GPU (auto-detect best backend) and CPU. Indexing on CPU is ~20× slower but works everywhere.">
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
      <Row label="Device (advanced)" hint="Override the auto-detection — only for power users with multiple GPUs or specific backend needs.">
        <select value={s.device} onChange={(e) => patch('indexing', { device: e.target.value })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value="auto">Auto-detect (recommended)</option>
          <option value="cuda">CUDA (NVIDIA)</option>
          <option value="dml">DirectML (AMD / Intel on Windows)</option>
          <option value="xpu">Intel XPU</option>
          <option value="mps">Apple MPS</option>
          <option value="cpu">CPU</option>
        </select>
      </Row>
      <Row label="Batch size" hint="Frames processed per inference pass">
        <Slider value={s.batch_size} min={1} max={64} step={1} onChange={(v) => patch('indexing', { batch_size: v })} />
      </Row>
      <Row label="Indexing mode" hint="Trade speed for accuracy">
        <div className="flex gap-2">
          {(['fast', 'balanced', 'accurate'] as const).map((m) => (
            <button key={m} onClick={() => patch('indexing', { mode: m })} className={`px-4 py-2 rounded-lg text-sm capitalize ${s.mode === m ? 'bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white shadow-lg shadow-[#8B5CF6]/30' : 'bg-[#27272A] text-[#71717A] hover:text-[#FAFAFA]'}`}>{m}</button>
          ))}
        </div>
      </Row>
      <Row label="Sub-segmentation" hint="Split long scenes via wd-tagger embedding drift">
        <Toggle on={s.sub_segmentation} onChange={(v) => patch('indexing', { sub_segmentation: v })} />
      </Row>
      <Row label="Sub-seg threshold" hint="Higher = fewer sub-cuts (0.2 - 0.5)">
        <Slider value={Math.round(s.sub_segmentation_threshold * 100)} min={20} max={50} step={1} onChange={(v) => patch('indexing', { sub_segmentation_threshold: v / 100 })} />
      </Row>
      <Row label="Generate proxies" hint="VP9 WebM proxies for fast hover preview">
        <Toggle on={s.generate_proxies} onChange={(v) => patch('indexing', { generate_proxies: v })} />
      </Row>
      <Row label="Proxy quality" hint="Bitrate for proxy files">
        <select value={s.proxy_quality} onChange={(e) => patch('indexing', { proxy_quality: e.target.value as any })} className="bg-[#0F0F11] border-2 border-[#EC4899]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#EC4899]">
          <option value="low">Low (300 kbps)</option>
          <option value="medium">Medium (500 kbps)</option>
          <option value="high">High (1 Mbps)</option>
        </select>
      </Row>
      <Row label="Auto-skip indexed" hint="Skip videos already in the database">
        <Toggle on={s.auto_skip_indexed} onChange={(v) => patch('indexing', { auto_skip_indexed: v })} />
      </Row>
      <Row label="Scene-detect threshold" hint="ffmpeg scene-change sensitivity">
        <Slider value={Math.round(s.scene_detect_threshold * 100)} min={10} max={60} step={1} onChange={(v) => patch('indexing', { scene_detect_threshold: v / 100 })} />
      </Row>
    </>
  );
}

function ModelsSection({ settings, patch, runtimeDevice }: { settings: AppSettings; patch: Patch; runtimeDevice?: string }) {
  const s = settings.models;
  const sam2Available = runtimeDevice !== 'dml';
  return (
    <>
      <Row label="wd-tagger variant" hint="Anime tagger backbone">
        <select value={s.wd_tagger_variant} onChange={(e) => patch('models', { wd_tagger_variant: e.target.value })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value="SmilingWolf/wd-vit-tagger-v3">wd-vit-v3 (default, fast)</option>
          <option value="SmilingWolf/wd-vit-large-tagger-v3">wd-vit-large-v3 (better)</option>
          <option value="SmilingWolf/wd-eva02-large-tagger-v3">wd-eva02-large-v3 (best)</option>
        </select>
      </Row>
      <Row label="SigLIP variant" hint="Semantic embedding backbone. base is fast but can confuse similar proper nouns (Lucario vs Dracaufeu). so400m is 4× bigger and discriminates them properly — pick it if your library has lots of named entities.">
        <select value={s.siglip_variant} onChange={(e) => patch('models', { siglip_variant: e.target.value })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value="google/siglip2-base-patch16-naflex">base-naflex — 93M, ~35s/ep, ⚡ Speed (default)</option>
          <option value="google/siglip2-so400m-patch16-naflex">so400m-naflex — 400M, ~60s/ep, ⭐ Legacy quality</option>
        </select>
      </Row>
      <Row label="SigLIP patches" hint="NaFlex token budget per image. 128 is the sweet spot; bumping to 256 only adds ~5-10% accuracy on fine visual detail and costs ~50% more wallclock per episode. Lower if you want to push speed further.">
        <select value={s.siglip_max_num_patches} onChange={(e) => patch('models', { siglip_max_num_patches: Number(e.target.value) })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value={64}>64 — fastest, noticeable accuracy drop</option>
          <option value={128}>128 — Balanced (default)</option>
          <option value={192}>192</option>
          <option value={256}>256 — legacy / max quality, ~+50% time</option>
        </select>
      </Row>
      <Row label="Tag storage threshold" hint="Discard tags below this confidence at index time">
        <Slider value={Math.round(s.default_tag_threshold * 100)} min={30} max={70} step={1} onChange={(v) => patch('models', { default_tag_threshold: v / 100 })} />
      </Row>
      <Row label="VRAM idle offload" hint="Seconds before models unload from GPU">
        <select value={s.vram_idle_offload} onChange={(e) => patch('models', { vram_idle_offload: Number(e.target.value) })} className="bg-[#0F0F11] border-2 border-[#EC4899]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#EC4899]">
          <option value={0}>Off</option>
          <option value={30}>30 s</option>
          <option value={60}>60 s</option>
          <option value={300}>5 min</option>
        </select>
      </Row>
      <Row label="TensorRT optimization" hint="CUDA only. Compiles SigLIP vision passes on first use when torch-tensorrt is installed">
        <Toggle on={s.use_tensorrt} onChange={(v) => patch('models', { use_tensorrt: v })} />
      </Row>
      <Row label="Mask & alpha export" hint="Adds a 🎭 Mask button in the MiniEditor. Lets you isolate a character and export the clip as ProRes 4444 alpha .mov for compositing in Resolve/AE.">
        <Toggle on={!!s.enable_mask} onChange={(v) => patch('models', { enable_mask: v })} />
      </Row>
      <Row label="Mask engine" hint="BiRefNet = automatic foreground segmentation. SAM 2 = click-to-mask tracking when you need to pick one character among several.">
        <select
          value={sam2Available ? (s.mask_engine || 'birefnet') : 'birefnet'}
          onChange={(e) => patch('models', { mask_engine: e.target.value as 'birefnet' | 'sam2' })}
          disabled={!s.enable_mask}
          className="bg-[#0F0F11] border-2 border-[#A855F7]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#A855F7] disabled:opacity-40"
        >
          <option value="birefnet">BiRefNet — Auto, recommended for anime (default)</option>
          <option value="sam2" disabled={!sam2Available}>SAM 2 - Manual click, for multi-subject scenes</option>
        </select>
      </Row>
      <Row label="BiRefNet variant" hint="Used by Auto. general = best all-rounder. HR = 2K+ inputs, fine hair/fur, ~2x VRAM. portrait = face-tuned.">
        <select
          value={s.birefnet_variant || 'general'}
          onChange={(e) => patch('models', { birefnet_variant: e.target.value as 'general' | 'hr' | 'portrait' })}
          disabled={!s.enable_mask || s.mask_engine !== 'birefnet'}
          className="bg-[#0F0F11] border-2 border-[#A855F7]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#A855F7] disabled:opacity-40"
        >
          <option value="general">general — ~440 MB, ⭐ Default</option>
          <option value="hr">HR — high-resolution, ~2× VRAM</option>
          <option value="portrait">portrait — face-tuned</option>
        </select>
      </Row>
      <Row label="SAM 2 variant" hint="Used only when Mask engine = SAM 2. Heavier = better mask quality on hard cases. base_plus is the qual/VRAM sweet spot on 8 GB GPUs.">
        <select
          value={s.sam2_variant || 'base_plus'}
          onChange={(e) => patch('models', { sam2_variant: e.target.value as 'tiny' | 'small' | 'base_plus' | 'large' })}
          disabled={!s.enable_mask || !sam2Available || s.mask_engine !== 'sam2'}
          className="bg-[#0F0F11] border-2 border-[#A855F7]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#A855F7] disabled:opacity-40"
        >
          <option value="tiny">tiny — ~150 MB, fastest, weakest on cluttered scenes</option>
          <option value="small">small — ~190 MB</option>
          <option value="base_plus">base_plus — ~350 MB, ⭐ Sweet spot (default)</option>
          <option value="large">large — ~900 MB, best quality, ~+50% VRAM</option>
        </select>
      </Row>
      <Row label="Roto resolution" hint="Max side sent to the roto models. 1080 preserves small anime hair better than 720; 1440+ is slower and heavier.">
        <Slider
          value={s.mask_max_dim ?? 1080}
          min={720}
          max={1440}
          step={120}
          onChange={(v) => patch('models', { mask_max_dim: v })}
        />
      </Row>
      <Row label="Hard-mask shrink (px)" hint="SAM 2/manual cleanup only. Auto soft alpha skips this so hair tips and antialiased outlines stay intact. 0 = preserve detail, 2+ = aggressive halo removal.">
        <Slider value={s.mask_shrink_px ?? 0} min={0} max={5} step={1} onChange={(v) => patch('models', { mask_shrink_px: v })} />
      </Row>
      <Row label="Suppress BG-colored halo" hint="SAM 2/manual cleanup only. Leave off for soft alpha: it can mistake dark hair or shaded line art for background and erase detail.">
        <Toggle on={s.mask_bg_suppress_enabled ?? false} onChange={(v) => patch('models', { mask_bg_suppress_enabled: v })} />
      </Row>
      <Row label="BG suppress threshold" hint="Hard-mask cleanup threshold. Lower (10-15) preserves detail, higher (40-50) removes more halo but can eat real line art. Ignored for Auto soft alpha.">
        <Slider
          value={s.mask_bg_suppress_threshold ?? 25}
          min={10}
          max={50}
          step={1}
          onChange={(v) => patch('models', { mask_bg_suppress_threshold: v })}
        />
      </Row>
      <Row label="Soft-alpha BG cleanup" hint="Auto cleanup. Removes background-coloured pixels near hair gaps while keeping soft alpha on the real strand edges.">
        <Toggle on={s.mask_soft_bg_suppress_enabled ?? false} onChange={(v) => patch('models', { mask_soft_bg_suppress_enabled: v })} />
      </Row>
      <Row label="Soft BG threshold" hint="Lower preserves more edge detail, higher removes more leaked background between hair spikes.">
        <Slider
          value={s.mask_soft_bg_suppress_threshold ?? 25}
          min={8}
          max={40}
          step={1}
          onChange={(v) => patch('models', { mask_soft_bg_suppress_threshold: v })}
        />
      </Row>
      <Row label="Soft cleanup band" hint="Width in pixels around the alpha boundary to inspect. Higher reaches deeper into concave hair gaps.">
        <Slider
          value={s.mask_soft_bg_suppress_edge_px ?? 16}
          min={2}
          max={24}
          step={1}
          onChange={(v) => patch('models', { mask_soft_bg_suppress_edge_px: v })}
        />
      </Row>
      <Row label="Soft alpha shrink" hint="Contracts Auto alpha before export. 0 preserves the model output; 1+ can remove haze but may eat hair/body parts.">
        <Slider
          value={s.mask_soft_shrink_px ?? 0}
          min={0}
          max={3}
          step={1}
          onChange={(v) => patch('models', { mask_soft_shrink_px: v })}
        />
      </Row>
      <Row label="Soft alpha low cut" hint="Pixels below this alpha are pushed toward transparent. Keep at 0 to avoid deleting weak hair/body alpha.">
        <Slider
          value={Math.round((s.mask_soft_alpha_black ?? 0.0) * 100)}
          min={0}
          max={25}
          step={1}
          onChange={(v) => patch('models', { mask_soft_alpha_black: v / 100 })}
        />
      </Row>
      <Row label="RGB decontaminate export" hint="Slow high-quality edge color cleanup for soft alpha. Keep off for fast exports; enable only when edge color spill is worse than export time.">
        <Toggle on={s.mask_rgb_decontaminate_enabled ?? false} onChange={(v) => patch('models', { mask_rgb_decontaminate_enabled: v })} />
      </Row>
      <Row label="Hugging Face token" hint="Optional — for private models or to avoid rate limits">
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
  const s = settings.search;
  return (
    <>
      <Row label="Default similarity threshold" hint="Below this, results are hidden. SigLIP 2 NaFlex scores are typically 0.05-0.20 so 10 is a good default.">
        <Slider value={Math.round(s.threshold * 100)} min={5} max={50} step={1} onChange={(v) => patch('search', { threshold: v / 100 })} />
      </Row>
      <Row label="Max results" hint="Cap on returned scenes per query">
        <select value={s.max_results} onChange={(e) => patch('search', { max_results: Number(e.target.value) })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value={50}>50</option>
          <option value={200}>200</option>
          <option value={1000}>1000</option>
          <option value={10000}>Unlimited</option>
        </select>
      </Row>
      <Row label="wd-tagger boost" hint="Score bonus per matched tag token">
        <Slider value={Math.round(s.tag_boost * 100)} min={0} max={30} step={1} onChange={(v) => patch('search', { tag_boost: v / 100 })} />
      </Row>
    </>
  );
}

function EditorSection({ settings, patch }: { settings: AppSettings; patch: Patch }) {
  const s = settings.interface;
  return (
    <>
      <Row label="Hover preview delay" hint="Time before proxy starts playing on hover">
        <select value={s.hover_delay_ms} onChange={(e) => patch('interface', { hover_delay_ms: Number(e.target.value) })} className="bg-[#0F0F11] border-2 border-[#EC4899]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#EC4899]">
          <option value={100}>100 ms</option>
          <option value={200}>200 ms</option>
          <option value={500}>500 ms</option>
        </select>
      </Row>
      <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
        <div className="text-[#FAFAFA] font-semibold mb-2">Keyboard shortcuts (mini-editor)</div>
        <div className="grid grid-cols-2 gap-2 text-sm font-mono">
          <div className="text-[#71717A]">J / K / L</div><div className="text-[#A1A1AA]">Scrub back · pause · scrub forward</div>
          <div className="text-[#71717A]">, / .</div><div className="text-[#A1A1AA]">Step 1 frame back / forward</div>
          <div className="text-[#71717A]">I / O</div><div className="text-[#A1A1AA]">Set IN / OUT to current frame</div>
          <div className="text-[#71717A]">← / →</div><div className="text-[#A1A1AA]">Prev / next scene in the current list</div>
          <div className="text-[#71717A]">Space</div><div className="text-[#A1A1AA]">Play / pause</div>
        </div>
      </div>
    </>
  );
}

function ExportSection({ settings, patch }: { settings: AppSettings; patch: Patch }) {
  const s = settings.export;
  return (
    <>
      <Row label="Codec" hint="Video codec for exported clips">
        <select value={s.codec} onChange={(e) => patch('export', { codec: e.target.value })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value="libx264">H.264</option>
          <option value="libx265">H.265 / HEVC</option>
          <option value="prores_ks">ProRes 422</option>
          <option value="dnxhr">DNxHR</option>
          <option value="libvpx-vp9">VP9</option>
          <option value="libsvtav1">AV1</option>
        </select>
      </Row>
      <Row label="CRF" hint="Lower = better quality, larger files">
        <Slider value={s.crf} min={0} max={51} step={1} onChange={(v) => patch('export', { crf: v })} />
      </Row>
      <Row label="Resolution" hint="Output resolution">
        <select value={s.resolution} onChange={(e) => patch('export', { resolution: e.target.value })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value="source">Source</option>
          <option value="1080p">1080p</option>
          <option value="720p">720p</option>
          <option value="480p">480p</option>
        </select>
      </Row>
      <Row label="Audio" hint="How to handle audio track">
        <select value={s.audio} onChange={(e) => patch('export', { audio: e.target.value })} className="bg-[#0F0F11] border-2 border-[#EC4899]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#EC4899]">
          <option value="copy">Copy (fast)</option>
          <option value="encode">Re-encode AAC</option>
          <option value="mute">Mute</option>
        </select>
      </Row>
      <Row label="Naming template" hint="Variables: {anime} {episode} {scene_id} {tags}">
        <input type="text" value={s.naming_template} onChange={(e) => patch('export', { naming_template: e.target.value })} className="bg-[#0F0F11] border-2 border-[#27272A] rounded-lg px-4 py-2.5 text-[#FAFAFA] text-sm font-mono focus:outline-none focus:border-[#8B5CF6] w-80" />
      </Row>
      <Row label="Output folder" hint="Where exported clips are saved">
        <button onClick={async () => {
          const p = await window.amvBridge?.openFileDialog({ directory: true });
          if (p && p[0]) patch('export', { output_folder: p[0] });
        }} className="bg-[#0F0F11] border-2 border-[#27272A] hover:border-[#8B5CF6]/50 rounded-lg px-4 py-2.5 text-[#A1A1AA] text-sm flex items-center gap-2 max-w-[400px] truncate">
          <FolderOpen size={14} />
          <span className="truncate">{s.output_folder || 'Choose folder...'}</span>
        </button>
      </Row>
      <Row label="Open folder after export" hint="Reveal exported file in the OS file manager">
        <Toggle on={s.open_folder_after} onChange={(v) => patch('export', { open_folder_after: v })} />
      </Row>
    </>
  );
}

function InterfaceSection({ settings, patch }: { settings: AppSettings; patch: Patch }) {
  const s = settings.interface;
  return (
    <>
      <Row label="Theme" hint="Light theme planned for v0.5">
        <select disabled value={s.theme} className="bg-[#0F0F11] border-2 border-[#27272A] rounded-lg px-5 py-2.5 text-[#71717A] opacity-60">
          <option value="dark">Dark</option>
        </select>
      </Row>
      <Row label="Thumbnail size" hint="Gallery thumbnail size">
        <select value={s.thumbnail_size} onChange={(e) => patch('interface', { thumbnail_size: e.target.value })} className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]">
          <option value="small">Small</option>
          <option value="medium">Medium</option>
          <option value="large">Large</option>
        </select>
      </Row>
      <Row label="Hover preview delay" hint="Time before video proxy starts">
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
  return (
    <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
      <div className="text-[#FAFAFA] font-semibold mb-1">AMV Tools v0.1.0-alpha</div>
      <div className="text-[#71717A] text-sm">Auto-update will ship with v1.0 via electron-updater.</div>
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

  return (
    <>
      <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
        <div className="text-[#FAFAFA] font-semibold mb-2">Settings JSON</div>
        <div className="text-[#71717A] text-sm mb-4">Export/import full settings. Useful for backups or sharing presets between machines.</div>
        <div className="flex gap-3">
          <button onClick={exportJson} className="bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-4 py-2 rounded-lg text-sm flex items-center gap-2"><Download size={14} /> Export JSON</button>
          <button onClick={() => ref.current?.click()} className="bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-4 py-2 rounded-lg text-sm flex items-center gap-2"><Upload size={14} /> Import JSON</button>
          <input ref={ref} type="file" accept=".json" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) importJson(f); }} />
        </div>
      </div>
      <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
        <div className="text-[#FAFAFA] font-semibold mb-1">Settings file</div>
        <div className="text-[#71717A] text-xs font-mono break-all">{`~/.AMVTools/settings.json`} (Windows: %APPDATA%\\AMVTools\\settings.json)</div>
      </div>
      <DangerZone />
    </>
  );
}

function DangerZone() {
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
      alert(`Clean install failed: ${(err as Error).message}`);
    }
  }

  return (
    <div className="bg-gradient-to-br from-red-500/5 to-[#0F0F11] border-2 border-red-500/40 rounded-xl p-5">
      <div className="flex items-start gap-3 mb-3">
        <AlertTriangle size={20} className="text-red-400 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <div className="text-red-300 font-semibold">Danger zone</div>
          <div className="text-[#A1A1AA] text-sm mt-1">
            Reset the Python environment and the local index, then relaunch the
            onboarding wizard. Use this if the GPU backend is misconfigured or if
            the venv is corrupted.
          </div>
        </div>
      </div>
      <div className="text-xs text-[#71717A] mb-4 ml-8">
        <div className="font-semibold text-[#A1A1AA] mb-1">This will delete:</div>
        <ul className="list-disc list-inside space-y-0.5">
          <li>The Python virtual environment (<span className="font-mono">.venv</span>) — full re-download of PyTorch &amp; co. (~1-3 GB) on next launch</li>
          <li>The scene index database (<span className="font-mono">amv_tools.db</span>) — all indexed scenes are lost</li>
          <li>Settings and thumbnails — your preferences will be reset</li>
        </ul>
        <div className="font-semibold text-[#A1A1AA] mt-2 mb-1">Preserved:</div>
        <ul className="list-disc list-inside space-y-0.5">
          <li>Your video files (we never touch them)</li>
          <li>Exported clips and proxy videos</li>
          <li>The uv binary itself (so reinstall can run)</li>
        </ul>
      </div>
      {!confirming ? (
        <button
          onClick={() => setConfirming(true)}
          disabled={!supported}
          className="ml-8 bg-red-500/20 hover:bg-red-500/30 border-2 border-red-500/40 text-red-200 px-5 py-2 rounded-lg text-sm font-semibold flex items-center gap-2 disabled:opacity-30 disabled:cursor-not-allowed"
          title={supported ? undefined : 'Only available inside the Electron app'}
        >
          <Trash2 size={14} /> Reinstall from scratch
        </button>
      ) : (
        <div className="ml-8 flex items-center gap-2">
          <button
            onClick={run}
            disabled={busy}
            className="bg-red-500/30 hover:bg-red-500/40 border-2 border-red-500/60 text-red-100 px-5 py-2 rounded-lg text-sm font-semibold flex items-center gap-2 disabled:opacity-50"
          >
            {busy ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
            {busy ? 'Wiping and relaunching…' : 'Yes, wipe everything'}
          </button>
          <button
            onClick={() => setConfirming(false)}
            disabled={busy}
            className="bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-5 py-2 rounded-lg text-sm"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}

function DatabasesSection() {
  const [databases, setDatabases] = useState<{ path: string; scenes: number; videos: number; size_kb: number; primary: boolean }[]>([]);
  const [queue, setQueue] = useState<IndexQueueItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  async function refresh() {
    const [dbs, q] = await Promise.all([api.listDatabases(), api.queue()]);
    setDatabases(dbs.databases);
    setQueue(q.items);
  }

  useEffect(() => { refresh().catch(() => {}); }, []);

  async function addFolder() {
    const paths = await window.amvBridge?.openFileDialog({ directory: true, multi: true });
    if (!paths || paths.length === 0) return;
    await api.enqueue(paths.map((p) => ({ path: p, recursive: true })));
    refresh();
  }

  async function startIndex(phases: ('tag' | 'embed')[]) {
    setBusy(true);
    try { await api.startIndexing(phases); }
    finally { setBusy(false); refresh(); }
  }

  async function addDb() {
    const paths = await window.amvBridge?.openFileDialog({
      directory: false,
      filters: [{ name: 'SQLite DB', extensions: ['db', 'sqlite', 'sqlite3'] }],
    });
    if (!paths || !paths[0]) return;
    await api.addDatabase(paths[0]);
    refresh();
  }

  async function newDb() {
    const path = await window.amvBridge?.saveFileDialog({
      defaultPath: 'amv-tools.db',
      filters: [{ name: 'SQLite DB', extensions: ['db'] }],
    });
    if (!path) return;
    await api.addDatabase(path);
    await api.setPrimaryDatabase(path);
    refresh();
  }

  const onDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files) as (File & { path?: string })[];
    const paths = files.map((f) => f.path).filter(Boolean) as string[];
    if (paths.length === 0) return;
    await api.enqueue(paths.map((p) => ({ path: p, recursive: true })));
    refresh();
  };

  return (
    <>
      <div className={`bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 ${dragOver ? 'border-[#8B5CF6]' : 'border-[#27272A]'} rounded-xl p-5 transition-colors`}
           onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
           onDragLeave={() => setDragOver(false)}
           onDrop={onDrop}>
        <div className="flex items-center justify-between mb-4">
          <div>
            <span className="text-[#FAFAFA] font-semibold">Index queue</span>
            <div className="text-[#71717A] text-sm">{queue.length} pending · drop folders here</div>
          </div>
          <div className="flex gap-2 flex-wrap">
            <button onClick={addFolder} className="flex items-center gap-2 bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-4 py-2 rounded-lg text-sm">
              <Plus size={14} /> Add folder
            </button>
            <button
              disabled={busy || queue.length === 0}
              onClick={() => startIndex(['tag', 'embed'])}
              title="Tags + embeddings (sequential, ~4 GB VRAM peak)"
              className="bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-[#FAFAFA] px-4 py-2 rounded-lg text-sm font-semibold disabled:opacity-30 flex items-center gap-2"
            >
              {busy ? <Loader2 size={14} className="animate-spin" /> : <><Zap size={14} /> Index full</>}
            </button>
            <button
              disabled={busy || queue.length === 0}
              onClick={() => startIndex(['tag'])}
              title="wd-tagger only — fast browsing by Danbooru tags"
              className="bg-[#0F0F11] border-2 border-[#8B5CF6]/40 hover:border-[#8B5CF6] text-[#FAFAFA] px-4 py-2 rounded-lg text-sm font-semibold disabled:opacity-30 flex items-center gap-2"
            >
              {busy ? <Loader2 size={14} className="animate-spin" /> : <><Target size={14} /> Tags only</>}
            </button>
            <button
              disabled={busy || queue.length === 0}
              onClick={() => startIndex(['embed'])}
              title="SigLIP only — semantic text/image search, no tag browsing"
              className="bg-[#0F0F11] border-2 border-[#EC4899]/40 hover:border-[#EC4899] text-[#FAFAFA] px-4 py-2 rounded-lg text-sm font-semibold disabled:opacity-30 flex items-center gap-2"
            >
              {busy ? <Loader2 size={14} className="animate-spin" /> : <><Layers size={14} /> Embeddings only</>}
            </button>
          </div>
        </div>
        {queue.length === 0 ? (
          <div className="text-[#71717A] text-sm text-center py-8 border-2 border-dashed border-[#27272A] rounded-lg">
            Drop folders or files here, or click <span className="text-[#FAFAFA]">Add folder</span>.
          </div>
        ) : (
          <div className="space-y-1 max-h-60 overflow-auto">
            {queue.map((q) => (
              <div key={q.id} className="text-xs font-mono text-[#A1A1AA] flex items-center gap-2 py-1 px-2 hover:bg-[#27272A]/50 rounded">
                <span className="text-[#8B5CF6]">{q.is_directory ? '📁' : '🎞'}</span>
                <span className="truncate flex-1">{q.path}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-[#FAFAFA] font-semibold">Active databases</div>
            <div className="text-[#71717A] text-sm">Click a row to set it as the primary indexing target.</div>
          </div>
          <div className="flex gap-2">
            <button onClick={addDb} className="bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-4 py-2 rounded-lg text-sm flex items-center gap-2"><FolderOpen size={14} /> Open existing</button>
            <button onClick={newDb} className="bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-4 py-2 rounded-lg text-sm flex items-center gap-2"><Plus size={14} /> Create new</button>
          </div>
        </div>
        {databases.length === 0 ? (
          <div className="text-[#71717A] text-sm">No databases configured yet.</div>
        ) : (
          <div className="space-y-2">
            {databases.map((db) => (
              <DbRow key={db.path} db={db} onChange={refresh} />
            ))}
          </div>
        )}
      </div>
    </>
  );
}

function DbRow({ db, onChange }: { db: { path: string; scenes: number; videos: number; size_kb: number; primary: boolean }; onChange: () => Promise<void> | void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [maintMsg, setMaintMsg] = useState<string | null>(null);

  async function setPrimary() {
    await api.setPrimaryDatabase(db.path);
    onChange();
  }

  async function cleanup() {
    setBusy('cleanup'); setMaintMsg(null);
    try {
      const r = await api.cleanupDatabase(db.path);
      setMaintMsg(`Removed ${r.removed} orphaned video(s)`);
      onChange();
    } catch (e) {
      setMaintMsg(`Error: ${(e as Error).message}`);
    } finally { setBusy(null); }
  }

  async function verify() {
    setBusy('verify'); setMaintMsg(null);
    try {
      const r = await api.verifyDatabase(db.path);
      setMaintMsg(r.missing.length === 0 ? 'All videos present.' : `${r.missing.length} missing video(s). Open Verify panel to relink.`);
    } catch (e) {
      setMaintMsg(`Error: ${(e as Error).message}`);
    } finally { setBusy(null); }
  }

  async function remove() {
    if (!confirm(`Remove ${db.path} from active list? Files on disk are not deleted.`)) return;
    await api.removeDatabase(db.path);
    onChange();
  }

  return (
    <div className={`p-3 rounded-lg border-2 transition-all ${db.primary ? 'border-[#8B5CF6] bg-[#8B5CF6]/10' : 'border-[#27272A] hover:border-[#3f3f46]'}`}>
      <div className="flex items-center justify-between gap-3">
        <button onClick={setPrimary} className="flex-1 min-w-0 text-left flex items-center gap-3">
          <span className="font-mono text-sm truncate flex-1">{db.path}</span>
          <div className="text-xs text-[#71717A] font-mono whitespace-nowrap">{db.videos} videos · {db.scenes} scenes · {(db.size_kb / 1024).toFixed(1)} MB</div>
          {db.primary && <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#8B5CF6]/20 text-[#8B5CF6] font-bold uppercase">Primary</span>}
        </button>
        <div className="flex items-center gap-1">
          <button onClick={verify} disabled={!!busy} title="Verify missing files" className="p-1.5 text-[#71717A] hover:text-[#8B5CF6] hover:bg-[#27272A] rounded">
            {busy === 'verify' ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
          </button>
          <button onClick={cleanup} disabled={!!busy} title="Cleanup orphans" className="p-1.5 text-[#71717A] hover:text-[#EC4899] hover:bg-[#27272A] rounded">
            {busy === 'cleanup' ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
          </button>
          <button onClick={remove} title="Remove from active list" className="p-1.5 text-[#71717A] hover:text-red-400 hover:bg-[#27272A] rounded">
            <Trash2 size={14} />
          </button>
        </div>
      </div>
      {maintMsg && <div className="text-xs text-[#A1A1AA] mt-2 font-mono">{maintMsg}</div>}
    </div>
  );
}
