import { useCallback, useEffect, useState } from 'react';
import { ChevronDown, ChevronRight, Loader2, Play, SlidersHorizontal, Square, Star, X } from 'lucide-react';
import { api } from '../api/client';
import type { AnalysisStatus, ColorBucket, MotionType, SearchFacets, SearchFilters } from '../api/types';
import { useT } from '../i18n';
import { COLOR_SWATCHES, MOTION_TYPES, countActiveFilters } from '../searchplus';
import { MotionIcon } from './SceneBadges';

interface Props {
  filters: SearchFilters;
  onChange: (f: SearchFilters) => void;
  onApply: () => void;
  open: boolean;
  onToggle: () => void;
}

function toggle<T>(list: T[] | undefined, v: T): T[] {
  const cur = list ?? [];
  return cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v];
}

function numOrNull(s: string, scale = 1): number | null {
  if (s.trim() === '') return null;
  const n = Number(s.replace(',', '.'));
  return Number.isFinite(n) ? n * scale : null;
}

/** Collapsible combined-filter panel for SearchTab. */
export default function SearchFiltersPanel({ filters, onChange, onApply, open, onToggle }: Props) {
  const t = useT();
  const [facets, setFacets] = useState<SearchFacets | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisStatus | null>(null);

  const loadFacets = useCallback(() => {
    api.searchFacets().then((f) => { setFacets(f); setAnalysis({ ...f.analysis, coverage: f.coverage }); }).catch(() => {});
  }, []);

  useEffect(() => { if (open) loadFacets(); }, [open, loadFacets]);

  // Poll while the background analyzer runs so the coverage line moves.
  useEffect(() => {
    if (!analysis?.running) return;
    const id = setInterval(() => {
      api.analysisStatus().then((s) => {
        setAnalysis(s);
        if (!s.running) loadFacets();
      }).catch(() => {});
    }, 2000);
    return () => clearInterval(id);
  }, [analysis?.running, loadFacets]);

  const set = (patch: Partial<SearchFilters>) => onChange({ ...filters, ...patch });
  const active = countActiveFilters(filters);
  const cov = analysis?.coverage ?? facets?.coverage;

  return (
    <div className="w-[750px] mt-3 bg-[#18181B]/50 backdrop-blur-sm rounded-xl border border-[#27272A]">
      <button type="button" onClick={onToggle} className="w-full flex items-center gap-2 px-4 py-3 text-sm text-[#A1A1AA] hover:text-[#FAFAFA]">
        <SlidersHorizontal size={15} className="text-[#8B5CF6]" />
        <span className="uppercase tracking-wider font-medium">{t('sp.filters.title')}</span>
        {active > 0 && (
          <span className="bg-[#8B5CF6]/20 text-[#8B5CF6] text-xs px-2 py-0.5 rounded-full font-bold">{t('sp.filters.active', { count: active })}</span>
        )}
        <span className="flex-1" />
        {active > 0 && (
          <span role="button" onClick={(e) => { e.stopPropagation(); onChange({}); }} className="text-xs text-[#71717A] hover:text-[#EC4899] flex items-center gap-1 mr-2">
            <X size={12} /> {t('sp.filters.reset')}
          </span>
        )}
        {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
      </button>

      {open && (
        <div className="px-4 pb-4 flex flex-col gap-4 text-sm">
          {/* characters */}
          <Section label={t('sp.filters.characters')}>
            {facets && facets.characters.length === 0 && <span className="text-xs text-[#71717A]">{t('sp.filters.charactersNone')}</span>}
            <div className="flex flex-wrap gap-1.5 items-center">
              {facets?.characters.map((c) => (
                <Pill key={c.id} active={!!filters.characters?.includes(c.id)} onClick={() => set({ characters: toggle(filters.characters, c.id) })}>
                  {c.name}
                  {c.group_name && <span className="opacity-60 ml-1 text-[10px]">{c.group_name}</span>}
                </Pill>
              ))}
              {(filters.characters?.length ?? 0) > 1 && (
                <select value={filters.character_mode ?? 'any'} onChange={(e) => set({ character_mode: e.target.value as 'any' | 'all' })}
                        className="bg-[#0F0F11] border border-[#27272A] rounded-md px-2 py-1 text-xs text-[#FAFAFA]">
                  <option value="any">{t('sp.filters.charMode.any')}</option>
                  <option value="all">{t('sp.filters.charMode.all')}</option>
                </select>
              )}
            </div>
          </Section>

          {/* motion */}
          <Section label={t('sp.filters.motion')}>
            <div className="flex flex-wrap gap-1.5">
              {MOTION_TYPES.map((m: MotionType) => {
                const count = facets?.motion_types.find((x) => x.id === m)?.count;
                return (
                  <Pill key={m} active={!!filters.motion?.includes(m)} onClick={() => set({ motion: toggle(filters.motion, m) })}>
                    <MotionIcon type={m} size={12} />
                    <span className="ml-1">{t(`sp.motion.${m}`)}</span>
                    {count != null && <span className="ml-1 opacity-50 text-[10px]">{count.toLocaleString()}</span>}
                  </Pill>
                );
              })}
            </div>
          </Section>

          <div className="grid grid-cols-2 gap-4">
            <Section label={t('sp.filters.intensity')}>
              <RangePair
                min={filters.intensity_min} max={filters.intensity_max}
                onChange={(lo, hi) => set({ intensity_min: lo, intensity_max: hi })}
              />
            </Section>
            <Section label={t('sp.filters.duration')}>
              <div className="flex items-center gap-2">
                <NumBox placeholder={t('sp.filters.min')} value={filters.duration_min_ms} onChange={(v) => set({ duration_min_ms: v })} />
                <span className="text-[#71717A]">–</span>
                <NumBox placeholder={t('sp.filters.max')} value={filters.duration_max_ms} onChange={(v) => set({ duration_max_ms: v })} />
              </div>
            </Section>
          </div>

          <Section label={t('sp.filters.color')}>
            <div className="flex flex-wrap gap-1.5">
              {(Object.keys(COLOR_SWATCHES) as ColorBucket[]).map((c) => {
                const on = !!filters.colors?.includes(c);
                return (
                  <button key={c} type="button" title={t(`sp.color.${c}`)} onClick={() => set({ colors: toggle(filters.colors, c) })}
                          className={`w-7 h-7 rounded-full border-2 transition-all ${on ? 'border-[#FAFAFA] scale-110 shadow-lg' : 'border-[#27272A] hover:border-[#71717A]'}`}
                          style={{ background: COLOR_SWATCHES[c] }} />
                );
              })}
            </div>
          </Section>

          <div className="grid grid-cols-2 gap-4">
            <Section label={t('sp.filters.groups')}>
              <div className="flex flex-wrap gap-1.5">
                {facets?.groups.map((g) => (
                  <Pill key={g.name} active={!!filters.groups?.includes(g.name)} onClick={() => set({ groups: toggle(filters.groups, g.name) })}>
                    {g.label ?? t('sp.filters.noGroup')}
                    <span className="ml-1 opacity-50 text-[10px]">{g.videos}</span>
                  </Pill>
                ))}
              </div>
            </Section>
            <Section label=" ">
              <Pill active={!!filters.favorites} onClick={() => set({ favorites: !filters.favorites })}>
                <Star size={12} fill={filters.favorites ? 'currentColor' : 'none'} />
                <span className="ml-1">{t('sp.filters.favorites')}</span>
              </Pill>
            </Section>
          </div>

          {/* background analysis coverage */}
          {cov && (
            <div className="flex items-center gap-3 text-xs text-[#71717A] border-t border-[#27272A] pt-3">
              {analysis?.running ? (
                <>
                  <Loader2 size={12} className="animate-spin text-[#8B5CF6]" />
                  <span className="truncate">{t('sp.analysis.running', { video: analysis.video ?? '…', percent: analysis.percent })}</span>
                  <button type="button" onClick={() => api.stopAnalysis().then(() => setTimeout(loadFacets, 500))}
                          className="flex items-center gap-1 text-[#A1A1AA] hover:text-red-400"><Square size={10} fill="currentColor" />{t('sp.analysis.stop')}</button>
                </>
              ) : (
                <>
                  <span>{t('sp.analysis.coverage', { analyzed: cov.analyzed.toLocaleString(), total: cov.total.toLocaleString() })}</span>
                  {cov.pending > 0 && (
                    <button type="button" onClick={() => api.startAnalysis().then((s) => setAnalysis({ ...s, coverage: cov })).catch(() => {})}
                            className="flex items-center gap-1 text-[#8B5CF6] hover:text-[#FAFAFA]"><Play size={10} fill="currentColor" />{t('sp.analysis.start')}</button>
                  )}
                  {(analysis?.skipped_videos ?? 0) > 0 && <span className="text-amber-300/80">{t('sp.analysis.skipped', { count: analysis!.skipped_videos })}</span>}
                </>
              )}
              <span className="flex-1" />
              {cov.pending > 0 && <span className="italic hidden lg:inline">{t('sp.filters.needsAnalysis')}</span>}
            </div>
          )}

          <div className="flex items-center gap-3">
            <button type="button" onClick={onApply}
                    className="bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-[#FAFAFA] px-5 py-2 rounded-lg text-sm font-semibold hover:shadow-lg hover:shadow-[#8B5CF6]/40 transition-all">
              {t('sp.filters.apply')}
            </button>
            <span className="text-xs text-[#71717A]">{t('sp.filters.applyHint')}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[#71717A] text-[11px] uppercase tracking-wider font-medium h-4">{label}</span>
      {children}
    </div>
  );
}

function Pill({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button type="button" onClick={onClick}
            className={`flex items-center px-2.5 py-1 rounded-full text-xs border transition-all ${
              active ? 'bg-[#8B5CF6]/25 border-[#8B5CF6] text-[#FAFAFA]' : 'bg-[#0F0F11] border-[#27272A] text-[#A1A1AA] hover:border-[#8B5CF6]/50 hover:text-[#FAFAFA]'
            }`}>
      {children}
    </button>
  );
}

function NumBox({ value, onChange, placeholder }: { value: number | null | undefined; onChange: (ms: number | null) => void; placeholder: string }) {
  const [text, setText] = useState(value != null ? String(value / 1000) : '');
  useEffect(() => { setText(value != null ? String(value / 1000) : ''); }, [value]);
  return (
    <input value={text} placeholder={placeholder} inputMode="decimal"
           onChange={(e) => setText(e.target.value)}
           onBlur={() => onChange(numOrNull(text, 1000))}
           onKeyDown={(e) => { if (e.key === 'Enter') onChange(numOrNull(text, 1000)); }}
           className="w-20 bg-[#0F0F11] border border-[#27272A] rounded-md px-2 py-1 text-xs text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]" />
  );
}

/** Two sliders (0-100 %) for an optional [min, max] interval. */
function RangePair({ min, max, onChange }: { min?: number | null; max?: number | null; onChange: (lo: number | null, hi: number | null) => void }) {
  const lo = Math.round((min ?? 0) * 100);
  const hi = Math.round((max ?? 1) * 100);
  return (
    <div className="flex items-center gap-2">
      <input type="range" min={0} max={100} value={lo} className="flex-1 accent-[#8B5CF6]"
             onChange={(e) => { const v = Number(e.target.value); onChange(v === 0 ? null : v / 100, Math.max(v, hi) === 100 ? null : Math.max(v, hi) / 100); }} />
      <input type="range" min={0} max={100} value={hi} className="flex-1 accent-[#EC4899]"
             onChange={(e) => { const v = Number(e.target.value); onChange(Math.min(v, lo) === 0 ? null : Math.min(v, lo) / 100, v === 100 ? null : v / 100); }} />
      <span className="text-[#8B5CF6] text-xs font-mono w-16 text-right">{lo}–{hi}%</span>
    </div>
  );
}
