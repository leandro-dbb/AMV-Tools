// Compact "what you want" panel: series, characters, cut rhythm, reuse,
// a global wish and per-passage keywords.
import { SlidersHorizontal, Tag, X } from 'lucide-react';
import type { MusicSection } from '../api/music';
import type { Character, SearchFacets } from '../api/types';
import { useT } from '../i18n';
import type { StoredConstraints } from './musicUtils';
import { formatTime, hexToRgba, sectionColor, sectionLabelKey } from './musicUtils';

interface Props {
  value: StoredConstraints;
  onChange: (v: StoredConstraints) => void;
  facets: SearchFacets | null;
  sections: MusicSection[];
  selectedSection: number | null;
  onSelectSection: (index: number | null) => void;
}

function toggle<T>(list: T[], v: T): T[] {
  return list.includes(v) ? list.filter((x) => x !== v) : [...list, v];
}

export default function MusicConstraints({ value, onChange, facets, sections, selectedSection, onSelectSection }: Props) {
  const t = useT();
  const set = (patch: Partial<StoredConstraints>) => onChange({ ...value, ...patch });
  const groups = facets?.groups ?? [];
  const characters: Character[] = facets?.characters ?? [];
  const section = selectedSection != null ? sections.find((s) => s.index === selectedSection) ?? null : null;
  const keyed = sections.filter((s) => value.sectionQueries[String(s.index)]?.trim());

  const setSectionQuery = (index: number, q: string) => {
    const next = { ...value.sectionQueries };
    if (q) next[String(index)] = q; else delete next[String(index)];
    set({ sectionQueries: next });
  };

  return (
    <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-4 flex flex-col gap-4 text-sm">
      <div className="flex items-center gap-2 text-[#A1A1AA]">
        <SlidersHorizontal size={15} className="text-[#8B5CF6]" />
        <span className="uppercase tracking-wider text-xs font-medium">{t('music.constraints.title')}</span>
      </div>

      {/* series */}
      <Field label={t('music.constraints.series')}>
        {facets && groups.length === 0 ? (
          <span className="text-xs text-[#71717A]">{t('music.constraints.seriesNone')}</span>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            <Pill active={value.groups.length === 0} onClick={() => set({ groups: [] })}>{t('music.constraints.seriesAll')}</Pill>
            {groups.map((g) => (
              <Pill key={g.name} active={value.groups.includes(g.name)} onClick={() => set({ groups: toggle(value.groups, g.name) })}>
                {g.label || g.name}
                <span className="opacity-50 ml-1 text-[10px]">{g.videos}</span>
              </Pill>
            ))}
          </div>
        )}
      </Field>

      {/* characters */}
      <Field label={t('music.constraints.characters')}>
        {facets && characters.length === 0 ? (
          <span className="text-xs text-[#71717A]">{t('music.constraints.charactersNone')}</span>
        ) : (
          <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto">
            <Pill active={value.characters.length === 0} onClick={() => set({ characters: [] })}>{t('music.constraints.charactersAny')}</Pill>
            {characters.map((c) => (
              <Pill key={c.id} active={value.characters.includes(c.id)} onClick={() => set({ characters: toggle(value.characters, c.id) })}>
                {c.name}
                {c.group_name && <span className="opacity-50 ml-1 text-[10px]">{c.group_name}</span>}
              </Pill>
            ))}
          </div>
        )}
      </Field>

      {/* density */}
      <Field label={t('music.constraints.density')}>
        <div className="flex items-center gap-3">
          <span className="text-xs text-[#A1A1AA] w-14 text-right">{t('music.constraints.densityCalm')}</span>
          <input
            type="range" min={0} max={1} step={0.05}
            value={value.density}
            onChange={(e) => set({ density: Number(e.target.value) })}
            className="flex-1 accent-[#EC4899]"
          />
          <span className="text-xs text-[#A1A1AA] w-14">{t('music.constraints.densityNervous')}</span>
        </div>
      </Field>

      <label className="flex items-center gap-2 text-xs text-[#A1A1AA] cursor-pointer">
        <input type="checkbox" checked={value.avoidReuse} onChange={(e) => set({ avoidReuse: e.target.checked })} className="accent-[#8B5CF6]" />
        <span>{t('music.constraints.avoidReuse')}</span>
      </label>

      {/* global wish */}
      <Field label={t('music.constraints.query')}>
        <input
          value={value.query}
          onChange={(e) => set({ query: e.target.value })}
          placeholder={t('music.constraints.queryPlaceholder')}
          className="w-full bg-[#0F0F11] border border-[#27272A] focus:border-[#8B5CF6] rounded-lg px-3 py-2 text-sm text-[#FAFAFA] focus:outline-none"
        />
      </Field>

      {/* per-passage keywords */}
      <Field label={t('music.constraints.sectionQuery')}>
        {section ? (
          <div className="rounded-lg border p-2.5 flex flex-col gap-2" style={{ borderColor: hexToRgba(sectionColor(section.type), 0.6), background: hexToRgba(sectionColor(section.type), 0.08) }}>
            <div className="flex items-center gap-2 text-xs">
              <span className="w-2.5 h-2.5 rounded-sm" style={{ background: sectionColor(section.type) }} />
              <span className="text-[#FAFAFA] font-semibold">
                {t('music.constraints.sectionRange', {
                  name: t(sectionLabelKey(section.type)),
                  start: formatTime(section.start_ms),
                  end: formatTime(section.end_ms),
                })}
              </span>
              <span className="flex-1" />
              <button type="button" onClick={() => onSelectSection(null)} className="text-[#71717A] hover:text-[#FAFAFA]" title={t('music.clip.close')}>
                <X size={13} />
              </button>
            </div>
            <input
              autoFocus
              value={value.sectionQueries[String(section.index)] ?? ''}
              onChange={(e) => setSectionQuery(section.index, e.target.value)}
              placeholder={t('music.constraints.sectionQueryPlaceholder')}
              className="w-full bg-[#0F0F11] border border-[#27272A] focus:border-[#8B5CF6] rounded-lg px-3 py-1.5 text-sm text-[#FAFAFA] focus:outline-none"
            />
          </div>
        ) : (
          <div className="text-xs text-[#71717A]">{t('music.constraints.sectionPick')}</div>
        )}
        {keyed.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2">
            {keyed.map((s) => (
              <span key={s.index} className="inline-flex items-center gap-1 text-[11px] rounded-md pl-1.5 pr-1 py-0.5 text-[#FAFAFA]"
                    style={{ background: hexToRgba(sectionColor(s.type), 0.25) }}>
                <button type="button" onClick={() => onSelectSection(s.index)} className="inline-flex items-center gap-1">
                  <Tag size={10} />
                  <span className="font-semibold">{t(sectionLabelKey(s.type))} {formatTime(s.start_ms)}</span>
                  <span className="opacity-80">· {value.sectionQueries[String(s.index)]}</span>
                </button>
                <button type="button" onClick={() => setSectionQuery(s.index, '')} title={t('music.constraints.clear')} className="opacity-60 hover:opacity-100">
                  <X size={10} />
                </button>
              </span>
            ))}
          </div>
        )}
      </Field>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[11px] uppercase tracking-wider text-[#71717A]">{label}</span>
      {children}
    </div>
  );
}

function Pill({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs border transition-colors ${
        active
          ? 'bg-[#8B5CF6]/20 border-[#8B5CF6] text-[#FAFAFA]'
          : 'bg-[#0F0F11] border-[#27272A] text-[#A1A1AA] hover:border-[#3F3F46] hover:text-[#FAFAFA]'
      }`}
    >
      {children}
    </button>
  );
}
