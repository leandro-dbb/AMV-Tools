import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Check, Crop, ImagePlus, Loader2, Pencil, Plus, Search, Trash2, Undo2, Upload, Users, X,
} from 'lucide-react';
import { api } from '../api/client';
import type { Character, CharacterExample, SceneResult, VideoSummary } from '../api/types';
import { useT } from '../i18n';
import { requestCharacterSearch } from '../searchplus';
import { SceneBadges, SimilarButton } from './SceneBadges';

interface Props {
  onOpenScene: (scene: SceneResult, allScenes: SceneResult[], index: number) => void;
}

/**
 * Characters — per-series "people" built from example scenes / images.
 * Left: list + create. Right: examples (crop / remove / drop images), a
 * text search to find more examples, and the scored scenes with
 * validate / reject feedback that re-scores the character.
 */
export default function CharactersTab({ onOpenScene }: Props) {
  const t = useT();
  const [characters, setCharacters] = useState<Character[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [creating, setCreating] = useState(false);
  const [groups, setGroups] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async (selectId?: number) => {
    try {
      const r = await api.listCharacters();
      setCharacters(r.characters);
      if (selectId != null) setSelectedId(selectId);
      else if (r.characters.length && !r.characters.some((c) => c.id === selectedId)) setSelectedId(r.characters[0].id);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [selectedId]);

  useEffect(() => {
    refresh();
    api.listVideos().then((r) => {
      const g = Array.from(new Set(r.videos.map((v: VideoSummary) => v.group_name).filter((x): x is string => !!x))).sort();
      setGroups(g);
    }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selected = characters.find((c) => c.id === selectedId) ?? null;

  return (
    <div className="flex h-full">
      {/* list */}
      <div className="w-72 border-r-2 border-[#27272A] bg-[#18181B]/40 flex flex-col">
        <div className="p-4 flex items-center gap-2 border-b border-[#27272A]">
          <Users size={16} className="text-[#EC4899]" />
          <span className="font-semibold text-[#FAFAFA] flex-1">{t('sp.chars.title')}</span>
          <button onClick={() => setCreating(true)} title={t('sp.chars.new')}
                  className="p-1.5 rounded-lg bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white"><Plus size={14} /></button>
        </div>
        {creating && (
          <CreateForm groups={groups} onCancel={() => setCreating(false)}
                      onCreated={(c) => { setCreating(false); refresh(c.id); }} />
        )}
        <div className="flex-1 overflow-auto p-2 flex flex-col gap-1">
          {characters.length === 0 && !creating && (
            <div className="text-xs text-[#71717A] p-3 leading-relaxed">{t('sp.chars.empty')}</div>
          )}
          {characters.map((c) => (
            <button key={c.id} onClick={() => setSelectedId(c.id)}
                    className={`flex items-center gap-3 p-2 rounded-xl text-left transition-all ${c.id === selectedId ? 'bg-gradient-to-r from-[#8B5CF6]/25 to-[#EC4899]/20 border border-[#8B5CF6]/60' : 'hover:bg-[#27272A] border border-transparent'}`}>
              <div className="w-14 aspect-video rounded-md overflow-hidden bg-black flex-shrink-0">
                {c.cover_scene_id != null ? (
                  <img src={api.thumbnailUrl(c.cover_scene_id)} className="w-full h-full object-cover" alt="" />
                ) : c.cover_example_id != null ? (
                  <img src={api.characterExampleThumbUrl(c.id, c.cover_example_id)} className="w-full h-full object-cover" alt="" />
                ) : null}
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-[#FAFAFA] truncate">{c.name}</div>
                <div className="text-[10px] text-[#71717A] truncate">{c.group_name ?? t('sp.chars.allLibrary')} · {c.matches}</div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* detail */}
      <div className="flex-1 overflow-auto">
        {error && <div className="m-4 bg-red-500/10 border border-red-500/40 text-red-300 text-sm rounded-lg px-4 py-2">{error}</div>}
        {selected ? (
          <CharacterDetail key={selected.id} character={selected} groups={groups} onOpenScene={onOpenScene}
                           onChanged={() => refresh()} onDeleted={() => { setSelectedId(null); refresh(); }} onError={setError} />
        ) : (
          <div className="h-full flex items-center justify-center text-[#71717A] text-sm">{t('sp.chars.pick')}</div>
        )}
      </div>
    </div>
  );
}

function CreateForm({ groups, onCancel, onCreated }: { groups: string[]; onCancel: () => void; onCreated: (c: Character) => void }) {
  const t = useT();
  const [name, setName] = useState('');
  const [group, setGroup] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function submit() {
    if (!name.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      onCreated(await api.createCharacter({ name: name.trim(), group_name: group || null }));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="p-3 border-b border-[#27272A] flex flex-col gap-2 bg-[#0F0F11]/60">
      <label className="text-[10px] uppercase tracking-wider text-[#71717A]">{t('sp.chars.name')}</label>
      <input autoFocus value={name} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') submit(); if (e.key === 'Escape') onCancel(); }}
             placeholder={t('sp.chars.namePlaceholder')}
             className="bg-[#18181B] border border-[#27272A] rounded-lg px-3 py-2 text-sm text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]" />
      <label className="text-[10px] uppercase tracking-wider text-[#71717A]">{t('sp.chars.group')}</label>
      <select value={group} onChange={(e) => setGroup(e.target.value)}
              className="bg-[#18181B] border border-[#27272A] rounded-lg px-3 py-2 text-sm text-[#FAFAFA]">
        <option value="">{t('sp.chars.allLibrary')}</option>
        {groups.map((g) => <option key={g} value={g}>{g}</option>)}
      </select>
      {err && <div className="text-xs text-red-400">{err}</div>}
      <div className="flex gap-2 justify-end">
        <button onClick={onCancel} className="px-3 py-1.5 text-xs text-[#A1A1AA] hover:text-[#FAFAFA]">{t('sp.chars.cancel')}</button>
        <button disabled={busy || !name.trim()} onClick={submit}
                className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white disabled:opacity-40 flex items-center gap-1">
          {busy && <Loader2 size={12} className="animate-spin" />}{t('sp.chars.create')}
        </button>
      </div>
    </div>
  );
}

function CharacterDetail({ character, groups, onOpenScene, onChanged, onDeleted, onError }: {
  character: Character;
  groups: string[];
  onOpenScene: Props['onOpenScene'];
  onChanged: () => void;
  onDeleted: () => void;
  onError: (msg: string | null) => void;
}) {
  const t = useT();
  const [detail, setDetail] = useState<Character>(character);
  const [matches, setMatches] = useState<SceneResult[]>([]);
  const [threshold, setThreshold] = useState(character.threshold);
  const [busy, setBusy] = useState(false);
  const [showBorderline, setShowBorderline] = useState(true);
  const [cropping, setCropping] = useState<CharacterExample | null>(null);
  const [renaming, setRenaming] = useState(false);
  const [nameDraft, setNameDraft] = useState(character.name);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    const [d, m] = await Promise.all([api.getCharacter(character.id), api.characterMatches(character.id, 400)]);
    setDetail(d);
    setMatches(m.results);
    setThreshold(m.threshold);
  }, [character.id]);

  useEffect(() => { load().catch((e) => onError((e as Error).message)); }, [load, onError]);

  // Every mutation returns the refreshed character; reload matches after.
  async function mutate(p: Promise<unknown>) {
    setBusy(true);
    onError(null);
    try {
      await p;
      await load();
      onChanged();
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const visible = useMemo(
    () => matches.filter((m) => showBorderline || m.score >= threshold || m.label === 'positive'),
    [matches, showBorderline, threshold],
  );
  const above = matches.filter((m) => m.score >= threshold).length;

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith('image/'));
    if (files.length) mutate((async () => { for (const f of files) await api.addCharacterImage(character.id, f); })());
  }

  const examples = detail.examples ?? [];

  return (
    <div className="p-6 flex flex-col gap-5">
      {/* header */}
      <div className="flex items-center gap-3 flex-wrap">
        {renaming ? (
          <input autoFocus value={nameDraft} onChange={(e) => setNameDraft(e.target.value)}
                 onBlur={() => { setRenaming(false); if (nameDraft.trim() && nameDraft !== detail.name) mutate(api.patchCharacter(character.id, { name: nameDraft.trim() })); }}
                 onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); if (e.key === 'Escape') setRenaming(false); }}
                 className="bg-[#0F0F11] border border-[#8B5CF6] rounded-lg px-3 py-1.5 text-xl font-bold text-[#FAFAFA] focus:outline-none" />
        ) : (
          <h2 className="text-2xl font-bold text-[#FAFAFA] flex items-center gap-2">
            {detail.name}
            <button onClick={() => { setNameDraft(detail.name); setRenaming(true); }} title={t('sp.chars.rename')} className="text-[#71717A] hover:text-[#FAFAFA]"><Pencil size={14} /></button>
          </h2>
        )}
        <select value={detail.group_name ?? ''} disabled={busy}
                onChange={(e) => mutate(e.target.value ? api.patchCharacter(character.id, { group_name: e.target.value }) : api.patchCharacter(character.id, { clear_group: true }))}
                className="bg-[#0F0F11] border border-[#27272A] rounded-lg px-3 py-1.5 text-sm text-[#FAFAFA]">
          <option value="">{t('sp.chars.allLibrary')}</option>
          {groups.map((g) => <option key={g} value={g}>{g}</option>)}
        </select>
        <span className="text-xs text-[#71717A]">{t('sp.chars.stats', { matches: above, pos: detail.positives, neg: detail.negatives })}</span>
        {busy && <span className="text-xs text-[#8B5CF6] flex items-center gap-1"><Loader2 size={12} className="animate-spin" />{t('sp.chars.working')}</span>}
        <span className="flex-1" />
        <button onClick={() => requestCharacterSearch(character.id)}
                className="px-4 py-2 rounded-lg text-sm font-semibold bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white flex items-center gap-2">
          <Search size={14} />{t('sp.chars.searchWith')}
        </button>
        <button title={t('sp.chars.delete')} onClick={() => { if (confirm(t('sp.chars.deleteConfirm', { name: detail.name }))) api.deleteCharacter(character.id).then(onDeleted).catch((e) => onError(e.message)); }}
                className="p-2 rounded-lg text-[#71717A] hover:text-red-400 hover:bg-[#27272A]"><Trash2 size={16} /></button>
      </div>

      {/* examples */}
      <div className={`rounded-xl border-2 p-4 transition-all ${dragOver ? 'border-dashed border-[#EC4899] bg-[#EC4899]/10' : 'border-[#27272A] bg-[#18181B]/40'}`}
           onDragOver={(e) => { e.preventDefault(); setDragOver(true); }} onDragLeave={() => setDragOver(false)} onDrop={onDrop}>
        <div className="flex items-center gap-2 mb-2">
          <span className="text-[#71717A] text-xs uppercase tracking-wider font-medium">{t('sp.chars.examples')}</span>
          <span className="flex-1" />
          <button onClick={() => fileRef.current?.click()} className="text-xs text-[#A1A1AA] hover:text-[#FAFAFA] flex items-center gap-1"><Upload size={12} />{t('sp.chars.addImage')}</button>
          <input ref={fileRef} type="file" accept="image/*" multiple className="hidden"
                 onChange={(e) => { const fs = Array.from(e.target.files ?? []); e.target.value = ''; if (fs.length) mutate((async () => { for (const f of fs) await api.addCharacterImage(character.id, f); })()); }} />
        </div>
        <div className="text-[11px] text-[#71717A] mb-3">{t('sp.chars.examplesHint')} <span className="italic">{t('sp.chars.needsModel')}</span></div>
        <div className="flex gap-2 flex-wrap">
          {examples.map((ex) => (
            <div key={ex.id} className={`relative w-36 aspect-video rounded-lg overflow-hidden border-2 group ${ex.polarity > 0 ? 'border-[#22C55E]/60' : 'border-red-500/60 opacity-70'}`}>
              {ex.has_thumbnail || ex.scene_id == null ? (
                <img src={api.characterExampleThumbUrl(character.id, ex.id)} className="w-full h-full object-cover" alt="" />
              ) : (
                <img src={api.thumbnailUrl(ex.scene_id)} className="w-full h-full object-cover" alt="" />
              )}
              {ex.crop && <div className="absolute top-1 left-1 bg-black/70 text-white rounded px-1 text-[9px] flex items-center gap-0.5"><Crop size={9} /></div>}
              {ex.polarity < 0 && <div className="absolute top-1 left-1 bg-red-500/90 text-white rounded px-1 text-[9px]">{t('sp.chars.negative')}</div>}
              <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                {ex.scene_id != null && ex.polarity > 0 && (
                  <button title={t('sp.chars.crop')} onClick={() => setCropping(ex)} className="p-1.5 rounded-md bg-[#8B5CF6] text-white"><Crop size={12} /></button>
                )}
                <button title={t('sp.chars.removeExample')} onClick={() => mutate(api.deleteCharacterExample(character.id, ex.id))} className="p-1.5 rounded-md bg-red-500 text-white"><X size={12} /></button>
              </div>
            </div>
          ))}
          {examples.length === 0 && (
            <div className="w-36 aspect-video rounded-lg border-2 border-dashed border-[#27272A] flex flex-col items-center justify-center text-[10px] text-[#71717A] gap-1">
              <ImagePlus size={16} />{t('sp.chars.dropImage')}
            </div>
          )}
        </div>
        <ExampleFinder onAdd={(sid) => mutate(api.addCharacterExample(character.id, { scene_id: sid }))} />
      </div>

      {/* matches */}
      <div className="flex items-center gap-4 bg-[#18181B]/50 rounded-xl p-3 border border-[#27272A]">
        <span className="text-[#71717A] text-xs uppercase tracking-wider font-medium">{t('sp.chars.matches')}</span>
        <span className="text-[#FAFAFA] font-mono text-sm">{above}</span>
        <span className="text-[#71717A] text-xs uppercase tracking-wider font-medium ml-4">{t('sp.chars.threshold')}</span>
        <input type="range" min={40} max={97} value={Math.round(threshold * 100)} className="w-48 accent-[#EC4899]"
               onChange={(e) => setThreshold(Number(e.target.value) / 100)}
               onMouseUp={() => mutate(api.patchCharacter(character.id, { threshold }))}
               onKeyUp={() => mutate(api.patchCharacter(character.id, { threshold }))} />
        <span className="text-[#EC4899] font-mono text-sm w-10">{Math.round(threshold * 100)}%</span>
        {detail.threshold_auto && <span className="text-[10px] text-[#71717A] uppercase">{t('sp.chars.thresholdAuto')}</span>}
        <span className="flex-1" />
        <label className="text-xs text-[#A1A1AA] flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={showBorderline} onChange={(e) => setShowBorderline(e.target.checked)} className="accent-[#8B5CF6]" />
          {t('sp.chars.showBorderline')}
        </label>
      </div>
      {/* Real run: Beerus from 3 examples matched 86 of 98 shots, Goku included.
          Nothing said that rejecting wrong shots is how it gets sharper. */}
      {examples.length > 0 && examples.length < 6 && above > 0 && (
        <div data-testid="chars-refine-hint" className="text-xs text-[#A1A1AA] -mt-2 px-1">{t('journey.chars.refine')}</div>
      )}

      {visible.length === 0 && <div className="text-center text-sm text-[#71717A] py-8">{t('sp.chars.noMatches')}</div>}
      <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))' }}>
        {visible.map((scene, idx) => {
          const below = scene.score < threshold && scene.label !== 'positive';
          return (
            <div key={scene.id} className={`group relative rounded-lg overflow-hidden border-2 bg-[#18181B] transition-all ${scene.label === 'positive' ? 'border-[#22C55E]/70' : below ? 'border-[#27272A] opacity-55 hover:opacity-100' : 'border-[#27272A] hover:border-[#8B5CF6]/60'}`}>
              <div className="relative aspect-video cursor-pointer" onClick={() => onOpenScene(scene, visible, idx)}>
                <img src={api.thumbnailUrl(scene.id)} className="w-full h-full object-cover" alt="" loading="lazy" />
                <SceneBadges scene={scene} />
                <div className="absolute top-2 right-2 bg-gradient-to-br from-[#8B5CF6] to-[#EC4899] text-white px-1.5 py-0.5 rounded text-[10px] font-mono font-bold">{Math.round(scene.score * 100)}</div>
              </div>
              <div className="flex items-center gap-1 p-1.5">
                <span className="text-[10px] text-[#71717A] truncate flex-1">{scene.video_display}</span>
                {scene.label ? (
                  <button title={t('sp.chars.undo')} onClick={() => mutate(api.characterFeedback(character.id, { clear: [scene.id] }))}
                          className="p-1 rounded text-[#71717A] hover:text-[#FAFAFA]"><Undo2 size={13} /></button>
                ) : (
                  <>
                    <button title={t('sp.chars.accept')} disabled={busy} onClick={() => mutate(api.characterFeedback(character.id, { accept: [scene.id] }))}
                            className="p-1 rounded text-[#71717A] hover:text-[#22C55E] hover:bg-[#27272A]"><Check size={14} /></button>
                    <button title={t('sp.chars.reject')} disabled={busy} onClick={() => mutate(api.characterFeedback(character.id, { reject: [scene.id] }))}
                            className="p-1 rounded text-[#71717A] hover:text-red-400 hover:bg-[#27272A]"><X size={14} /></button>
                  </>
                )}
                <SimilarButton sceneId={scene.id} className="p-1 rounded text-[#71717A] hover:text-[#8B5CF6] hover:bg-[#27272A]" />
              </div>
            </div>
          );
        })}
      </div>

      {cropping && cropping.scene_id != null && (
        <CropDialog sceneId={cropping.scene_id} initial={cropping.crop}
                    onCancel={() => setCropping(null)}
                    onSave={(box) => {
                      const ex = cropping;
                      setCropping(null);
                      // Replace the example by its cropped version.
                      mutate((async () => {
                        await api.addCharacterExample(character.id, { scene_id: ex.scene_id!, crop: box });
                        await api.deleteCharacterExample(character.id, ex.id);
                      })());
                    }} />
      )}
    </div>
  );
}

/** Text search to find more example scenes without leaving the tab. */
function ExampleFinder({ onAdd }: { onAdd: (sceneId: number) => void }) {
  const t = useT();
  const [q, setQ] = useState('');
  const [results, setResults] = useState<SceneResult[]>([]);
  const [loading, setLoading] = useState(false);
  async function run() {
    if (!q.trim()) return;
    setLoading(true);
    try {
      const r = await api.search({ query: q, threshold: 0.05, top_k: 24 });
      setResults(r.results);
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  }
  return (
    <div className="mt-4">
      <form onSubmit={(e) => { e.preventDefault(); run(); }} className="relative max-w-md">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#71717A]" />
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder={t('sp.chars.findPlaceholder')}
               className="w-full bg-[#0F0F11] border border-[#27272A] rounded-lg pl-9 pr-9 py-2 text-sm text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6]" />
        {loading && <Loader2 size={14} className="absolute right-3 top-1/2 -translate-y-1/2 animate-spin text-[#8B5CF6]" />}
      </form>
      {results.length > 0 && (
        <div className="flex gap-2 overflow-x-auto mt-3 pb-2">
          {results.map((r) => (
            <button key={r.id} onClick={() => { onAdd(r.id); setResults((xs) => xs.filter((x) => x.id !== r.id)); }}
                    title={t('sp.chars.addAsExample')}
                    className="relative flex-shrink-0 w-32 aspect-video rounded-md overflow-hidden border-2 border-[#27272A] hover:border-[#22C55E] group">
              <img src={api.thumbnailUrl(r.id)} className="w-full h-full object-cover" alt="" loading="lazy" />
              <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 flex items-center justify-center"><Plus size={18} className="text-white" /></div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** Drag a normalised box on the scene thumbnail. */
function CropDialog({ sceneId, initial, onCancel, onSave }: {
  sceneId: number;
  initial: [number, number, number, number] | null;
  onCancel: () => void;
  onSave: (box: [number, number, number, number]) => void;
}) {
  const t = useT();
  const [box, setBox] = useState<[number, number, number, number] | null>(initial);
  const start = useRef<[number, number] | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  function pos(e: React.MouseEvent): [number, number] {
    const r = ref.current!.getBoundingClientRect();
    return [Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1), Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1)];
  }
  return (
    <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-8" onClick={onCancel}>
      <div className="bg-[#18181B] border-2 border-[#27272A] rounded-2xl p-5 max-w-4xl w-full" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center mb-3">
          <span className="font-semibold text-[#FAFAFA] flex items-center gap-2"><Crop size={16} className="text-[#8B5CF6]" />{t('sp.chars.crop')}</span>
          <span className="flex-1" />
          <button onClick={onCancel} className="text-[#71717A] hover:text-[#FAFAFA]"><X size={18} /></button>
        </div>
        <div className="text-xs text-[#71717A] mb-3">{t('sp.chars.cropHint')}</div>
        <div className="flex justify-center bg-black rounded-lg">
        <div ref={ref} className="relative inline-block select-none cursor-crosshair"
             onMouseDown={(e) => { const p = pos(e); start.current = p; setBox([p[0], p[1], p[0], p[1]]); }}
             onMouseMove={(e) => { if (!start.current) return; const p = pos(e); const s = start.current; setBox([Math.min(s[0], p[0]), Math.min(s[1], p[1]), Math.max(s[0], p[0]), Math.max(s[1], p[1])]); }}
             onMouseUp={() => { start.current = null; }}>
          {/* The box is relative to the image itself (not a letterboxed frame). */}
          <img src={api.thumbnailUrl(sceneId)} className="block max-w-full max-h-[60vh] min-w-[480px] pointer-events-none" alt="" draggable={false} />
          {box && (
            <div className="absolute border-2 border-[#EC4899] bg-[#EC4899]/10 shadow-[0_0_0_9999px_rgba(0,0,0,0.5)]"
                 style={{ left: `${box[0] * 100}%`, top: `${box[1] * 100}%`, width: `${(box[2] - box[0]) * 100}%`, height: `${(box[3] - box[1]) * 100}%` }} />
          )}
        </div>
        </div>
        <div className="flex justify-end gap-2 mt-4">
          <button onClick={onCancel} className="px-4 py-2 text-sm text-[#A1A1AA] hover:text-[#FAFAFA]">{t('sp.chars.cancel')}</button>
          <button disabled={!box || box[2] - box[0] < 0.03 || box[3] - box[1] < 0.03} onClick={() => box && onSave(box)}
                  className="px-4 py-2 rounded-lg text-sm font-semibold bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white disabled:opacity-40">{t('sp.chars.cropSave')}</button>
        </div>
      </div>
    </div>
  );
}
