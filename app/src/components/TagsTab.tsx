import { ChevronDown, MoreVertical, Film, Tag, ArrowUpDown, Loader2, Check, Layers, Download, Search } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api/client';
import { useT } from '../i18n';
import type { SceneResult, TagSummary, VideoSummary } from '../api/types';

interface Props {
  onOpenScene: (scene: SceneResult, allScenes: SceneResult[], index: number) => void;
  hoverDelayMs: number;
}

export default function TagsTab({ onOpenScene, hoverDelayMs }: Props) {
  const t = useT();
  const [videos, setVideos] = useState<VideoSummary[]>([]);
  const [tags, setTags] = useState<TagSummary[]>([]);
  const [scenes, setScenes] = useState<SceneResult[]>([]);
  const [videoIds, setVideoIds] = useState<number[]>([]);
  const [videoPickerOpen, setVideoPickerOpen] = useState(false);
  const [tagPickerOpen, setTagPickerOpen] = useState(false);
  const [tagSearch, setTagSearch] = useState('');
  const tagSearchRef = useRef<HTMLInputElement | null>(null);
  // Empty string = "no tag filter" → show all scenes of the picked videos.
  const [tagName, setTagName] = useState<string>('');
  const [threshold, setThreshold] = useState(75);
  const [sort, setSort] = useState<'timecode' | 'confidence'>('timecode');
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [loadingScenes, setLoadingScenes] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportSummary, setExportSummary] = useState<{ text: string; isError: boolean } | null>(null);

  useEffect(() => {
    api.listVideos().then((r) => {
      setVideos(r.videos);
      if (r.videos.length > 0 && videoIds.length === 0) setVideoIds([r.videos[0].id]);
    }).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (videoIds.length === 0) {
      setTags([]);
      // Keep tagName as '' so the next selection defaults to "All scenes".
      return;
    }
    const fetches = videoIds.map((id) => api.videoTags(id).then((r) => r.tags));
    Promise.all(fetches).then((all) => {
      const map = new Map<string, { category: string; count: number }>();
      all.flat().forEach((t) => {
        const cur = map.get(t.tag);
        if (cur) map.set(t.tag, { category: cur.category, count: cur.count + t.count });
        else map.set(t.tag, { category: t.category, count: t.count });
      });
      const merged = Array.from(map.entries())
        .map(([tag, v]) => ({ tag, category: v.category, count: v.count }))
        .sort((a, b) => b.count - a.count);
      setTags(merged);
      // If the previously picked tag no longer exists in this selection,
      // fall back to "All scenes" (empty string) — don't auto-pick the top
      // tag, which used to silently filter the list and hide most scenes.
      if (tagName && !merged.find((t) => t.tag === tagName)) setTagName('');
    }).catch((e) => setError(e.message));
  }, [videoIds.join(',')]);

  useEffect(() => {
    if (videoIds.length === 0) {
      setScenes([]);
      return;
    }
    setLoadingScenes(true);
    // tagName = '' means "no tag filter": the backend ignores the param and
    // returns every scene of the selected video(s).
    const tagFilter = tagName || undefined;
    const p = videoIds.length === 1
      ? api.videoScenes(videoIds[0], { tag: tagFilter, threshold: threshold / 100, sort })
      : api.videoScenes(videoIds, { tag: tagFilter, threshold: threshold / 100, sort });
    p.then((r) => setScenes(r.scenes))
      .catch((e) => setError(e.message))
      .finally(() => setLoadingScenes(false));
  }, [videoIds.join(','), tagName, threshold, sort]);

  const selectedTagSummary = useMemo(() => tags.find((t) => t.tag === tagName), [tags, tagName]);

  // Tags filtered by the dropdown search box. Empty query → all tags.
  const filteredTags = useMemo(() => {
    const q = tagSearch.trim().toLowerCase();
    if (!q) return tags;
    return tags.filter((t) => t.tag.toLowerCase().includes(q));
  }, [tags, tagSearch]);

  // Autofocus the search input when the dropdown opens; reset query when closed.
  useEffect(() => {
    if (tagPickerOpen) {
      setTimeout(() => tagSearchRef.current?.focus(), 0);
    } else {
      setTagSearch('');
    }
  }, [tagPickerOpen]);
  const videoLabel = useMemo(() => {
    if (videoIds.length === 0) return t('tags.noVideos');
    if (videoIds.length === 1) return videos.find((v) => v.id === videoIds[0])?.display_name ?? '...';
    return t('tags.videosSelected', { count: videoIds.length });
  }, [videoIds, videos, t]);

  function toggleSelect(id: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function toggleVideo(id: number) {
    setVideoIds((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]);
  }

  async function bulkExport() {
    if (selected.size === 0) return;
    setExporting(true);
    setExportSummary(null);
    try {
      const r = await api.exportBatch(Array.from(selected));
      setExportSummary({
        text: r.failed.length
          ? t('tags.exportedFailed', { count: r.exported.length, failed: r.failed.length })
          : t('tags.exported', { count: r.exported.length }),
        isError: false,
      });
      setSelected(new Set());
    } catch (err) {
      setExportSummary({ text: t('common.error', { message: (err as Error).message }), isError: true });
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className="flex flex-col h-full p-8 overflow-auto">
      {error && <div className="mb-4 bg-red-500/10 border border-red-500/40 text-red-300 text-sm rounded-lg px-4 py-2">{error}</div>}

      <div className="flex gap-4 mb-6">
        {/* Multi-select video picker */}
        <div className="flex-1 relative">
          <button onClick={() => setVideoPickerOpen((o) => !o)} className="w-full bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] hover:border-[#8B5CF6]/50 rounded-xl px-12 py-4 text-[#FAFAFA] flex items-center justify-between transition-all">
            <Film size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#8B5CF6]" />
            <span className="truncate text-left flex-1">{videoLabel}</span>
            {videoIds.length > 1 && <span className="ml-2 bg-[#8B5CF6]/20 text-[#8B5CF6] text-xs px-2 py-0.5 rounded-full font-bold">{videoIds.length}</span>}
            <ChevronDown size={20} className="text-[#8B5CF6] ml-2" />
          </button>
          {videoPickerOpen && (
            <div className="absolute top-full mt-2 left-0 right-0 z-40 bg-[#18181B] border border-[#27272A] rounded-xl max-h-80 overflow-auto shadow-2xl">
              {videos.length === 0 && <div className="px-4 py-3 text-[#71717A] text-sm">{t('tags.noVideosIndexed')}</div>}
              {videos.map((v) => {
                const checked = videoIds.includes(v.id);
                return (
                  <button key={v.id} onClick={() => toggleVideo(v.id)} className="w-full px-4 py-2 flex items-center gap-3 hover:bg-[#27272A] text-left">
                    <div className={`w-4 h-4 rounded border flex items-center justify-center ${checked ? 'border-[#8B5CF6] bg-[#8B5CF6]' : 'border-[#3f3f46]'}`}>
                      {checked && <Check size={12} className="text-white" />}
                    </div>
                    <span className="flex-1 text-sm text-[#FAFAFA] truncate">{v.display_name}</span>
                    <span className="text-xs text-[#71717A] font-mono">{t('tags.sceneCount', { count: v.scene_count })}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Tag picker — custom dropdown so it matches the dark theme. The
            native <select> rendered with the OS chrome (white popup on Windows)
            which clashed badly. */}
        <div className="flex-1 relative">
          <button
            onClick={() => setTagPickerOpen((o) => !o)}
            className="w-full bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] hover:border-[#EC4899]/50 rounded-xl px-12 py-4 text-[#FAFAFA] flex items-center justify-between transition-all"
          >
            <Tag size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#EC4899]" />
            <span className="truncate text-left flex-1">
              {tagName ? tagName : <span className="text-[#A1A1AA]">{t('tags.allScenes')} {t('tags.noTagFilter')}</span>}
            </span>
            {selectedTagSummary && (
              <span className="ml-2 bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white px-2 py-0.5 rounded-full text-xs font-bold">{selectedTagSummary.count}</span>
            )}
            <ChevronDown size={20} className="text-[#EC4899] ml-2" />
          </button>
          {tagPickerOpen && (
            <div className="absolute top-full mt-2 left-0 right-0 z-40 bg-[#18181B] border border-[#27272A] rounded-xl shadow-2xl flex flex-col max-h-96">
              {/* Search bar */}
              <div className="relative border-b border-[#27272A] flex-shrink-0">
                <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#71717A] pointer-events-none" />
                <input
                  ref={tagSearchRef}
                  type="text"
                  value={tagSearch}
                  onChange={(e) => setTagSearch(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Escape') setTagPickerOpen(false);
                    if (e.key === 'Enter' && filteredTags.length > 0) {
                      setTagName(filteredTags[0].tag);
                      setTagPickerOpen(false);
                    }
                  }}
                  placeholder={t('tags.filterTags')}
                  className="w-full bg-transparent pl-10 pr-4 py-3 text-[#FAFAFA] placeholder-[#71717A] text-sm focus:outline-none"
                />
              </div>
              {/* Options list */}
              <div className="overflow-auto flex-1">
                <button
                  onClick={() => { setTagName(''); setTagPickerOpen(false); }}
                  className={`w-full px-4 py-2 flex items-center gap-3 text-left hover:bg-[#27272A] ${!tagName ? 'bg-[#27272A]' : ''}`}
                >
                  <div className={`w-4 h-4 rounded-full border ${!tagName ? 'border-[#EC4899] bg-[#EC4899]' : 'border-[#3f3f46]'} flex items-center justify-center`}>
                    {!tagName && <Check size={12} className="text-white" />}
                  </div>
                  <span className="flex-1 text-sm text-[#FAFAFA]">{t('tags.allScenes')} <span className="text-[#71717A]">{t('tags.noTagFilter')}</span></span>
                </button>
                {filteredTags.length === 0 && tags.length > 0 && (
                  <div className="px-4 py-3 text-[#71717A] text-sm">{t('tags.noTagMatches', { query: tagSearch })}</div>
                )}
                {tags.length === 0 && (
                  <div className="px-4 py-3 text-[#71717A] text-sm">{t('tags.noTagsInSelection')}</div>
                )}
                {filteredTags.map((t) => {
                  const checked = tagName === t.tag;
                  return (
                    <button
                      key={t.tag}
                      onClick={() => { setTagName(t.tag); setTagPickerOpen(false); }}
                      className={`w-full px-4 py-2 flex items-center gap-3 text-left hover:bg-[#27272A] ${checked ? 'bg-[#27272A]' : ''}`}
                    >
                      <div className={`w-4 h-4 rounded-full border ${checked ? 'border-[#EC4899] bg-[#EC4899]' : 'border-[#3f3f46]'} flex items-center justify-center`}>
                        {checked && <Check size={12} className="text-white" />}
                      </div>
                      <span className="flex-1 text-sm text-[#FAFAFA] truncate">{t.tag}</span>
                      <span className="text-xs text-[#71717A] font-mono">{t.count}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between mb-6 bg-[#18181B]/50 backdrop-blur-sm rounded-xl p-4 border border-[#27272A]">
        <div className="flex items-center gap-8">
          <div className="flex items-center gap-4">
            <span className="text-[#71717A] text-sm uppercase tracking-wider font-medium">{t('tags.threshold')}</span>
            <div className="relative w-48 h-2 bg-gradient-to-r from-[#27272A] via-[#EC4899]/20 to-[#EC4899]/40 rounded-full overflow-hidden">
              <input type="range" min={0} max={100} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="absolute inset-0 w-full opacity-0 cursor-pointer" />
              <div className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-[#EC4899] rounded-full border-2 border-[#FAFAFA] shadow-lg shadow-[#EC4899]/50 pointer-events-none" style={{ left: `${threshold}%` }}></div>
            </div>
            <span className="text-[#EC4899] text-sm font-mono font-semibold w-10">{threshold}%</span>
          </div>
          <button onClick={() => setSort(sort === 'timecode' ? 'confidence' : 'timecode')} className="flex items-center gap-2 bg-[#0F0F11] border border-[#27272A] hover:border-[#8B5CF6]/50 rounded-lg px-4 py-2 text-[#A1A1AA] hover:text-[#FAFAFA] text-sm transition-all capitalize">
            <ArrowUpDown size={14} />
            <span>{t(`tags.sort.${sort}`)}</span>
          </button>
          {scenes.length > 0 && (
            <button onClick={() => { if (selected.size === scenes.length) setSelected(new Set()); else setSelected(new Set(scenes.map((s) => s.id))); }} className="text-xs text-[#71717A] hover:text-[#FAFAFA] flex items-center gap-1">
              <Layers size={12} /> {selected.size === scenes.length ? t('tags.clear') : t('tags.selectAll')}
            </button>
          )}
        </div>
        <div className="flex items-center gap-3">
          {exportSummary && <span className={`text-xs font-mono ${exportSummary.isError ? 'text-red-400' : 'text-[#8B5CF6]'}`}>{exportSummary.text}</span>}
          <button disabled={selected.size === 0 || exporting} onClick={bulkExport} className="bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-[#FAFAFA] px-6 py-2.5 rounded-xl text-sm font-semibold hover:shadow-lg hover:shadow-[#8B5CF6]/50 transition-all disabled:opacity-30 disabled:cursor-not-allowed flex items-center gap-2">
            {exporting ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
            {selected.size > 0 ? t('tags.exportSelectedCount', { count: selected.size }) : t('tags.exportSelected')}
          </button>
        </div>
      </div>

      {loadingScenes && (
        <div className="flex items-center justify-center text-[#71717A] py-12 gap-2"><Loader2 size={16} className="animate-spin" /> {t('tags.loadingScenes')}</div>
      )}

      {!loadingScenes && scenes.length === 0 && videoIds.length > 0 && (
        <div className="text-center text-[#71717A] py-12 text-sm">
          {tagName
            ? t('tags.noScenesMatch', { tag: tagName, threshold })
            : t('tags.noScenesYet')}
        </div>
      )}

      <div className="flex flex-col gap-2">
        {scenes.map((occ, idx) => {
          const isSelected = selected.has(occ.id);
          return (
            <TagRow
              key={occ.id}
              occ={occ}
              isSelected={isSelected}
              onToggle={() => toggleSelect(occ.id)}
              onOpen={() => onOpenScene(occ, scenes, idx)}
              hoverDelayMs={hoverDelayMs}
              showVideo={videoIds.length > 1}
            />
          );
        })}
      </div>
    </div>
  );
}

function TagRow({ occ, isSelected, onToggle, onOpen, hoverDelayMs, showVideo }:
  { occ: SceneResult; isSelected: boolean; onToggle: () => void; onOpen: () => void; hoverDelayMs: number; showVideo: boolean }) {
  const t = useT();
  const [showProxy, setShowProxy] = useState(false);
  const timeoutRef = useRef<number | null>(null);
  function start() {
    if (occ.proxy_path) timeoutRef.current = window.setTimeout(() => setShowProxy(true), hoverDelayMs);
  }
  function end() {
    setShowProxy(false);
    if (timeoutRef.current) { clearTimeout(timeoutRef.current); timeoutRef.current = null; }
  }
  return (
    <div className={`flex items-center gap-4 p-3 rounded-xl border-2 transition-all cursor-pointer ${isSelected ? 'bg-gradient-to-r from-[#8B5CF6]/20 to-[#EC4899]/20 border-[#8B5CF6] shadow-lg shadow-[#8B5CF6]/20' : 'bg-[#18181B]/30 border-transparent hover:border-[#27272A] hover:bg-[#18181B]/60'}`} onClick={onOpen}>
      <div onClick={(e) => { e.stopPropagation(); onToggle(); }} className="flex items-center justify-center">
        <input type="checkbox" checked={isSelected} readOnly className="w-5 h-5 accent-[#8B5CF6] cursor-pointer rounded" />
      </div>
      <div onMouseEnter={start} onMouseLeave={end} className="w-44 aspect-video bg-[#0F0F11] rounded-lg overflow-hidden border-2 border-[#27272A] relative group">
        {showProxy && occ.proxy_path ? (
          <video src={api.proxyUrl(occ.id)} autoPlay muted loop playsInline className="w-full h-full object-cover" />
        ) : (
          <img src={api.thumbnailUrl(occ.id)} alt={t('tags.sceneAlt')} className="w-full h-full object-cover" loading="lazy" />
        )}
      </div>
      <div className="flex-1 flex flex-col">
        <div className="font-mono text-[#FAFAFA] text-lg tracking-tight flex items-center gap-2">
          <span className="text-[#71717A]">›</span>
          {fmt(occ.start_ms)} → {fmt(occ.end_ms)}
        </div>
        {showVideo && <div className="text-xs text-[#71717A] truncate">{occ.video_display}</div>}
      </div>
      <div className="bg-gradient-to-br from-[#8B5CF6]/20 to-[#EC4899]/20 text-[#FAFAFA] px-4 py-2 rounded-lg font-mono font-bold border border-[#8B5CF6]/30 backdrop-blur-sm">
        {Math.round((occ.confidence ?? occ.score) * 100)}
      </div>
      <button onClick={(e) => e.stopPropagation()} className="text-[#71717A] hover:text-[#8B5CF6] transition-colors p-2 hover:bg-[#27272A] rounded-lg">
        <MoreVertical size={20} />
      </button>
    </div>
  );
}

function fmt(ms: number): string {
  const total = Math.floor(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}
