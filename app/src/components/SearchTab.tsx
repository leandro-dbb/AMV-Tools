import { Search, Play, Sparkles, Loader2, FolderPlus, X, Image as ImageIcon, Clapperboard } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api/client';
import type { SceneResult, SearchFilters } from '../api/types';
import { useT } from '../i18n';
import TimelineExportDialog from './TimelineExportDialog';
import { filtersActive, searchDefaults } from '../searchplus';
import type { SearchPlusRequest } from '../searchplus';
import SearchFiltersPanel from './SearchFiltersPanel';
import SourcePreview from './SourcePreview';
import { AddToCharacterButton, CharacterChips, SceneBadges, SimilarButton } from './SceneBadges';

const FILTERS_KEY = 'amv-search-filters';

function loadFilters(): SearchFilters {
  try { return JSON.parse(localStorage.getItem(FILTERS_KEY) || '{}') as SearchFilters; } catch { return {}; }
}

// Shown until the first searches: generic shots any anime library has.
const DEFAULT_RECENT = ['explosion', 'close-up face', 'gros plan visage'];

interface Props {
  onOpenScene: (scene: SceneResult, allScenes: SceneResult[], index: number) => void;
  hoverDelayMs: number;
  // search+: "similar shots" / "search this character" coming from other tabs
  request?: { req: SearchPlusRequest; nonce: number } | null;
}

export default function SearchTab({ onOpenScene, hoverDelayMs, request }: Props) {
  const t = useT();
  const [query, setQuery] = useState('');
  const [submitted, setSubmitted] = useState('');
  // SigLIP 2 NaFlex produces lower cosine scores than v1 (sigmoid loss).
  // 10% is a sane default that surfaces semantic matches without burying them.
  const [threshold, setThreshold] = useState(10);
  const [sort, setSort] = useState<'relevance' | 'duration' | 'video' | 'random'>('relevance');
  const [maxResults, setMaxResults] = useState(200);
  // The "index a folder first" hint used to show on every fresh Search tab,
  // even with an indexed library. null = not known yet (show neither).
  const [libraryScenes, setLibraryScenes] = useState<number | null>(null);
  useEffect(() => {
    api.listVideos().then((r) => setLibraryScenes(r.videos.reduce((n, v) => n + (v.scene_count || 0), 0))).catch(() => {});
  }, []);
  // Start from Settings > Search (threshold, max results, sort): the tab used
  // to ignore them and always search with 10 % / 200 / relevance.
  useEffect(() => {
    api.getSettings().then((st) => {
      const d = searchDefaults(st.search);
      setThreshold(d.threshold);
      setSort(d.sort);
      setMaxResults(d.maxResults);
    }).catch(() => {});
  }, []);
  const [results, setResults] = useState<SceneResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recent, setRecent] = useState<string[]>(DEFAULT_RECENT);
  const [empty, setEmpty] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [allTags, setAllTags] = useState<{ tag: string; count: number }[]>([]);
  const [suggestionsOpen, setSuggestionsOpen] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [timelineOpen, setTimelineOpen] = useState(false);
  const [filters, setFiltersState] = useState<SearchFilters>(loadFilters);
  const [filtersOpen, setFiltersOpen] = useState(() => filtersActive(loadFilters()));
  const [similar, setSimilar] = useState<{ seed: SceneResult; excludeSameVideo: boolean; sameGroup: boolean } | null>(null);
  const setFilters = useCallback((f: SearchFilters) => {
    setFiltersState(f);
    try { localStorage.setItem(FILTERS_KEY, JSON.stringify(f)); } catch {}
  }, []);
  const activeFilters = filtersActive(filters) ? filters : null;

  useEffect(() => {
    api.recentSearches().then((r) => {
      if (r.queries.length > 0) setRecent(r.queries.slice(0, 5));
    }).catch(() => {});
    api.allTags(1000).then((r) => setAllTags(r.tags)).catch(() => {});
  }, []);

  const runTextSearch = useCallback(async (q: string, override?: SearchFilters | null) => {
    const f = override !== undefined ? override : activeFilters;
    // Empty query + filters = browse ("every zoom-in of Gojo").
    if (!q.trim() && !f) return;
    setImagePreview(null);
    setSimilar(null);
    setLoading(true);
    setError(null);
    setSubmitted(q.trim() || '[filters]');
    lastTextQuery.current = q;
    setSuggestionsOpen(false);
    try {
      const res = await api.search({ query: q.trim() || undefined, threshold: threshold / 100, sort, top_k: maxResults, filters: f });
      setResults(res.results);
      // The chips under the box only ever showed the searches of the last launch.
      if (q.trim()) setRecent((prev) => [q.trim(), ...prev.filter((x) => x !== q.trim())].slice(0, 5));
      setEmpty(res.results.length === 0);
    } catch (err) {
      setError((err as Error).message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [threshold, sort, maxResults, activeFilters]);

  const runSimilar = useCallback(async (sceneId: number, opts: { excludeSameVideo: boolean; sameGroup: boolean }) => {
    setImagePreview(null);
    setLoading(true);
    setError(null);
    setSubmitted(`[similar] ${sceneId}`);
    lastTextQuery.current = null;
    try {
      const res = await api.similarScenes(sceneId, {
        top_k: maxResults, exclude_same_video: opts.excludeSameVideo, same_group: opts.sameGroup,
        sort, filters: activeFilters,
      });
      setSimilar({ seed: res.seed, ...opts });
      setResults(res.results);
      setEmpty(res.results.length === 0);
    } catch (err) {
      setError((err as Error).message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [sort, maxResults, activeFilters]);

  // Moving the threshold slider or changing the sort did nothing until the
  // user pressed Enter again: re-run the last text search (debounced).
  const lastTextQuery = useRef<string | null>(null);
  const firstOptionsRun = useRef(true);
  useEffect(() => {
    if (firstOptionsRun.current) { firstOptionsRun.current = false; return; }
    const q = lastTextQuery.current;
    if (q === null || similar || imagePreview) return;
    const id = window.setTimeout(() => { runTextSearch(q); }, 350);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threshold, sort]);

  // Requests from other tabs (Tags, Derush, MiniEditor, Characters).
  const lastNonce = useRef<number | null>(null);
  useEffect(() => {
    if (!request || request.nonce === lastNonce.current) return;
    lastNonce.current = request.nonce;
    if (request.req.kind === 'similar') {
      runSimilar(request.req.sceneId, { excludeSameVideo: true, sameGroup: false });
    } else {
      const f: SearchFilters = { ...filters, characters: [request.req.characterId] };
      setFilters(f);
      setFiltersOpen(true);
      runTextSearch(query, f);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request]);

  const runImageSearch = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setSimilar(null);
    setSubmitted(`[image] ${file.name}`);
    lastTextQuery.current = null;
    const url = URL.createObjectURL(file);
    setImagePreview(url);
    try {
      const res = await api.searchImage(file, { threshold: threshold / 100, top_k: maxResults, filters: activeFilters });
      setResults(res.results);
      setEmpty(res.results.length === 0);
    } catch (err) {
      setError((err as Error).message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [threshold, maxResults, activeFilters]);

  const lastToken = useMemo(() => {
    const m = query.match(/(\S+)$/);
    return m ? m[1].toLowerCase() : '';
  }, [query]);

  const suggestions = useMemo(() => {
    if (!lastToken || lastToken.length < 2) return [];
    return allTags.filter((t) => t.tag.toLowerCase().includes(lastToken)).slice(0, 8);
  }, [lastToken, allTags]);

  function applySuggestion(tag: string) {
    setQuery((q) => q.replace(/(\S+)$/, tag) + ' ');
    setSuggestionsOpen(false);
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      runImageSearch(file);
    }
  }, [runImageSearch]);

  return (
    <div className="flex flex-col h-full p-8 overflow-auto relative"
         onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
         onDragLeave={() => setDragOver(false)}
         onDrop={onDrop}>
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[500px] h-[500px] bg-[#8B5CF6]/10 rounded-full blur-[120px] pointer-events-none"></div>

      {dragOver && (
        <div className="absolute inset-4 border-4 border-dashed border-[#8B5CF6] rounded-2xl bg-[#8B5CF6]/10 flex items-center justify-center pointer-events-none z-30">
          <div className="text-center">
            <ImageIcon size={48} className="text-[#8B5CF6] mx-auto mb-2" />
            <div className="text-[#FAFAFA] font-semibold text-lg">{t('search.dropImage')}</div>
          </div>
        </div>
      )}

      <div className="flex flex-col items-center mb-6 relative z-10">
        <form onSubmit={(e) => { e.preventDefault(); runTextSearch(query); }} className="relative w-[750px] mb-6 group">
          <div className="absolute -inset-1 bg-gradient-to-r from-[#8B5CF6] via-[#EC4899] to-[#8B5CF6] rounded-xl opacity-0 group-hover:opacity-20 blur transition-opacity"></div>
          <div className="relative">
            <Search className="absolute left-5 top-1/2 -translate-y-1/2 text-[#8B5CF6]" size={22} />
            <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
              <button type="button" onClick={() => fileInputRef.current?.click()} className="text-[#8B5CF6]/60 hover:text-[#8B5CF6] p-1.5 rounded-lg hover:bg-[#27272A]" title={t('search.byImageTitle')}>
                <ImageIcon size={18} />
              </button>
              {loading ? <Loader2 className="text-[#8B5CF6] animate-spin" size={20} /> : <Sparkles className="text-[#8B5CF6]/60" size={18} />}
            </div>
            <input
              value={query}
              onChange={(e) => { setQuery(e.target.value); setSuggestionsOpen(true); }}
              onFocus={() => setSuggestionsOpen(true)}
              onBlur={() => setTimeout(() => setSuggestionsOpen(false), 150)}
              type="text"
              spellCheck={false}
              placeholder={t('search.placeholder')}
              className="w-full bg-[#18181B]/80 backdrop-blur-xl border-2 border-[#27272A] rounded-xl pl-14 pr-24 py-4 text-[#FAFAFA] text-lg placeholder:text-[#71717A] focus:outline-none focus:border-[#8B5CF6] transition-all shadow-2xl shadow-[#8B5CF6]/5"
            />
            <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) runImageSearch(f); }} />
            {suggestionsOpen && suggestions.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-2 bg-[#18181B] border border-[#27272A] rounded-xl overflow-hidden shadow-2xl z-50">
                {suggestions.map((s) => (
                  <button key={s.tag} type="button" onMouseDown={(e) => e.preventDefault()} onClick={() => applySuggestion(s.tag)} className="w-full px-4 py-2 text-left text-sm hover:bg-[#27272A] flex items-center justify-between">
                    <span className="text-[#FAFAFA] font-mono">{s.tag}</span>
                    <span className="text-[#71717A] text-xs">{s.count.toLocaleString()}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </form>

        {imagePreview && (
          <div className="mb-4 relative">
            <img src={imagePreview} className="h-20 rounded-lg border-2 border-[#8B5CF6]/50" alt={t('search.refAlt')} />
            <button onClick={() => { setImagePreview(null); setSubmitted(''); setResults([]); }} className="absolute -top-2 -right-2 bg-[#0F0F11] border border-[#27272A] rounded-full p-1 hover:border-[#EC4899]">
              <X size={12} className="text-[#A1A1AA]" />
            </button>
          </div>
        )}

        <div className="flex gap-2 mb-6 flex-wrap justify-center max-w-[750px]">
          {recent.map((s, i) => (
            <button key={i} onClick={() => { setQuery(s); runTextSearch(s); }} className="px-4 py-2 bg-gradient-to-br from-[#27272A] to-[#18181B] text-[#A1A1AA] rounded-full text-sm hover:from-[#8B5CF6]/20 hover:to-[#EC4899]/20 hover:text-[#FAFAFA] hover:border-[#8B5CF6]/50 border border-[#27272A] transition-all">
              {s}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-8 w-[750px] bg-[#18181B]/50 backdrop-blur-sm rounded-xl p-4 border border-[#27272A]">
          <div className="flex items-center gap-3">
            <span className="text-[#71717A] text-sm uppercase tracking-wider font-medium">{t('search.sort')}</span>
            <select value={sort} onChange={(e) => setSort(e.target.value as any)} className="bg-[#0F0F11] border border-[#8B5CF6]/30 rounded-lg px-4 py-2 text-[#FAFAFA] text-sm focus:outline-none focus:border-[#8B5CF6] cursor-pointer">
              <option value="relevance">{t('search.sortRelevance')}</option>
              <option value="duration">{t('search.sortDuration')}</option>
              <option value="video">{t('search.sortVideo')}</option>
              <option value="random">{t('search.sortRandom')}</option>
            </select>
          </div>
          <div className="flex items-center gap-4 flex-1">
            <span className="text-[#71717A] text-sm uppercase tracking-wider font-medium">{t('search.threshold')}</span>
            <div className="relative flex-1 h-2 bg-gradient-to-r from-[#27272A] via-[#8B5CF6]/20 to-[#8B5CF6]/40 rounded-full overflow-hidden">
              <input type="range" min={0} max={100} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="absolute inset-0 w-full opacity-0 cursor-pointer" />
              <div className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-[#8B5CF6] rounded-full border-2 border-[#FAFAFA] shadow-lg shadow-[#8B5CF6]/50 pointer-events-none" style={{ left: `${threshold}%` }}></div>
            </div>
            <span className="text-[#8B5CF6] text-sm font-mono font-semibold w-10">{threshold}%</span>
          </div>
        </div>

        <SearchFiltersPanel
          filters={filters}
          onChange={setFilters}
          open={filtersOpen}
          onToggle={() => setFiltersOpen((o) => !o)}
          onApply={() => {
            if (similar) runSimilar(similar.seed.id, similar);
            else runTextSearch(query);
          }}
        />

        {similar && (
          <div className="w-[750px] mt-3 flex items-center gap-4 bg-[#8B5CF6]/10 border border-[#8B5CF6]/40 rounded-xl p-3">
            <img src={api.thumbnailUrl(similar.seed.id)} alt="" className="h-14 aspect-video object-cover rounded-md border border-[#8B5CF6]/60" />
            <div className="min-w-0 flex-1">
              <div className="text-[11px] uppercase tracking-wider text-[#A1A1AA]">{t('sp.similar.title')}</div>
              <div className="text-sm text-[#FAFAFA] truncate">{similar.seed.video_display} · {formatTime(similar.seed.start_ms)}</div>
            </div>
            <label className="text-xs text-[#A1A1AA] flex items-center gap-1.5 cursor-pointer">
              <input type="checkbox" className="accent-[#8B5CF6]" checked={similar.excludeSameVideo}
                     onChange={(e) => runSimilar(similar.seed.id, { ...similar, excludeSameVideo: e.target.checked })} />
              {t('sp.similar.excludeSameVideo')}
            </label>
            <label className="text-xs text-[#A1A1AA] flex items-center gap-1.5 cursor-pointer">
              <input type="checkbox" className="accent-[#8B5CF6]" checked={similar.sameGroup}
                     onChange={(e) => runSimilar(similar.seed.id, { ...similar, sameGroup: e.target.checked })} />
              {t('sp.similar.sameGroup')}
            </label>
            <button onClick={() => { setSimilar(null); setResults([]); setSubmitted(''); }} title={t('sp.similar.clear')}
                    className="p-1.5 rounded-lg text-[#71717A] hover:text-[#EC4899] hover:bg-[#27272A]"><X size={14} /></button>
          </div>
        )}
      </div>

      {error && (
        <div className="max-w-[750px] mx-auto mb-4 bg-red-500/10 border border-red-500/40 text-red-300 text-sm rounded-lg px-4 py-2">{error}</div>
      )}

      {loading && results.length === 0 && (
        <div data-testid="search-loading" className="flex items-center justify-center gap-2 mt-12 text-sm text-[#A1A1AA]">
          <Loader2 size={16} className="animate-spin text-[#8B5CF6]" />
          {/* The first search of a session loads SigLIP: 30 s+ on a CPU. */}
          {t('journey.search.loading')}
        </div>
      )}

      {!submitted && !loading && results.length === 0 && libraryScenes !== null && libraryScenes > 0 && (
        <div data-testid="search-ready" className="flex flex-col items-center justify-center text-center max-w-md mx-auto mt-12 text-[#71717A]">
          <Search size={40} className="mb-3 text-[#8B5CF6]/60" />
          <div className="text-[#A1A1AA] mb-1">{t('journey.search.ready', { n: (libraryScenes ?? 0).toLocaleString() })}</div>
          <div className="text-xs">{t('journey.search.readyHint')}</div>
        </div>
      )}

      {!submitted && !loading && results.length === 0 && libraryScenes === 0 && (
        <div className="flex flex-col items-center justify-center text-center max-w-md mx-auto mt-12 text-[#71717A]">
          <FolderPlus size={40} className="mb-3 text-[#8B5CF6]/60" />
          <div className="text-[#A1A1AA] mb-1">{t('search.emptyIndexHint')}</div>
          <div className="text-xs">{t('search.hintGoTo')} <span className="text-[#FAFAFA]">{t('search.hintPath')}</span> {t('search.hintOrDrop')}</div>
        </div>
      )}

      {empty && submitted && (
        <div className="text-center text-[#71717A] mt-8 text-sm">{similar ? t('sp.similar.none') : t('search.noMatches', { threshold })}</div>
      )}

      {results.length > 0 && (
        <div className="flex items-center justify-between mb-3 relative z-10">
          <div className="text-xs text-[#71717A] flex items-center gap-3 flex-wrap">
            <span>{t('timeline.scope.searchResults')} · {results.length}</span>
            {/* Real run: "gros plan visage" found 1 shot where "close-up face" found 6 at 10 %. */}
            {results.length < 5 && !similar && !imagePreview && submitted && !submitted.startsWith('[') && threshold > 5 && (
              <span data-testid="few-results" className="text-[#FBBF24]">
                {t('journey.search.few')}{' '}
                <button onClick={() => setThreshold(5)} className="underline hover:text-[#FAFAFA]">{t('journey.search.lowerThreshold')}</button>
              </span>
            )}
          </div>
          <button onClick={() => setTimelineOpen(true)} className="flex items-center gap-2 border-2 border-[#8B5CF6]/50 bg-[#18181B] text-[#FAFAFA] px-4 py-2 rounded-xl text-sm font-semibold hover:bg-[#8B5CF6]/15">
            <Clapperboard size={14} className="text-[#8B5CF6]" />
            {t('timeline.button')}
          </button>
        </div>
      )}

      <div className="grid grid-cols-6 gap-3 relative z-10">
        {results.map((scene, idx) => (
          <SceneCard key={scene.id} scene={scene} hoverDelayMs={hoverDelayMs}
                     onClick={() => onOpenScene(scene, results, idx)} />
        ))}
      </div>

      {timelineOpen && (
        <TimelineExportDialog
          selection={{ scene_ids: results.map((r) => r.id) }}
          defaultName="AMV search"
          scopeLabel={t('timeline.scope.searchResults')}
          onClose={() => setTimelineOpen(false)}
        />
      )}
    </div>
  );
}

function SceneCard({ scene, hoverDelayMs, onClick }: { scene: SceneResult; hoverDelayMs: number; onClick: () => void }) {
  const [hovering, setHovering] = useState(false);
  const [showProxy, setShowProxy] = useState(false);
  const timeoutRef = useRef<number | null>(null);

  // Proxies are off by default: without one, preview the scene straight from
  // the source file (falls back to the thumbnail if Chromium cannot decode it,
  // e.g. HEVC on Linux).
  const [sourceFailed, setSourceFailed] = useState(false);
  const canPreview = !!scene.proxy_path || !sourceFailed;
  function startHover() {
    setHovering(true);
    if (canPreview) {
      timeoutRef.current = window.setTimeout(() => setShowProxy(true), hoverDelayMs);
    }
  }
  function endHover() {
    setHovering(false);
    setShowProxy(false);
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }

  return (
    <div className="relative group cursor-pointer" onMouseEnter={startHover} onMouseLeave={endHover} onClick={onClick}>
      <div className="aspect-video bg-[#18181B] rounded-lg overflow-hidden border border-[#27272A] relative group-hover:border-[#8B5CF6]/50 transition-all group-hover:scale-[1.02] group-hover:shadow-xl group-hover:shadow-[#8B5CF6]/20">
        {showProxy && scene.proxy_path ? (
          <video src={api.proxyUrl(scene.id)} autoPlay muted loop playsInline className="w-full h-full object-cover" />
        ) : showProxy && !sourceFailed ? (
          <SourcePreview scene={scene} onFail={() => { setSourceFailed(true); setShowProxy(false); }} />
        ) : (
          <img src={api.thumbnailUrl(scene.id)} alt={scene.video_display} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" loading="lazy" />
        )}
        <div className="absolute top-2 right-2 bg-gradient-to-br from-[#8B5CF6] to-[#EC4899] text-[#FAFAFA] px-2 py-1 rounded-md text-xs font-mono font-bold shadow-lg">{Math.round(scene.score * 100)}</div>
        {!showProxy && <SceneBadges scene={scene} />}
        {hovering && (
          <>
            {!showProxy && (
              <div className="absolute inset-0 bg-gradient-to-t from-[#8B5CF6]/40 via-black/30 to-black/10 flex items-center justify-center">
                <div className="w-12 h-12 bg-[#8B5CF6] rounded-full flex items-center justify-center shadow-2xl shadow-[#8B5CF6]/50">
                  <Play className="text-white ml-1" size={20} fill="white" />
                </div>
              </div>
            )}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black via-black/80 to-transparent p-3 pt-8">
              <div className="text-[#FAFAFA] text-xs font-semibold mb-1 flex items-center justify-between">
                <span className="truncate mr-2">{scene.video_display}</span>
                <span className="font-mono text-[#8B5CF6]">{formatTime(scene.start_ms)}</span>
              </div>
              <CharacterChips scene={scene} />
              <div className="text-[#A1A1AA] text-xs flex gap-1 flex-wrap mt-1">
                {scene.tags.slice(0, 3).map((tag, i) => (
                  <span key={i} className="bg-[#27272A]/80 px-2 py-0.5 rounded-full">{tag}</span>
                ))}
              </div>
            </div>
          </>
        )}
        <div className="absolute top-10 right-2 flex flex-col gap-1 z-20 opacity-0 group-hover:opacity-100 transition-opacity">
          <SimilarButton sceneId={scene.id} />
          <AddToCharacterButton sceneId={scene.id} />
        </div>
      </div>
    </div>
  );
}

function formatTime(ms: number): string {
  const total = Math.floor(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}
