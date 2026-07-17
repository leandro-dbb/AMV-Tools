import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Check, ChevronDown, ChevronRight, Clapperboard, Combine, Download, FolderPlus, Folder, FolderOpen,
  Library, Pause, Pencil, Play, Rewind, FastForward, SkipBack, SkipForward,
  Star, Trash2, Volume2, VolumeX, X, Loader2, MonitorPlay, ZoomIn, ZoomOut,
} from 'lucide-react';
import { api, getBaseUrl } from '../api/client';
import type { DerushFolder, DerushItem, ProgressEvent, SceneResult, VideoSummary } from '../api/types';
import { DERUSH_DEFAULT_KEYS, displayKey } from '../derushKeys';
import { useT } from '../i18n';

/*
 * Derush mode — the AMV-maker dailies workflow.
 *
 * Player view: play whole episodes with Adobe-style JKL shuttle, keep the
 * scene under the playhead with one key (Enter), chain episodes.
 * Library view: browse everything you kept, rename, organise into folders,
 * batch-export the lot with the regular export settings.
 */

type View = 'player' | 'library';

// Shuttle velocity ladder: L = +1 step, J = -1 step, on a signed scale of
// -10…-1, 1…10 (0 = paused). L at 10 wraps back to 1, J at -10 wraps to -1.
const MAX_SPEED = 10;
// Timeline zoom bounds: ×64 turns a 2-frame scene of a 24-min episode into a
// clickable ~15 px block.
const MAX_ZOOM = 64;

function fmt(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  const mm = (m % 60).toString().padStart(2, '0');
  const ss = (s % 60).toString().padStart(2, '0');
  return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

export default function DerushTab({ onOpenScene }: {
  onOpenScene?: (scenes: SceneResult[], index: number) => void;
}) {
  const t = useT();
  const [view, setView] = useState<View>('player');
  const [folders, setFolders] = useState<DerushFolder[]>([]);
  const [items, setItems] = useState<DerushItem[]>([]);
  const [lastEvent, setLastEvent] = useState<ProgressEvent | null>(null);
  const [keys, setKeys] = useState<Record<string, string>>({ ...DERUSH_DEFAULT_KEYS });

  const refresh = useCallback(() => {
    api.derushList().then((r) => { setFolders(r.folders); setItems(r.items); }).catch(() => {});
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  // Key bindings from Settings > Derush (remounting the tab refreshes them).
  useEffect(() => {
    api.getSettings()
      .then((s) => setKeys({ ...DERUSH_DEFAULT_KEYS, ...(s.derush?.keys ?? {}) }))
      .catch(() => {});
  }, []);

  // Dedicated progress socket: derush_export + derush_prepare events.
  useEffect(() => {
    let ws: WebSocket | null = null;
    let alive = true;
    const open = () => {
      try {
        ws = api.progressSocket();
        ws.onmessage = (msg) => {
          try {
            const evt = JSON.parse(msg.data) as ProgressEvent;
            if (evt.type === 'derush_export' || evt.type === 'derush_prepare') setLastEvent(evt);
          } catch {}
        };
        ws.onclose = () => { if (alive) setTimeout(open, 1500); };
      } catch {}
    };
    open();
    return () => { alive = false; ws?.close(); };
  }, []);

  const keptIds = useMemo(() => new Set(items.map((i) => i.scene_id)), [items]);
  const favIds = useMemo(() => new Set(items.filter((i) => i.favorite).map((i) => i.scene_id)), [items]);
  const keptByVideo = useMemo(() => {
    const m = new Map<number, number>();
    for (const i of items) m.set(i.video_id, (m.get(i.video_id) ?? 0) + 1);
    return m;
  }, [items]);

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 border-b-2 border-[#27272A] bg-[#18181B]/40">
        {([
          { id: 'player', label: t('derush.tab.player'), icon: MonitorPlay },
          { id: 'library', label: t('derush.tab.library', { n: items.length }), icon: Library },
        ] as { id: View; label: string; icon: typeof Library }[]).map((v) => (
          <button
            key={v.id}
            onClick={() => setView(v.id)}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-semibold transition-all ${
              view === v.id
                ? 'bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white'
                : 'text-[#71717A] hover:text-[#A1A1AA] hover:bg-[#27272A]'
            }`}
          >
            <v.icon size={15} /> {v.label}
          </button>
        ))}
        <div className="ml-auto text-xs text-[#71717A]" title={t('derush.hint.remapTitle')}>
          {displayKey(keys.shuttle_slower)}/{displayKey(keys.pause)}/{displayKey(keys.shuttle_faster)} {t('derush.hint.shuttle')}
          {' · '}{displayKey(keys.level_up)}/{displayKey(keys.level_down)} {t('derush.hint.keepLevel')}
          {' · '}{displayKey(keys.prev_scene)}/{displayKey(keys.next_scene)} {t('derush.hint.scene')}
          {' · '}{displayKey(keys.frame_back)}/{displayKey(keys.frame_forward)} {t('derush.hint.frame')}
          {' · '}Ctrl+click + {displayKey(keys.merge)} {t('derush.hint.merge')}
          {' · '}{displayKey(keys.mute)} {t('derush.hint.mute')}
          {' · '}{displayKey(keys.prev_episode)}/{displayKey(keys.next_episode)} {t('derush.hint.episode')}
        </div>
      </div>
      <div className="flex-1 overflow-hidden">
        {view === 'player' ? (
          <PlayerView folders={folders} keptIds={keptIds} favIds={favIds} keptByVideo={keptByVideo} keys={keys} lastEvent={lastEvent} onChanged={refresh} />
        ) : (
          <LibraryView folders={folders} items={items} lastEvent={lastEvent} onChanged={refresh} onOpenScene={onOpenScene} />
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────── player ────
function PlayerView({ folders, keptIds, favIds, keptByVideo, keys, lastEvent, onChanged }: {
  folders: DerushFolder[];
  keptIds: Set<number>;
  favIds: Set<number>;
  keptByVideo: Map<number, number>;
  keys: Record<string, string>;
  lastEvent: ProgressEvent | null;
  onChanged: () => void;
}) {
  const t = useT();
  const [videos, setVideos] = useState<VideoSummary[]>([]);
  const [video, setVideo] = useState<VideoSummary | null>(null);
  const [scenes, setScenes] = useState<SceneResult[]>([]);
  const [src, setSrc] = useState<string | null>(null);
  const [loadingSrc, setLoadingSrc] = useState(false);
  const [playError, setPlayError] = useState<'decode' | 'missing' | null>(null);
  const [transcoding, setTranscoding] = useState(false);
  const [currentMs, setCurrentMs] = useState(0);
  const [durationMs, setDurationMs] = useState(0);
  const [muted, setMuted] = useState(false);
  const [autoNext, setAutoNext] = useState(true);
  const [targetFolder, setTargetFolder] = useState<number | ''>('');
  const [flash, setFlash] = useState<{ text: string; kept: boolean } | null>(null);
  const [vel, setVel] = useState(0);                 // signed shuttle velocity, 0 = paused
  const [selected, setSelected] = useState<Set<number>>(new Set());

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const velRef = useRef(vel);
  velRef.current = vel;
  const rafRef = useRef<number>(0);
  const flashTimer = useRef<number>(0);

  const showFlash = useCallback((text: string, kept = true) => {
    setFlash({ text, kept });
    window.clearTimeout(flashTimer.current);
    flashTimer.current = window.setTimeout(() => setFlash(null), 1100);
  }, []);

  useEffect(() => {
    api.listVideos().then((r) => {
      const usable = r.videos.filter((v) => v.scene_count > 0);
      setVideos(usable);
      if (usable.length && !video) loadEpisode(usable[0]);
    }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadEpisode = useCallback((v: VideoSummary) => {
    setVideo(v);
    setScenes([]);
    setSrc(null);
    setPlayError(null);
    setCurrentMs(0);
    setDurationMs(v.duration_ms || 0);
    setVel(0);
    setSelected(new Set());
    setLoadingSrc(true);
    api.videoScenes(v.id, { topOnly: true }).then((r) => setScenes(r.scenes)).catch(() => {});
    api.videoPlayable(v.id)
      .then((r) => setSrc(`${getBaseUrl()}${r.url}`))
      .catch((e) => {
        // 410 = the source file is gone (moved/renamed on disk) — a totally
        // different problem than an undecodable codec; don't conflate them.
        const msg = (e as Error).message || '';
        setPlayError(msg.includes('410') || msg.includes('missing') ? 'missing' : 'decode');
      })
      .finally(() => setLoadingSrc(false));
  }, []);

  // Transcode finished for the loaded episode → reload the playable URL.
  useEffect(() => {
    if (!lastEvent || lastEvent.type !== 'derush_prepare') return;
    if (lastEvent.video_id !== video?.id) return;
    if (lastEvent.done) {
      setTranscoding(false);
      if (lastEvent.ok && video) {
        setPlayError(null);
        api.videoPlayable(video.id).then((r) => setSrc(`${getBaseUrl()}${r.url}`)).catch(() => {});
      }
    } else {
      setTranscoding(true);
    }
  }, [lastEvent, video]);

  // Warm the remux cache of the NEXT episode in the background so chaining
  // (auto-next / N key) starts instantly instead of waiting for ffmpeg.
  useEffect(() => {
    if (!video) return;
    const idx = videos.findIndex((v) => v.id === video.id);
    const next = idx >= 0 ? videos[idx + 1] : undefined;
    if (!next) return;
    const t = window.setTimeout(() => { api.videoPlayable(next.id).catch(() => {}); }, 4000);
    return () => window.clearTimeout(t);
  }, [video, videos]);

  // Playlist grouped by import subfolder; named groups first (backend order),
  // ungrouped episodes render flat at the end without a header.
  const episodeGroups = useMemo(() => {
    const m = new Map<string, VideoSummary[]>();
    for (const v of videos) {
      const g = v.group_name ?? '';
      if (!m.has(g)) m.set(g, []);
      m.get(g)!.push(v);
    }
    const entries = [...m.entries()];
    entries.sort((a, b) => (a[0] === '' ? 1 : b[0] === '' ? -1 : a[0].localeCompare(b[0])));
    return entries;
  }, [videos]);
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set());
  const [editingVideo, setEditingVideo] = useState<number | null>(null);
  const [groupInput, setGroupInput] = useState('');
  const groupNames = useMemo(
    () => episodeGroups.map(([g]) => g).filter((g) => g !== ''),
    [episodeGroups],
  );

  const reloadVideos = useCallback(() => {
    api.listVideos().then((r) => setVideos(r.videos.filter((v) => v.scene_count > 0))).catch(() => {});
  }, []);

  const commitGroupMove = useCallback((videoId: number) => {
    api.setVideoGroup(videoId, groupInput.trim() || null).then(() => {
      setEditingVideo(null);
      reloadVideos();
    }).catch(() => setEditingVideo(null));
  }, [groupInput, reloadVideos]);

  const videoIndex = useMemo(() => videos.findIndex((v) => v.id === video?.id), [videos, video]);
  const gotoEpisode = useCallback((delta: number) => {
    if (videoIndex < 0) return;
    const next = videos[videoIndex + delta];
    if (next) loadEpisode(next);
  }, [videoIndex, videos, loadEpisode]);

  const currentSceneIdx = useMemo(() => {
    if (!scenes.length) return -1;
    return scenes.findIndex((s) => currentMs >= s.start_ms && currentMs < s.end_ms);
  }, [scenes, currentMs]);
  const currentScene = currentSceneIdx >= 0 ? scenes[currentSceneIdx] : null;

  // ── shuttle engine ────────────────────────────────────────────────────────
  const stopReverseLoop = useCallback(() => {
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = 0; }
  }, []);

  const applyShuttle = useCallback((v: number) => {
    const el = videoRef.current;
    if (!el) return;
    stopReverseLoop();
    setVel(v);
    // Synchronous ref update: the reverse loop below (and shuttleStep) must
    // see the new velocity IMMEDIATELY — waiting for the re-render to refresh
    // the ref made the first rAF tick read a stale >= 0 value and die.
    velRef.current = v;
    if (v > 0) {
      el.playbackRate = v;
      el.play().catch(() => {});
    } else if (v === 0) {
      el.pause();
      el.playbackRate = 1;
    } else {
      // Chromium can't play backwards, so reverse = a chain of seeks. Two
      // rules make it actually move: (1) never issue a seek while the
      // previous one is still decoding — precise seeks on long-GOP h264 take
      // far longer than a display frame, and re-assigning currentTime every
      // rAF just cancels them forever (frozen picture); (2) anchor the
      // target to the wall clock so the requested speed holds on average
      // even when the decoder is slow (it skips frames instead of lagging).
      el.pause();
      const anchorTime = el.currentTime;
      const anchorClock = performance.now();
      let pendingSeek = false;
      const onSeeked = () => { pendingSeek = false; };
      const tick = () => {
        const cur = velRef.current;
        if (cur >= 0) { el.removeEventListener('seeked', onSeeked); return; }
        if (!pendingSeek) {
          const target = anchorTime + ((performance.now() - anchorClock) / 1000) * cur;
          if (target <= 0) {
            el.currentTime = 0;
            el.removeEventListener('seeked', onSeeked);
            setVel(0);
            velRef.current = 0;
            return;
          }
          pendingSeek = true;
          el.currentTime = target;
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      el.addEventListener('seeked', onSeeked);
      rafRef.current = requestAnimationFrame(tick);
    }
  }, [stopReverseLoop]);

  // L = one step faster, J = one step slower (crossing into reverse below 1×).
  // At the top of the ladder the next press wraps back to 1× in the same
  // direction — "x2 puis x3 … jusqu'à x10 avant de revenir à 1".
  const shuttleStep = useCallback((delta: 1 | -1) => {
    const v = velRef.current;
    let next: number;
    if (delta === 1) next = v === MAX_SPEED ? 1 : v === -1 ? 1 : v + 1;
    else next = v === -MAX_SPEED ? -1 : v === 1 ? -1 : v - 1;
    applyShuttle(next);
  }, [applyShuttle]);

  useEffect(() => () => stopReverseLoop(), [stopReverseLoop]);

  // ── keep scene (favorite = the "Favoris" pre-treatment level) ────────────
  const keepCurrent = useCallback((favorite = false) => {
    if (!currentScene) return;
    api.derushToggle(currentScene.id, targetFolder === '' ? null : targetFolder, favorite).then((r) => {
      showFlash(
        !r.kept ? t('derush.flash.removed') : r.favorite ? t('derush.flash.favorite') : t('derush.flash.kept'),
        r.kept,
      );
      onChanged();
    }).catch(() => {});
  }, [currentScene, targetFolder, onChanged, showFlash, t]);

  // ── keep ladder (arrow keys): none ↔ kept ↔ favorite ─────────────────────
  const levelChange = useCallback((delta: 1 | -1) => {
    if (!currentScene) return;
    const before = favIds.has(currentScene.id) ? 2 : keptIds.has(currentScene.id) ? 1 : 0;
    api.derushLevel(currentScene.id, delta, targetFolder === '' ? null : targetFolder).then((r) => {
      if (r.level === before) {
        if (before === 2 && delta === 1) showFlash(t('derush.flash.alreadyFavorite'), true);
        return;
      }
      const msg = delta === 1
        ? (r.level === 2 ? t('derush.flash.favorite') : t('derush.flash.kept'))
        : (r.level === 1 ? t('derush.flash.favToKept') : t('derush.flash.removed'));
      showFlash(msg, r.level > 0);
      onChanged();
    }).catch(() => {});
  }, [currentScene, favIds, keptIds, targetFolder, onChanged, showFlash, t]);

  const seekTo = useCallback((ms: number) => {
    const el = videoRef.current;
    if (el) el.currentTime = Math.max(0, ms / 1000);
    followOnce.current = true;   // bring the playhead back into view when zoomed
  }, []);

  // ── Premiere-style scrubbing: drag the playhead anywhere on the strip ────
  // Seeks are throttled through rAF so a fast drag doesn't flood the decoder
  // with hundreds of currentTime writes.
  const stripRef = useRef<HTMLDivElement | null>(null);       // inner (zoomed) strip
  const scrollRef = useRef<HTMLDivElement | null>(null);      // scroll viewport
  const [scrubbing, setScrubbing] = useState(false);
  const scrubbingRef = useRef(false);
  const scrubTargetMs = useRef<number | null>(null);
  const scrubRaf = useRef(0);
  const [zoom, setZoom] = useState(1);                        // 1 = fit, up to MAX_ZOOM
  const followOnce = useRef(false);                           // centre playhead after a jump

  const posToMs = useCallback((clientX: number) => {
    const rect = stripRef.current?.getBoundingClientRect();
    if (!rect || durationMs <= 0) return 0;
    const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    return frac * durationMs;
  }, [durationMs]);

  const pumpScrub = useCallback(() => {
    const el = videoRef.current;
    if (el && scrubTargetMs.current != null) {
      el.currentTime = scrubTargetMs.current / 1000;
      scrubTargetMs.current = null;
    }
    scrubRaf.current = scrubbingRef.current ? requestAnimationFrame(pumpScrub) : 0;
  }, []);

  const onStripPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    // Ctrl/⌘+click is the merge-selection gesture on segments — leave it be.
    if (e.ctrlKey || e.metaKey || e.button !== 0 || durationMs <= 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    scrubbingRef.current = true;
    setScrubbing(true);
    applyShuttle(0);                       // pause while scrubbing, like Premiere
    const ms = posToMs(e.clientX);
    scrubTargetMs.current = ms;
    setCurrentMs(ms);
    if (!scrubRaf.current) scrubRaf.current = requestAnimationFrame(pumpScrub);
  }, [durationMs, posToMs, applyShuttle, pumpScrub]);

  const onStripPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!scrubbingRef.current) return;
    // Dragging against a viewport edge pans the zoomed timeline, NLE-style.
    const sc = scrollRef.current;
    if (sc) {
      const rect = sc.getBoundingClientRect();
      if (e.clientX > rect.right - 32) sc.scrollLeft += 18;
      else if (e.clientX < rect.left + 32) sc.scrollLeft -= 18;
    }
    const ms = posToMs(e.clientX);
    scrubTargetMs.current = ms;
    setCurrentMs(ms);
  }, [posToMs]);

  const endScrub = useCallback(() => {
    scrubbingRef.current = false;
    setScrubbing(false);
  }, []);

  useEffect(() => () => { scrubbingRef.current = false; if (scrubRaf.current) cancelAnimationFrame(scrubRaf.current); }, []);

  // ── timeline zoom ─────────────────────────────────────────────────────────
  // Zoom keeping a given viewport x (px) anchored to the same timestamp.
  const zoomAround = useCallback((factor: number, anchorViewportX?: number) => {
    const sc = scrollRef.current;
    setZoom((z) => {
      const nz = Math.min(MAX_ZOOM, Math.max(1, z * factor));
      if (sc && nz !== z) {
        const anchor = anchorViewportX ?? sc.clientWidth / 2;
        const frac = (sc.scrollLeft + anchor) / Math.max(1, sc.scrollWidth);
        requestAnimationFrame(() => {
          sc.scrollLeft = frac * sc.clientWidth * nz - anchor;
        });
      }
      return nz;
    });
  }, []);

  // Ctrl+wheel zooms around the cursor; plain wheel pans when zoomed in.
  // Attached manually because React wheel listeners are passive (we need
  // preventDefault to stop the page/browser zoom).
  useEffect(() => {
    const sc = scrollRef.current;
    if (!sc) return;
    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const rect = sc.getBoundingClientRect();
        zoomAround(e.deltaY < 0 ? 1.35 : 1 / 1.35, e.clientX - rect.left);
      } else if (sc.scrollWidth > sc.clientWidth) {
        e.preventDefault();
        sc.scrollLeft += e.deltaY !== 0 ? e.deltaY : e.deltaX;
      }
    };
    sc.addEventListener('wheel', onWheel, { passive: false });
    return () => sc.removeEventListener('wheel', onWheel);
  }, [zoomAround]);

  // Keep the playhead in view: continuously while playing, once after a jump.
  useEffect(() => {
    const sc = scrollRef.current;
    if (!sc || durationMs <= 0 || scrubbingRef.current) return;
    if (velRef.current === 0 && !followOnce.current) return;
    followOnce.current = false;
    const px = (currentMs / durationMs) * sc.scrollWidth;
    if (px < sc.scrollLeft + 48 || px > sc.scrollLeft + sc.clientWidth - 48) {
      sc.scrollLeft = px - sc.clientWidth / 2;
    }
  }, [currentMs, durationMs]);

  // Fresh episode → back to fit.
  useEffect(() => { setZoom(1); }, [video?.id]);

  // ── merge selection ───────────────────────────────────────────────────────
  const toggleSelect = useCallback((sceneId: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(sceneId)) next.delete(sceneId); else next.add(sceneId);
      return next;
    });
  }, []);

  // Selection is mergeable when it maps to a consecutive run of segments.
  const mergeable = useMemo(() => {
    if (selected.size < 2) return false;
    const idxs = scenes.reduce<number[]>((acc, s, i) => (selected.has(s.id) ? [...acc, i] : acc), []);
    return idxs.length === selected.size && idxs.every((v, k) => k === 0 || v === idxs[k - 1] + 1);
  }, [selected, scenes]);

  const mergeSelection = useCallback(() => {
    if (!video) return;
    if (!mergeable) {
      if (selected.size >= 2) showFlash(t('derush.flash.mergeAdjacent'), false);
      return;
    }
    api.mergeScenes([...selected]).then((r) => {
      showFlash(t('derush.flash.merged', { start: fmt(r.start_ms), end: fmt(r.end_ms) }));
      setSelected(new Set());
      api.videoScenes(video.id, { topOnly: true }).then((res) => setScenes(res.scenes)).catch(() => {});
      onChanged();
    }).catch((e) => showFlash((e as Error).message.replace(/^API \d+: /, ''), false));
  }, [video, mergeable, selected, onChanged, showFlash, t]);

  // ── keyboard ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
      const el = videoRef.current;
      if (!el) return;
      const pressed = e.key.toLowerCase();
      if (pressed === 'escape') { e.preventDefault(); setSelected(new Set()); return; }
      // Remappable bindings (Settings > Derush) — key → action lookup.
      const action = Object.keys(keys).find((a) => keys[a] === pressed);
      if (!action) return;
      e.preventDefault();
      switch (action) {
        case 'shuttle_slower': shuttleStep(-1); break;
        case 'pause': applyShuttle(0); break;
        case 'shuttle_faster': shuttleStep(1); break;
        case 'play_pause': applyShuttle(velRef.current !== 0 ? 0 : 1); break;
        case 'mute': setMuted((m) => !m); break;
        case 'level_up': levelChange(1); break;
        case 'level_down': levelChange(-1); break;
        case 'toggle_keep': keepCurrent(false); break;
        case 'toggle_favorite': keepCurrent(true); break;
        case 'merge': mergeSelection(); break;
        case 'next_episode': gotoEpisode(1); break;
        case 'prev_episode': gotoEpisode(-1); break;
        case 'prev_scene':
          if (currentSceneIdx > 0) seekTo(scenes[currentSceneIdx - 1].start_ms + 1);
          break;
        case 'next_scene':
          if (currentSceneIdx >= 0 && currentSceneIdx < scenes.length - 1) seekTo(scenes[currentSceneIdx + 1].start_ms + 1);
          break;
        case 'frame_back':
          applyShuttle(0);
          el.currentTime = Math.max(0, el.currentTime - 1 / (video?.fps || 24));
          break;
        case 'frame_forward':
          applyShuttle(0);
          el.currentTime = el.currentTime + 1 / (video?.fps || 24);
          break;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [keys, shuttleStep, applyShuttle, keepCurrent, levelChange, mergeSelection, gotoEpisode, seekTo, scenes, currentSceneIdx, video?.fps]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.muted = muted;
  }, [muted, src]);

  const keptInEpisode = useMemo(() => scenes.filter((s) => keptIds.has(s.id)).length, [scenes, keptIds]);

  return (
    <div className="h-full flex overflow-hidden">
      {/* episode playlist, grouped by import subfolder */}
      <div className="w-64 shrink-0 border-r-2 border-[#27272A] bg-[#18181B]/30 overflow-y-auto">
        <div className="px-3 py-2 text-xs uppercase tracking-wider text-[#71717A] font-semibold">{t('derush.playlist.episodes')}</div>
        {episodeGroups.map(([group, vids]) => (
          <div key={group || '(root)'}>
            {group !== '' && (
              <button
                onClick={() => setCollapsedGroups((prev) => {
                  const next = new Set(prev);
                  if (next.has(group)) next.delete(group); else next.add(group);
                  return next;
                })}
                className="w-full flex items-center gap-1.5 px-2 py-1.5 text-xs font-semibold text-[#A1A1AA] hover:text-[#FAFAFA] hover:bg-[#27272A]/40"
              >
                {collapsedGroups.has(group) ? <ChevronRight size={13} /> : <ChevronDown size={13} />}
                <Folder size={12} className="text-[#8B5CF6]" />
                <span className="truncate">{group}</span>
                <span className="ml-auto text-[#71717A] font-normal">{vids.length}</span>
              </button>
            )}
            {(group === '' || !collapsedGroups.has(group)) && vids.map((v) => (
              <div key={v.id} className="relative">
                <button
                  onClick={() => loadEpisode(v)}
                  className={`w-full text-left py-2 border-l-2 transition-all text-sm ${group !== '' ? 'pl-7 pr-12' : 'pl-3 pr-12'} ${
                    v.id === video?.id
                      ? 'border-[#EC4899] bg-[#27272A]/70 text-[#FAFAFA]'
                      : 'border-transparent text-[#A1A1AA] hover:bg-[#27272A]/40'
                  }`}
                >
                  <div className="flex items-center gap-1.5 min-w-0">
                    {v.derushed && <Check size={12} className="text-emerald-400 shrink-0" strokeWidth={3} />}
                    <span className={`truncate font-medium ${v.derushed ? 'text-[#71717A]' : ''}`}>{v.display_name}</span>
                  </div>
                  <div className="text-xs text-[#71717A]">
                    {fmt(v.duration_ms)} · {t('derush.playlist.scenes', { n: v.scene_count })}
                    {(keptByVideo.get(v.id) ?? 0) > 0 && (
                      <span className="text-[#EC4899]"> · {t('derush.playlist.kept', { n: keptByVideo.get(v.id)! })}</span>
                    )}
                  </div>
                </button>
                <div className="absolute right-1 top-2 flex gap-0.5">
                  <button
                    title={v.derushed ? t('derush.playlist.markNotDerushed') : t('derush.playlist.markDerushed')}
                    onClick={() => api.setVideoDerushed(v.id, !v.derushed).then(reloadVideos).catch(() => {})}
                    className={`p-1 rounded hover:bg-[#27272A] ${v.derushed ? 'text-emerald-400' : 'text-[#71717A]/60 hover:text-emerald-400'}`}
                  >
                    <Check size={12} strokeWidth={3} />
                  </button>
                  <button
                    title={t('derush.playlist.moveToFolder')}
                    onClick={() => { setEditingVideo(v.id); setGroupInput(v.group_name ?? ''); }}
                    className="p-1 rounded text-[#71717A]/60 hover:text-[#FAFAFA] hover:bg-[#27272A]"
                  >
                    <FolderOpen size={12} />
                  </button>
                </div>
                {editingVideo === v.id && (
                  <div className={`pb-2 pr-3 ${group !== '' ? 'pl-7' : 'pl-3'}`}>
                    <input
                      autoFocus
                      list="derush-group-suggestions"
                      value={groupInput}
                      placeholder={t('derush.playlist.folderPlaceholder')}
                      onChange={(e) => setGroupInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') commitGroupMove(v.id);
                        if (e.key === 'Escape') setEditingVideo(null);
                      }}
                      onBlur={() => setEditingVideo(null)}
                      className="w-full bg-[#0F0F11] border border-[#8B5CF6] rounded px-2 py-1 text-xs text-[#FAFAFA] focus:outline-none"
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        ))}
        {videos.length === 0 && (
          <div className="px-3 py-4 text-xs text-[#71717A]">{t('derush.playlist.empty')}</div>
        )}
        <datalist id="derush-group-suggestions">
          {groupNames.map((g) => <option key={g} value={g} />)}
        </datalist>
      </div>

      {/* player column */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="relative flex-1 bg-black flex items-center justify-center min-h-0">
          {src && !playError ? (
            <video
              ref={videoRef}
              src={src}
              className="max-h-full max-w-full"
              muted={muted}
              preload="auto"
              onTimeUpdate={(e) => { if (!scrubbingRef.current) setCurrentMs(e.currentTarget.currentTime * 1000); }}
              onDurationChange={(e) => setDurationMs(e.currentTarget.duration * 1000)}
              onEnded={() => { if (autoNext) gotoEpisode(1); }}
              onError={() => setPlayError('decode')}
            />
          ) : (
            <div className="text-center text-[#A1A1AA] p-8">
              {loadingSrc ? (
                <div className="flex items-center gap-3"><Loader2 className="animate-spin" size={20} /> {t('derush.player.preparing')}</div>
              ) : playError === 'missing' && video ? (
                <div className="max-w-md">
                  <div className="text-lg font-semibold text-[#FAFAFA] mb-2">{t('derush.player.missingTitle')}</div>
                  <div className="text-sm mb-2">
                    {t('derush.player.missingBody')}
                  </div>
                  <div className="text-xs text-[#71717A]">
                    {t('derush.player.missingFixPrefix')} <span className="text-[#A1A1AA]">{t('derush.player.missingFixTarget')}</span> {t('derush.player.missingFixSuffix')}
                  </div>
                </div>
              ) : playError === 'decode' && video ? (
                <div className="max-w-md">
                  <div className="text-lg font-semibold text-[#FAFAFA] mb-2">{t('derush.player.decodeTitle')}</div>
                  <div className="text-sm mb-4">{t('derush.player.decodeBody')}</div>
                  {transcoding ? (
                    <div className="flex items-center justify-center gap-3 text-sm">
                      <Loader2 className="animate-spin" size={16} />
                      {t('derush.player.transcoding')} {lastEvent?.type === 'derush_prepare' && lastEvent.video_id === video.id ? `${lastEvent.percent ?? 0}%` : ''}
                    </div>
                  ) : (
                    <button
                      onClick={() => { setTranscoding(true); api.videoTranscode(video.id).catch(() => setTranscoding(false)); }}
                      className="px-4 py-2 rounded-lg bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white text-sm font-semibold"
                    >
                      {t('derush.player.generatePlayable')}
                    </button>
                  )}
                </div>
              ) : (
                <div className="flex items-center gap-3"><Clapperboard size={20} /> {t('derush.player.pickEpisode')}</div>
              )}
            </div>
          )}

          {/* keep flash */}
          {flash && (
            <div className={`absolute top-6 left-1/2 -translate-x-1/2 px-4 py-2 rounded-xl text-sm font-bold shadow-xl ${
              flash.kept ? 'bg-emerald-500/90 text-white' : 'bg-[#27272A]/90 text-[#FAFAFA]'
            }`}>
              {flash.text}
            </div>
          )}

          {/* shuttle indicator */}
          {vel !== 0 && (
            <div className="absolute bottom-4 right-4 px-3 py-1 rounded-lg bg-black/70 text-[#FAFAFA] text-sm font-mono flex items-center gap-2">
              {vel < 0 ? <Rewind size={14} /> : <FastForward size={14} />} {Math.abs(vel)}×
            </div>
          )}

          {/* current scene badge */}
          {currentScene && (
            <div className={`absolute top-4 left-4 px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 ${
              favIds.has(currentScene.id) ? 'bg-[#F59E0B]/90 text-black'
                : keptIds.has(currentScene.id) ? 'bg-[#EC4899]/90 text-white' : 'bg-black/70 text-[#A1A1AA]'
            }`}>
              {keptIds.has(currentScene.id) && <Star size={12} fill="currentColor" />}
              {favIds.has(currentScene.id) && `${t('derush.player.favBadge')} · `}
              {t('derush.player.sceneOf', { n: currentSceneIdx + 1, total: scenes.length })} · {fmt(currentScene.start_ms)}–{fmt(currentScene.end_ms)}
            </div>
          )}
        </div>

        {/* timeline: drag the blue playhead to scrub; Ctrl+wheel zooms, wheel pans */}
        <div
          ref={scrollRef}
          className="relative bg-[#0F0F11] border-t-2 border-[#27272A] overflow-x-auto overflow-y-hidden"
          style={{ scrollbarWidth: 'thin' }}
        >
        <div
          ref={stripRef}
          className={`relative h-10 select-none ${scrubbing ? 'cursor-grabbing' : 'cursor-pointer'}`}
          style={{ touchAction: 'none', width: `${zoom * 100}%`, minWidth: '100%' }}
          onPointerDown={onStripPointerDown}
          onPointerMove={onStripPointerMove}
          onPointerUp={endScrub}
          onPointerCancel={endScrub}
        >
          {durationMs > 0 && scenes.map((s, i) => {
            const left = (s.start_ms / durationMs) * 100;
            const widthPct = ((s.end_ms - s.start_ms) / durationMs) * 100;
            const kept = keptIds.has(s.id);
            const fav = favIds.has(s.id);
            const current = i === currentSceneIdx;
            const isSelected = selected.has(s.id);
            return (
              <div
                key={s.id}
                title={t('derush.strip.title', {
                  n: i + 1,
                  start: fmt(s.start_ms),
                  end: fmt(s.end_ms),
                  status: fav ? t('derush.strip.statusFavorite') : kept ? t('derush.strip.statusKept') : '',
                })}
                onClick={(e) => {
                  if (e.ctrlKey || e.metaKey) { e.stopPropagation(); toggleSelect(s.id); }
                }}
                className={`absolute top-[9px] bottom-1 rounded-sm transition-colors ${
                  fav ? 'bg-[#F59E0B]' : kept ? 'bg-[#EC4899]' : 'bg-[#3F3F46] hover:bg-[#52525B]'
                } ${isSelected ? 'ring-2 ring-white' : current ? 'ring-2 ring-[#8B5CF6]' : ''}`}
                // Gap and minimum width are in PIXELS, not % of the strip:
                // a %-based gap scales with zoom and turned into big black
                // bars at ×64, and a %-based min-width made short scenes
                // overlap their neighbours when zoomed in.
                style={{ left: `${left}%`, width: `calc(${widthPct}% - 1px)`, minWidth: '2px' }}
              />
            );
          })}
          {durationMs > 0 && (
            <div
              className="absolute top-0 bottom-0 pointer-events-none z-10"
              style={{ left: `${Math.min(100, (currentMs / durationMs) * 100)}%` }}
            >
              {/* Premiere-style playhead: blue handle + needle */}
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-0 h-0 border-l-[6px] border-r-[6px] border-t-[8px] border-l-transparent border-r-transparent border-t-[#3B82F6] drop-shadow" />
              <div className="absolute top-0 bottom-0 left-1/2 w-[2px] -translate-x-1/2 bg-[#3B82F6] shadow-[0_0_4px_rgba(59,130,246,0.8)]" />
            </div>
          )}
        </div>
        </div>

        {/* transport bar */}
        <div className="flex items-center gap-2 px-4 py-2.5 bg-[#18181B] border-t-2 border-[#27272A]">
          <button title={t('derush.transport.prevScene')} onClick={() => currentSceneIdx > 0 && seekTo(scenes[currentSceneIdx - 1].start_ms + 1)} className="p-2 rounded-lg text-[#A1A1AA] hover:bg-[#27272A]"><SkipBack size={16} /></button>
          <button title={t('derush.transport.slower')} onClick={() => shuttleStep(-1)} className={`p-2 rounded-lg hover:bg-[#27272A] ${vel < 0 ? 'text-[#EC4899]' : 'text-[#A1A1AA]'}`}><Rewind size={18} /></button>
          <button
            title={t('derush.transport.playPause')}
            onClick={() => applyShuttle(vel !== 0 ? 0 : 1)}
            className="p-2.5 rounded-xl bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white"
          >
            {vel === 0 ? <Play size={18} /> : <Pause size={18} />}
          </button>
          <button title={t('derush.transport.faster')} onClick={() => shuttleStep(1)} className={`p-2 rounded-lg hover:bg-[#27272A] ${vel > 1 ? 'text-[#EC4899]' : 'text-[#A1A1AA]'}`}><FastForward size={18} /></button>
          <button title={t('derush.transport.nextScene')} onClick={() => currentSceneIdx < scenes.length - 1 && seekTo(scenes[currentSceneIdx + 1].start_ms + 1)} className="p-2 rounded-lg text-[#A1A1AA] hover:bg-[#27272A]"><SkipForward size={16} /></button>

          <div className="font-mono text-xs text-[#A1A1AA] ml-2 w-28">{fmt(currentMs)} / {fmt(durationMs)}</div>

          <div className="flex items-center gap-0.5 ml-1" title={t('derush.transport.zoomTitle')}>
            <button onClick={() => zoomAround(1 / 1.5)} className="p-1.5 rounded-lg text-[#A1A1AA] hover:bg-[#27272A] disabled:opacity-30" disabled={zoom <= 1}>
              <ZoomOut size={14} />
            </button>
            <span className="text-[10px] font-mono text-[#71717A] w-8 text-center">{zoom < 10 ? zoom.toFixed(1) : Math.round(zoom)}×</span>
            <button onClick={() => zoomAround(1.5)} className="p-1.5 rounded-lg text-[#A1A1AA] hover:bg-[#27272A] disabled:opacity-30" disabled={zoom >= MAX_ZOOM}>
              <ZoomIn size={14} />
            </button>
            {zoom > 1 && (
              <button onClick={() => setZoom(1)} className="px-1.5 py-0.5 rounded text-[10px] text-[#71717A] hover:text-[#FAFAFA] hover:bg-[#27272A]">
                {t('derush.transport.fit')}
              </button>
            )}
          </div>

          <button title={t('derush.transport.mute')} onClick={() => setMuted((m) => !m)} className={`p-2 rounded-lg hover:bg-[#27272A] ${muted ? 'text-[#EC4899]' : 'text-[#A1A1AA]'}`}>
            {muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
          </button>

          <div className="flex-1" />

          {selected.size > 0 && (
            <div className="flex items-center gap-1.5 mr-2">
              <span className="text-xs text-[#F59E0B]">{t('derush.transport.selected', { n: selected.size })}</span>
              <button
                onClick={mergeSelection}
                disabled={!mergeable}
                title={mergeable ? t('derush.transport.mergeTitle') : t('derush.transport.mergeDisabledTitle')}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold bg-[#F59E0B] text-black disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Combine size={13} /> {t('derush.transport.merge')}
              </button>
              <button title={t('derush.transport.clearSelection')} onClick={() => setSelected(new Set())} className="p-1 text-[#71717A] hover:text-[#FAFAFA]"><X size={13} /></button>
            </div>
          )}

          <label className="flex items-center gap-1.5 text-xs text-[#71717A] mr-2 cursor-pointer">
            <input type="checkbox" checked={autoNext} onChange={(e) => setAutoNext(e.target.checked)} className="accent-[#8B5CF6]" />
            {t('derush.transport.autoNext')}
          </label>

          <select
            value={targetFolder}
            onChange={(e) => setTargetFolder(e.target.value === '' ? '' : Number(e.target.value))}
            title={t('derush.transport.folderTitle')}
            className="bg-[#0F0F11] border border-[#27272A] rounded-lg px-2 py-1.5 text-xs text-[#A1A1AA] focus:outline-none focus:border-[#8B5CF6] max-w-[140px]"
          >
            <option value="">{t('derush.transport.unfiled')}</option>
            {folders.map((f) => <option key={f.id} value={f.id}>→ {f.name}</option>)}
          </select>

          <button
            onClick={() => keepCurrent(false)}
            disabled={!currentScene}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-bold transition-all disabled:opacity-40 ${
              currentScene && keptIds.has(currentScene.id)
                ? 'bg-[#27272A] text-[#EC4899] border-2 border-[#EC4899]/50'
                : 'bg-gradient-to-r from-[#EC4899] to-[#8B5CF6] text-white'
            }`}
          >
            <Star size={15} fill={currentScene && keptIds.has(currentScene.id) ? 'currentColor' : 'none'} />
            {currentScene && keptIds.has(currentScene.id)
              ? t('derush.transport.keptRemove', { key: displayKey(keys.level_down) })
              : t('derush.transport.keepScene', { key: displayKey(keys.level_up) })}
          </button>
          <button
            onClick={() => keepCurrent(true)}
            disabled={!currentScene}
            title={t('derush.transport.favTitle', { key: displayKey(keys.level_up), altKey: displayKey(keys.toggle_favorite) })}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-xl text-sm font-bold transition-all disabled:opacity-40 ${
              currentScene && favIds.has(currentScene.id)
                ? 'bg-[#F59E0B] text-black'
                : 'bg-[#27272A] text-[#F59E0B] border-2 border-[#F59E0B]/50'
            }`}
          >
            <Star size={15} fill={currentScene && favIds.has(currentScene.id) ? 'currentColor' : 'none'} />
            {t('derush.transport.fav', { key: displayKey(keys.level_up) })}
          </button>

          <div className="text-xs text-[#71717A] ml-1">{t('derush.transport.keptInEp', { n: keptInEpisode })}</div>
        </div>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────── library ────
function toSceneResult(it: DerushItem): SceneResult {
  return {
    id: it.scene_id,
    video_id: it.video_id,
    video_display: it.video_display,
    scene_index: it.scene_index,
    start_ms: it.start_ms,
    end_ms: it.end_ms,
    score: 0,
    tags: [],
    proxy_path: it.has_proxy ? true : null,
  };
}

function LibraryView({ folders, items, lastEvent, onChanged, onOpenScene }: {
  folders: DerushFolder[];
  items: DerushItem[];
  lastEvent: ProgressEvent | null;
  onChanged: () => void;
  onOpenScene?: (scenes: SceneResult[], index: number) => void;
}) {
  const t = useT();
  const [filter, setFilter] = useState<'all' | 'root' | 'fav' | number>('all');
  const [renamingItem, setRenamingItem] = useState<number | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [newFolderOpen, setNewFolderOpen] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');
  const [renamingFolder, setRenamingFolder] = useState<number | null>(null);
  const [folderRename, setFolderRename] = useState('');
  const [hoverId, setHoverId] = useState<number | null>(null);
  const [exportDone, setExportDone] = useState<ProgressEvent | null>(null);

  const exporting = lastEvent?.type === 'derush_export' && !lastEvent.done;
  useEffect(() => {
    if (lastEvent?.type === 'derush_export' && lastEvent.done) {
      setExportDone(lastEvent);
      onChanged();
    }
  }, [lastEvent, onChanged]);

  const filtered = useMemo(() => {
    if (filter === 'all') return items;
    if (filter === 'root') return items.filter((i) => i.folder_ids.length === 0);
    if (filter === 'fav') return items.filter((i) => i.favorite);
    return items.filter((i) => i.folder_ids.includes(filter));
  }, [items, filter]);

  const rootCount = useMemo(() => items.filter((i) => i.folder_ids.length === 0).length, [items]);
  const favCount = useMemo(() => items.filter((i) => i.favorite).length, [items]);

  const commitItemRename = (item: DerushItem) => {
    const v = renameValue.trim();
    const p = v ? api.derushPatchItem(item.id, { custom_name: v }) : api.derushPatchItem(item.id, { clear_name: true });
    p.then(onChanged).catch(() => {});
    setRenamingItem(null);
  };

  const startExport = () => {
    const scope = filter === 'all' ? 'all' : filter === 'root' ? 'root' : filter === 'fav' ? 'favorites' : 'folder';
    setExportDone(null);
    api.derushExport(scope, typeof filter === 'number' ? filter : undefined).catch((e) => {
      setExportDone({ type: 'derush_export', done: true, ok: false, message: (e as Error).message });
    });
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* folder chips + export */}
      <div className="flex items-center gap-2 px-4 py-2.5 border-b-2 border-[#27272A] bg-[#18181B]/30 flex-wrap">
        <Chip active={filter === 'all'} onClick={() => setFilter('all')} icon={<Library size={13} />} label={t('derush.library.all', { n: items.length })} />
        <Chip active={filter === 'fav'} onClick={() => setFilter('fav')} icon={<Star size={13} fill="currentColor" />} label={t('derush.library.favorites', { n: favCount })} />
        <Chip active={filter === 'root'} onClick={() => setFilter('root')} icon={<Folder size={13} />} label={t('derush.library.unfiled', { n: rootCount })} />
        {folders.map((f) => (
          <div key={f.id} className="flex items-center">
            {renamingFolder === f.id ? (
              <input
                autoFocus
                value={folderRename}
                onChange={(e) => setFolderRename(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') { api.derushRenameFolder(f.id, folderRename).then(onChanged).catch(() => {}); setRenamingFolder(null); }
                  if (e.key === 'Escape') setRenamingFolder(null);
                }}
                onBlur={() => setRenamingFolder(null)}
                className="bg-[#0F0F11] border border-[#8B5CF6] rounded-lg px-2 py-1 text-xs text-[#FAFAFA] w-32 focus:outline-none"
              />
            ) : (
              <Chip
                active={filter === f.id}
                onClick={() => setFilter(f.id)}
                icon={filter === f.id ? <FolderOpen size={13} /> : <Folder size={13} />}
                label={`${f.name} (${f.item_count})`}
              />
            )}
            {filter === f.id && renamingFolder !== f.id && (
              <div className="flex items-center ml-0.5">
                <button title={t('derush.library.renameFolder')} onClick={() => { setRenamingFolder(f.id); setFolderRename(f.name); }} className="p-1 text-[#71717A] hover:text-[#FAFAFA]"><Pencil size={11} /></button>
                <button title={t('derush.library.deleteFolder')} onClick={() => { api.derushDeleteFolder(f.id).then(() => { setFilter('all'); onChanged(); }).catch(() => {}); }} className="p-1 text-[#71717A] hover:text-red-400"><Trash2 size={11} /></button>
              </div>
            )}
          </div>
        ))}
        {newFolderOpen ? (
          <input
            autoFocus
            value={newFolderName}
            placeholder={t('derush.library.folderNamePlaceholder')}
            onChange={(e) => setNewFolderName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && newFolderName.trim()) {
                api.derushCreateFolder(newFolderName.trim()).then(() => { setNewFolderName(''); setNewFolderOpen(false); onChanged(); }).catch(() => {});
              }
              if (e.key === 'Escape') setNewFolderOpen(false);
            }}
            onBlur={() => setNewFolderOpen(false)}
            className="bg-[#0F0F11] border border-[#8B5CF6] rounded-lg px-2 py-1 text-xs text-[#FAFAFA] w-32 focus:outline-none"
          />
        ) : (
          <button onClick={() => setNewFolderOpen(true)} className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs text-[#71717A] hover:text-[#FAFAFA] hover:bg-[#27272A] border border-dashed border-[#3F3F46]">
            <FolderPlus size={13} /> {t('derush.library.newFolder')}
          </button>
        )}

        <div className="flex-1" />

        {exporting ? (
          <div className="flex items-center gap-2 text-xs text-[#A1A1AA]">
            <Loader2 size={14} className="animate-spin text-[#EC4899]" />
            {t('derush.library.exporting', { current: lastEvent?.current ?? 0, total: lastEvent?.total ?? 0, message: lastEvent?.message ?? '' })}
            <div className="w-28 h-1.5 bg-[#27272A] rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-[#8B5CF6] to-[#EC4899]" style={{ width: `${lastEvent?.percent ?? 0}%` }} />
            </div>
          </div>
        ) : (
          <button
            onClick={startExport}
            disabled={filtered.length === 0}
            className="flex items-center gap-2 px-4 py-1.5 rounded-xl text-sm font-bold bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white disabled:opacity-40"
          >
            <Download size={15} />
            {filter === 'all'
              ? t('derush.library.exportAll', { n: items.length })
              : filtered.length > 1
                ? t('derush.library.exportMany', { n: filtered.length })
                : t('derush.library.exportOne')}
          </button>
        )}
      </div>

      {exportDone?.done && (
        <div className={`flex items-center gap-2 px-4 py-2 text-xs border-b border-[#27272A] ${exportDone.ok ? 'text-emerald-400' : 'text-amber-400'}`}>
          {exportDone.ok ? <Check size={14} /> : <X size={14} />}
          {exportDone.message ?? t('derush.library.exportFinished')}
          {exportDone.output && (
            <button onClick={() => api.revealOutput(exportDone.output!)} className="underline text-[#A1A1AA] hover:text-[#FAFAFA] flex items-center gap-1">
              {t('derush.library.openFolder')} <ChevronRight size={12} />
            </button>
          )}
          <button onClick={() => setExportDone(null)} className="ml-auto text-[#71717A] hover:text-[#FAFAFA]"><X size={13} /></button>
        </div>
      )}

      {/* items grid */}
      <div className="flex-1 overflow-y-auto p-4">
        {filtered.length === 0 ? (
          <div className="h-full flex items-center justify-center text-[#71717A] text-sm">
            {t('derush.library.empty')}
          </div>
        ) : (
          <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))' }}>
            {filtered.map((it, idx) => (
              <div key={it.id} className="group bg-[#18181B] border-2 border-[#27272A] hover:border-[#8B5CF6]/50 rounded-xl overflow-hidden transition-all">
                <div
                  className="relative aspect-video bg-black"
                  onMouseEnter={() => setHoverId(it.id)}
                  onMouseLeave={() => setHoverId(null)}
                >
                  {hoverId === it.id && it.has_proxy ? (
                    <video src={api.proxyUrl(it.scene_id)} autoPlay muted loop className="w-full h-full object-cover" />
                  ) : (
                    <img src={api.thumbnailUrl(it.scene_id)} alt="" className="w-full h-full object-cover" loading="lazy" />
                  )}
                  {onOpenScene && (
                    <button
                      title={t('derush.library.openInEditor')}
                      onClick={() => onOpenScene(filtered.map(toSceneResult), idx)}
                      className="absolute inset-0 flex items-center justify-center bg-black/45 opacity-0 hover:opacity-100 transition-opacity"
                    >
                      <Play size={30} className="text-white drop-shadow-lg" fill="currentColor" />
                    </button>
                  )}
                  <div className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded bg-black/70 text-[10px] font-mono text-[#FAFAFA] pointer-events-none">
                    {((it.end_ms - it.start_ms) / 1000).toFixed(1)}s
                  </div>
                  <button
                    title={it.favorite ? t('derush.library.unfavorite') : t('derush.library.favorite')}
                    onClick={() => api.derushPatchItem(it.id, { favorite: !it.favorite }).then(onChanged).catch(() => {})}
                    className={`absolute top-1 left-1 p-1 rounded-md transition-colors ${
                      it.favorite ? 'bg-[#F59E0B] text-black' : 'bg-black/60 text-white/60 hover:text-[#F59E0B]'
                    }`}
                  >
                    <Star size={12} fill={it.favorite ? 'currentColor' : 'none'} />
                  </button>
                </div>
                <div className="p-2.5">
                  {renamingItem === it.id ? (
                    <input
                      autoFocus
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') commitItemRename(it);
                        if (e.key === 'Escape') setRenamingItem(null);
                      }}
                      onBlur={() => commitItemRename(it)}
                      className="w-full bg-[#0F0F11] border border-[#8B5CF6] rounded px-1.5 py-1 text-xs text-[#FAFAFA] focus:outline-none"
                    />
                  ) : (
                    <div
                      className="text-xs font-semibold text-[#FAFAFA] truncate cursor-text"
                      title={t('derush.library.clickToRename')}
                      onClick={() => { setRenamingItem(it.id); setRenameValue(it.custom_name ?? ''); }}
                    >
                      {it.custom_name ?? t('derush.library.defaultName', { video: it.video_display, n: it.scene_index })}
                    </div>
                  )}
                  <div className="text-[10px] text-[#71717A] truncate mt-0.5">
                    {it.video_display} · {fmt(it.start_ms)}–{fmt(it.end_ms)}
                  </div>
                  <div className="flex items-center gap-1 mt-2">
                    <FolderPicker item={it} folders={folders} onChanged={onChanged} />
                    <button
                      title={t('common.rename')}
                      onClick={() => { setRenamingItem(it.id); setRenameValue(it.custom_name ?? ''); }}
                      className="p-1 text-[#71717A] hover:text-[#FAFAFA]"
                    ><Pencil size={12} /></button>
                    <button
                      title={t('derush.library.removeItem')}
                      onClick={() => api.derushDeleteItem(it.id).then(onChanged).catch(() => {})}
                      className="p-1 text-[#71717A] hover:text-red-400"
                    ><Trash2 size={12} /></button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function FolderPicker({ item, folders, onChanged }: {
  item: DerushItem;
  folders: DerushFolder[];
  onChanged: () => void;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const names = folders.filter((f) => item.folder_ids.includes(f.id)).map((f) => f.name);
  const label = names.length === 0 ? t('derush.picker.unfiled') : names.length === 1 ? names[0] : t('derush.picker.folders', { n: names.length });

  const toggle = (fid: number) => {
    const next = item.folder_ids.includes(fid)
      ? item.folder_ids.filter((x) => x !== fid)
      : [...item.folder_ids, fid];
    api.derushPatchItem(item.id, { folder_ids: next }).then(onChanged).catch(() => {});
  };

  return (
    <div className="relative flex-1 min-w-0">
      <button
        onClick={() => setOpen((o) => !o)}
        title={names.length ? names.join(', ') : t('derush.picker.pickTitle')}
        className="w-full flex items-center justify-between gap-1 bg-[#0F0F11] border border-[#27272A] rounded px-1.5 py-0.5 text-[10px] text-[#A1A1AA] hover:border-[#8B5CF6] focus:outline-none"
      >
        <span className="truncate">{label}</span>
        <ChevronDown size={10} className="shrink-0" />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute z-20 bottom-full mb-1 left-0 w-44 bg-[#18181B] border border-[#3F3F46] rounded-lg shadow-2xl p-1.5 max-h-44 overflow-y-auto">
            {folders.length === 0 && (
              <div className="text-[10px] text-[#71717A] px-1 py-1">{t('derush.picker.empty')}</div>
            )}
            {folders.map((f) => (
              <label key={f.id} className="flex items-center gap-2 px-1.5 py-1 rounded hover:bg-[#27272A] cursor-pointer text-xs text-[#FAFAFA]">
                <input
                  type="checkbox"
                  checked={item.folder_ids.includes(f.id)}
                  onChange={() => toggle(f.id)}
                  className="accent-[#8B5CF6]"
                />
                <span className="truncate">{f.name}</span>
              </label>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function Chip({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
        active ? 'bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white' : 'text-[#A1A1AA] hover:bg-[#27272A] border border-[#27272A]'
      }`}
    >
      {icon} {label}
    </button>
  );
}
