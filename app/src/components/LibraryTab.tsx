// Library tab — the top-level home for importing episodes, indexing them,
// organising them into folders and managing databases. This is the first
// screen a newcomer needs, so the import flow is front and centre; database
// management is tucked behind an "advanced" disclosure.
import { useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, Folder, FolderOpen, FolderPlus, GripVertical, HardDrive, Layers, Link, Loader2, Pencil, Plus, RefreshCw, Target, Trash2, Zap } from 'lucide-react';
import { api } from '../api/client';
import type { IndexQueueItem, VideoSummary } from '../api/types';
import { useT } from '../i18n';

const VIDEO_FILE_FILTERS = [{
  name: 'Video',
  extensions: ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm', 'ts', 'm2ts', 'mts', 'mpg', 'mpeg', 'vob', 'm4v', 'f4v', '3gp', 'ogv', 'mxf'],
}];

export default function LibraryTab() {
  const t = useT();
  const [databases, setDatabases] = useState<{ path: string; scenes: number; videos: number; size_kb: number; primary: boolean }[]>([]);
  const [queue, setQueue] = useState<IndexQueueItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [importGroup, setImportGroup] = useState('');
  const [relinkDb, setRelinkDb] = useState<string | null>(null);
  const [showDbs, setShowDbs] = useState(false);

  async function refresh() {
    const [dbs, q] = await Promise.all([api.listDatabases(), api.queue()]);
    setDatabases(dbs.databases);
    setQueue(q.items);
  }

  useEffect(() => { refresh().catch(() => {}); }, []);

  const totals = useMemo(() => databases.reduce(
    (acc, db) => ({ videos: acc.videos + db.videos, scenes: acc.scenes + db.scenes }),
    { videos: 0, scenes: 0 },
  ), [databases]);

  async function addFolder() {
    const paths = await window.amvBridge?.openFileDialog({ directory: true, multi: true });
    if (!paths || paths.length === 0) return;
    await api.enqueue(paths.map((p) => ({ path: p, recursive: true, group: importGroup.trim() || null })));
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
    // Library imported from another machine? Open the relink panel right away.
    try {
      const v = await api.verifyDatabase(paths[0]);
      if (v.missing.length > 0) setRelinkDb(paths[0]);
    } catch {}
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
    await api.enqueue(paths.map((p) => ({ path: p, recursive: true, group: importGroup.trim() || null })));
    refresh();
  };

  const libraryEmpty = totals.videos === 0 && queue.length === 0;

  return (
    <div className="h-full overflow-auto relative">
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[500px] h-[500px] bg-[#F59E0B]/5 rounded-full blur-[120px] pointer-events-none"></div>

      <div className="max-w-5xl mx-auto p-8 relative z-10 flex flex-col gap-6">
        <div>
          <h2 className="text-3xl font-bold bg-gradient-to-r from-[#FAFAFA] to-[#A1A1AA] bg-clip-text text-transparent">{t('library.title')}</h2>
          <div className="h-1 w-20 bg-gradient-to-r from-[#F59E0B] to-[#EC4899] rounded-full mt-2 mb-2"></div>
          <div className="text-sm text-[#A1A1AA]">{t('library.subtitle')}</div>
        </div>

        {/* ── Import card ─────────────────────────────────────────────── */}
        <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            className={`rounded-xl border-2 border-dashed transition-colors flex flex-col items-center justify-center text-center py-10 px-6 ${
              dragOver ? 'border-[#F59E0B] bg-[#F59E0B]/10' : 'border-[#3F3F46] bg-[#0F0F11]/50'
            }`}
          >
            <FolderPlus size={36} className={dragOver ? 'text-[#F59E0B]' : 'text-[#71717A]'} />
            <div className="text-[#FAFAFA] font-semibold mt-3">{t('library.drop.big')}</div>
            <div className="text-xs text-[#71717A] my-2">{t('library.drop.or')}</div>
            <button onClick={addFolder} className="bg-gradient-to-r from-[#F59E0B] to-[#EC4899] text-[#FAFAFA] px-6 py-2.5 rounded-lg text-sm font-semibold hover:shadow-lg hover:shadow-[#F59E0B]/30 transition-all flex items-center gap-2">
              <Plus size={14} /> {t('settings.db.addFolder')}
            </button>
          </div>

          <div className="flex items-center gap-3 mt-4 flex-wrap">
            <label className="text-xs text-[#71717A]">{t('library.group.label')}</label>
            <input
              type="text"
              value={importGroup}
              onChange={(e) => setImportGroup(e.target.value)}
              placeholder={t('settings.db.group.placeholder')}
              title={t('settings.db.group.tooltip')}
              className="bg-[#0F0F11] border-2 border-[#27272A] focus:border-[#F59E0B] rounded-lg px-3 py-2 text-sm text-[#FAFAFA] w-56 focus:outline-none"
            />
          </div>

          {queue.length > 0 && (
            <div className="mt-4 border-t-2 border-[#27272A] pt-4">
              <div className="text-sm font-semibold text-[#FAFAFA] mb-2">{t('library.queue.title')}</div>
              <div className="space-y-1 max-h-44 overflow-auto mb-4">
                {queue.map((q) => (
                  <div key={q.id} className="text-xs font-mono text-[#A1A1AA] flex items-center gap-2 py-1 px-2 hover:bg-[#27272A]/50 rounded">
                    <span className="text-[#F59E0B]">{q.is_directory ? '📁' : '🎞'}</span>
                    <span className="truncate flex-1">{q.path}</span>
                    {q.group_name && (
                      <span className="px-1.5 py-0.5 rounded bg-[#F59E0B]/20 text-[#FBBF24] text-[10px] shrink-0">→ {q.group_name}</span>
                    )}
                  </div>
                ))}
              </div>
              <div className="flex items-center gap-3 flex-wrap">
                <button
                  disabled={busy}
                  onClick={() => startIndex(['tag', 'embed'])}
                  className="bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-[#FAFAFA] px-6 py-2.5 rounded-lg text-sm font-semibold disabled:opacity-30 flex items-center gap-2"
                >
                  {busy ? <Loader2 size={14} className="animate-spin" /> : <Zap size={14} />} {t('library.index.start', { n: queue.length })}
                </button>
                <span className="text-xs text-[#71717A] flex-1 min-w-[200px]">{t('library.index.explain')}</span>
              </div>
              <div className="text-xs text-[#71717A] mt-3">
                {t('library.index.advanced')}
                <button disabled={busy} onClick={() => startIndex(['tag'])} title={t('settings.db.tagsOnly.tooltip')} className="underline decoration-dotted hover:text-[#FAFAFA] disabled:opacity-40 mx-1 inline-flex items-center gap-1">
                  <Target size={11} /> {t('settings.db.tagsOnly')}
                </button>
                ·
                <button disabled={busy} onClick={() => startIndex(['embed'])} title={t('settings.db.embedsOnly.tooltip')} className="underline decoration-dotted hover:text-[#FAFAFA] disabled:opacity-40 mx-1 inline-flex items-center gap-1">
                  <Layers size={11} /> {t('settings.db.embedsOnly')}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* ── Episodes ────────────────────────────────────────────────── */}
        {libraryEmpty ? (
          <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-8 text-center">
            <div className="text-[#FAFAFA] font-semibold mb-1">{t('library.empty.title')}</div>
            <div className="text-sm text-[#71717A]">{t('library.empty.body')}</div>
          </div>
        ) : (
          <>
            <div className="flex items-center justify-between -mb-2">
              <div className="text-sm font-semibold text-[#FAFAFA]">{t('library.videos.title')}</div>
              <div className="text-xs text-[#71717A]">{t('library.videos.count', { videos: totals.videos, scenes: totals.scenes })}</div>
            </div>
            <EpisodeFoldersPanel />
          </>
        )}

        {/* ── Databases (advanced) ────────────────────────────────────── */}
        <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
          <button onClick={() => setShowDbs((v) => !v)} className="w-full flex items-center gap-2 text-left">
            {showDbs ? <ChevronDown size={16} className="text-[#71717A]" /> : <ChevronRight size={16} className="text-[#71717A]" />}
            <HardDrive size={14} className="text-[#8B5CF6]" />
            <span className="text-[#FAFAFA] font-semibold flex-1">{t('library.db.title')}</span>
            <span className="text-xs text-[#71717A]">{databases.length}</span>
          </button>
          {showDbs && (
            <div className="mt-4">
              <div className="text-xs text-[#71717A] mb-4">{t('library.db.hint')}</div>
              <div className="flex gap-2 mb-4">
                <button onClick={addDb} className="bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-4 py-2 rounded-lg text-sm flex items-center gap-2"><FolderOpen size={14} /> {t('settings.db.openExisting')}</button>
                <button onClick={newDb} className="bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-4 py-2 rounded-lg text-sm flex items-center gap-2"><Plus size={14} /> {t('settings.db.createNew')}</button>
              </div>
              <div className="text-[#71717A] text-xs mb-2">{t('settings.db.active.hint')}</div>
              {databases.length === 0 ? (
                <div className="text-[#71717A] text-sm">{t('settings.db.none')}</div>
              ) : (
                <div className="space-y-2">
                  {databases.map((db) => (
                    <DbRow key={db.path} db={db} onChange={refresh} onRelink={() => setRelinkDb(db.path)} />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {relinkDb && <RelinkPanel dbPath={relinkDb} onClose={() => setRelinkDb(null)} onChanged={refresh} />}
    </div>
  );
}

function EpisodeFoldersPanel() {
  const t = useT();
  const [videos, setVideos] = useState<VideoSummary[]>([]);
  const [draftFolders, setDraftFolders] = useState<string[]>([]);
  const [newName, setNewName] = useState('');
  const [renaming, setRenaming] = useState<string | null>(null);
  const [renameVal, setRenameVal] = useState('');
  const [dragOverGroup, setDragOverGroup] = useState<string | null>(null);   // '' = Ungrouped box

  const refresh = () => api.listVideos().then((r) => setVideos(r.videos)).catch(() => {});
  useEffect(() => { refresh(); }, []);

  // Folders are derived from the episodes themselves + local drafts for
  // freshly-created (still empty) ones. An empty folder only becomes
  // persistent once an episode is dropped into it.
  const groups = useMemo(() => {
    const m = new Map<string, VideoSummary[]>();
    for (const g of draftFolders) m.set(g, []);
    for (const v of videos) {
      const g = (v.group_name ?? '').trim();
      if (!g) continue;
      if (!m.has(g)) m.set(g, []);
      m.get(g)!.push(v);
    }
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [videos, draftFolders]);
  const ungrouped = useMemo(() => videos.filter((v) => !(v.group_name ?? '').trim()), [videos]);

  const dropHandlers = (group: string) => ({
    onDragOver: (e: React.DragEvent) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; setDragOverGroup(group); },
    onDragLeave: () => setDragOverGroup((cur) => (cur === group ? null : cur)),
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOverGroup(null);
      const id = Number(e.dataTransfer.getData('application/x-amv-video'));
      if (!id) return;
      api.setVideoGroup(id, group || null).then(refresh).catch(() => {});
    },
  });

  const createFolder = () => {
    const name = newName.trim();
    if (!name) return;
    setDraftFolders((d) => (d.includes(name) || groups.some(([g]) => g === name) ? d : [...d, name]));
    setNewName('');
  };

  const commitRename = (oldName: string) => {
    const name = renameVal.trim();
    setRenaming(null);
    if (!name || name === oldName) return;
    api.renameVideoGroup(oldName, name).then(() => {
      setDraftFolders((d) => d.map((x) => (x === oldName ? name : x)));
      refresh();
    }).catch(() => {});
  };

  const dissolve = (name: string) => {
    api.renameVideoGroup(name, null).then(() => {
      setDraftFolders((d) => d.filter((x) => x !== name));
      refresh();
    }).catch(() => {});
  };

  const EpisodeChip = ({ v }: { v: VideoSummary }) => (
    <div
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData('application/x-amv-video', String(v.id));
        e.dataTransfer.effectAllowed = 'move';
      }}
      title={t('settings.folders.chipTitle', { path: v.filepath })}
      className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-[#0F0F11] border border-[#27272A] text-xs text-[#A1A1AA] cursor-grab active:cursor-grabbing hover:border-[#8B5CF6]/60 hover:text-[#FAFAFA] select-none"
    >
      <GripVertical size={10} className="text-[#3F3F46] shrink-0" />
      <span className="truncate max-w-[240px]">{v.display_name}</span>
    </div>
  );

  return (
    <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
      <div className="flex items-center justify-between mb-1">
        <div>
          <div className="text-[#FAFAFA] font-semibold">{t('settings.folders.title')}</div>
          <div className="text-[#71717A] text-sm">
            {t('settings.folders.desc')}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') createFolder(); }}
            placeholder={t('settings.folders.newPlaceholder')}
            className="bg-[#0F0F11] border-2 border-[#27272A] focus:border-[#8B5CF6] rounded-lg px-3 py-2 text-sm text-[#FAFAFA] w-44 focus:outline-none"
          />
          <button onClick={createFolder} disabled={!newName.trim()} className="flex items-center gap-2 bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-3 py-2 rounded-lg text-sm disabled:opacity-40">
            <Plus size={14} /> {t('common.create')}
          </button>
        </div>
      </div>

      <div className="grid gap-3 mt-4" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))' }}>
        {groups.map(([name, vids]) => (
          <div
            key={name}
            {...dropHandlers(name)}
            className={`rounded-xl border-2 p-3 min-h-[90px] transition-colors ${
              dragOverGroup === name ? 'border-[#8B5CF6] bg-[#8B5CF6]/10' : 'border-[#27272A] bg-[#0F0F11]/50'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <FolderOpen size={14} className="text-[#8B5CF6] shrink-0" />
              {renaming === name ? (
                <input
                  autoFocus
                  value={renameVal}
                  onChange={(e) => setRenameVal(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') commitRename(name);
                    if (e.key === 'Escape') setRenaming(null);
                  }}
                  onBlur={() => commitRename(name)}
                  className="flex-1 bg-[#0F0F11] border border-[#8B5CF6] rounded px-2 py-0.5 text-sm text-[#FAFAFA] focus:outline-none"
                />
              ) : (
                <span className="flex-1 text-sm font-semibold text-[#FAFAFA] truncate">{name}</span>
              )}
              <span className="text-xs text-[#71717A]">{vids.length}</span>
              <button title={t('settings.folders.renameTooltip')} onClick={() => { setRenaming(name); setRenameVal(name); }} className="p-1 text-[#71717A] hover:text-[#FAFAFA]"><Pencil size={12} /></button>
              <button title={t('settings.folders.dissolveTooltip')} onClick={() => dissolve(name)} className="p-1 text-[#71717A] hover:text-red-400"><Trash2 size={12} /></button>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {vids.length === 0 ? (
                <div className="text-xs text-[#71717A] border-2 border-dashed border-[#27272A] rounded-lg px-3 py-2 w-full text-center">
                  {t('settings.folders.dropHere')}
                </div>
              ) : (
                vids.map((v) => <EpisodeChip key={v.id} v={v} />)
              )}
            </div>
          </div>
        ))}

        <div
          {...dropHandlers('')}
          className={`rounded-xl border-2 p-3 min-h-[90px] transition-colors ${
            dragOverGroup === '' ? 'border-[#EC4899] bg-[#EC4899]/10' : 'border-dashed border-[#27272A] bg-transparent'
          }`}
        >
          <div className="flex items-center gap-2 mb-2">
            <Folder size={14} className="text-[#71717A] shrink-0" />
            <span className="flex-1 text-sm font-semibold text-[#A1A1AA]">{t('settings.folders.ungrouped')}</span>
            <span className="text-xs text-[#71717A]">{ungrouped.length}</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {ungrouped.length === 0 ? (
              <div className="text-xs text-[#71717A] w-full text-center py-2">{t('settings.folders.allFiled')}</div>
            ) : (
              ungrouped.map((v) => <EpisodeChip key={v.id} v={v} />)
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function DbRow({ db, onChange, onRelink }: { db: { path: string; scenes: number; videos: number; size_kb: number; primary: boolean }; onChange: () => Promise<void> | void; onRelink: () => void }) {
  const t = useT();
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
      setMaintMsg(t('settings.db.row.removed', { n: r.removed }));
      onChange();
    } catch (e) {
      setMaintMsg(t('common.error', { message: (e as Error).message }));
    } finally { setBusy(null); }
  }

  async function verify() {
    setBusy('verify'); setMaintMsg(null);
    try {
      const r = await api.verifyDatabase(db.path);
      setMaintMsg(r.missing.length === 0 ? t('settings.db.row.allPresent') : t('settings.db.row.missing', { n: r.missing.length }));
    } catch (e) {
      setMaintMsg(t('common.error', { message: (e as Error).message }));
    } finally { setBusy(null); }
  }

  async function remove() {
    if (!confirm(t('settings.db.row.confirmRemove', { path: db.path }))) return;
    await api.removeDatabase(db.path);
    onChange();
  }

  return (
    <div className={`p-3 rounded-lg border-2 transition-all ${db.primary ? 'border-[#8B5CF6] bg-[#8B5CF6]/10' : 'border-[#27272A] hover:border-[#3f3f46]'}`}>
      <div className="flex items-center justify-between gap-3">
        <button onClick={setPrimary} className="flex-1 min-w-0 text-left flex items-center gap-3">
          <span className="font-mono text-sm truncate flex-1">{db.path}</span>
          <div className="text-xs text-[#71717A] font-mono whitespace-nowrap">{t('settings.db.row.stats', { videos: db.videos, scenes: db.scenes, size: (db.size_kb / 1024).toFixed(1) })}</div>
          {db.primary && <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#8B5CF6]/20 text-[#8B5CF6] font-bold uppercase">{t('settings.db.row.primary')}</span>}
        </button>
        <div className="flex items-center gap-1">
          <button onClick={onRelink} disabled={!!busy} title={t('settings.db.row.relinkTooltip')} className="p-1.5 text-[#71717A] hover:text-[#8B5CF6] hover:bg-[#27272A] rounded">
            <Link size={14} />
          </button>
          <button onClick={verify} disabled={!!busy} title={t('settings.db.row.verifyTooltip')} className="p-1.5 text-[#71717A] hover:text-[#8B5CF6] hover:bg-[#27272A] rounded">
            {busy === 'verify' ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
          </button>
          <button onClick={cleanup} disabled={!!busy} title={t('settings.db.row.cleanupTooltip')} className="p-1.5 text-[#71717A] hover:text-[#EC4899] hover:bg-[#27272A] rounded">
            {busy === 'cleanup' ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
          </button>
          <button onClick={remove} title={t('settings.db.row.removeTooltip')} className="p-1.5 text-[#71717A] hover:text-red-400 hover:bg-[#27272A] rounded">
            <Trash2 size={14} />
          </button>
        </div>
      </div>
      {maintMsg && <div className="text-xs text-[#A1A1AA] mt-2 font-mono">{maintMsg}</div>}
    </div>
  );
}

function RelinkPanel({ dbPath, onClose, onChanged }: { dbPath: string; onClose: () => void; onChanged: () => void }) {
  const t = useT();
  const [missing, setMissing] = useState<{ id: number; filepath: string }[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const refresh = () => api.verifyDatabase(dbPath).then((r) => setMissing(r.missing)).catch(() => setMissing([]));
  useEffect(() => { refresh(); }, [dbPath]);

  // Group missing videos by their old parent folder so a moved season/subfolder
  // can be relinked on its own.
  const groups = useMemo(() => {
    const m = new Map<string, { id: number; filepath: string }[]>();
    for (const v of missing ?? []) {
      const parent = v.filepath.replace(/[\\/][^\\/]*$/, '');
      if (!m.has(parent)) m.set(parent, []);
      m.get(parent)!.push(v);
    }
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [missing]);

  function reportFolderResult(r: { relinked: number; ambiguous: string[]; still_missing: string[] }) {
    let out = t('settings.db.row.relinkResult', { n: r.relinked, left: r.still_missing.length });
    if (r.ambiguous.length > 0) out += t('settings.db.row.relinkAmbiguous', { a: r.ambiguous.length });
    setMsg(out);
  }

  async function relinkFromFolder(videoIds?: number[]) {
    const dirs = await window.amvBridge?.openFileDialog({ directory: true, multi: true });
    if (!dirs || dirs.length === 0) return;
    setBusy(true); setMsg(null);
    try {
      const r = await api.relinkFolder({ db_path: dbPath, search_dirs: dirs, video_ids: videoIds });
      reportFolderResult(r);
      await refresh();
      onChanged();
    } catch (e) {
      setMsg(t('common.error', { message: (e as Error).message }));
    } finally { setBusy(false); }
  }

  async function relinkOne(videoId: number) {
    const files = await window.amvBridge?.openFileDialog({ directory: false, filters: VIDEO_FILE_FILTERS });
    if (!files || !files[0]) return;
    setBusy(true); setMsg(null);
    try {
      const r = await api.relinkVideo({ db_path: dbPath, video_id: videoId, new_filepath: files[0] });
      if (!r.ok) setMsg(t('settings.relink.oneFailed', { message: r.error ?? '?' }));
      await refresh();
      onChanged();
    } catch (e) {
      setMsg(t('common.error', { message: (e as Error).message }));
    } finally { setBusy(false); }
  }

  const lastSegment = (p: string) => p.split(/[\\/]/).pop() || p;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-[60] p-8" onClick={onClose}>
      <div className="bg-gradient-to-br from-[#18181B] via-[#0F0F11] to-[#18181B] border-2 border-[#8B5CF6]/40 rounded-2xl p-6 max-w-2xl w-full max-h-[80vh] flex flex-col shadow-2xl shadow-[#8B5CF6]/30" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between mb-1">
          <div>
            <div className="text-xl font-bold text-[#FAFAFA] flex items-center gap-2"><Link size={18} className="text-[#8B5CF6]" /> {t('settings.relink.title')}</div>
            <div className="text-sm text-[#A1A1AA] mt-1">
              {missing === null ? t('common.loading') : missing.length === 0 ? t('settings.relink.none') : t('settings.relink.missingCount', { n: missing.length })}
            </div>
          </div>
          <button onClick={onClose} className="text-[#71717A] hover:text-[#FAFAFA] text-sm px-3 py-1.5 rounded-lg bg-[#27272A] hover:bg-[#3f3f46]">{t('common.close')}</button>
        </div>

        {missing !== null && missing.length > 0 && (
          <>
            <div className="text-xs text-[#71717A] mb-4">{t('settings.relink.hint')}</div>

            <button
              onClick={() => relinkFromFolder(undefined)}
              disabled={busy}
              className="mb-4 bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-[#FAFAFA] px-5 py-2.5 rounded-lg text-sm font-semibold disabled:opacity-40 flex items-center justify-center gap-2"
            >
              {busy ? <Loader2 size={14} className="animate-spin" /> : <FolderOpen size={14} />} {t('settings.relink.all')}
            </button>

            <div className="overflow-auto flex-1 space-y-4 pr-1">
              {groups.length > 1 && (
                <div className="space-y-2">
                  {groups.map(([parent, vids]) => (
                    <div key={parent} className="flex items-center gap-3 bg-[#0F0F11] border-2 border-[#27272A] rounded-lg px-4 py-2.5">
                      <FolderOpen size={14} className="text-[#8B5CF6] shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-semibold text-[#FAFAFA] truncate">{lastSegment(parent)}</div>
                        <div className="text-[10px] text-[#71717A] font-mono truncate" title={parent}>{parent}</div>
                      </div>
                      <span className="text-xs text-[#71717A] shrink-0">{vids.length}</span>
                      <button
                        onClick={() => relinkFromFolder(vids.map((v) => v.id))}
                        disabled={busy}
                        className="shrink-0 bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-3 py-1.5 rounded-lg text-xs disabled:opacity-40"
                      >
                        {t('settings.relink.groupPick')}
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <div>
                <div className="text-xs font-semibold text-[#A1A1AA] mb-2">{t('settings.relink.remaining')}</div>
                <div className="space-y-1">
                  {missing.map((v) => (
                    <div key={v.id} className="flex items-center gap-3 px-3 py-1.5 rounded-lg hover:bg-[#27272A]/50">
                      <span className="text-xs font-mono text-[#A1A1AA] truncate flex-1" title={v.filepath}>{lastSegment(v.filepath)}</span>
                      <button
                        onClick={() => relinkOne(v.id)}
                        disabled={busy}
                        className="shrink-0 bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-3 py-1 rounded-lg text-xs disabled:opacity-40"
                      >
                        {t('settings.relink.filePick')}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}

        {msg && <div className="mt-3 text-xs text-[#A1A1AA] font-mono">{msg}</div>}
      </div>
    </div>
  );
}
