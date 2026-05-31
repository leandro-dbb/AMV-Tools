import { useEffect, useState } from 'react';
import { AlertTriangle, ChevronDown, Database, Zap, Square } from 'lucide-react';
import SearchTab from './components/SearchTab';
import TagsTab from './components/TagsTab';
import SettingsTab from './components/SettingsTab';
import MiniEditor from './components/MiniEditor';
import OnboardingScreen from './components/OnboardingScreen';
import BootstrapScreen from './components/BootstrapScreen';
import TutorialOverlay from './components/TutorialOverlay';
import { api } from './api/client';
import type { AppSettings, ProgressEvent, SceneResult, SystemStatus } from './api/types';

type Tab = 'search' | 'tags' | 'settings';

interface BootstrapLine { line: string; stream: 'stdout' | 'stderr' }

const TUTORIAL_KEY = 'amv-tutorial-shown';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('search');
  const [editor, setEditor] = useState<{ scenes: SceneResult[]; index: number } | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [indexedCount, setIndexedCount] = useState<number>(0);
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [showTutorial, setShowTutorial] = useState(false);
  const [bootstrapPhase, setBootstrapPhase] = useState<'starting' | 'ready' | 'error'>(() => {
    // If we're not in Electron (e.g. running Vite alone), skip bootstrap.
    return window.amvBridge ? 'starting' : 'ready';
  });
  const [bootstrapLines, setBootstrapLines] = useState<BootstrapLine[]>([]);
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);

  // Wire the IPC bridge that streams sidecar stdout/stderr until the backend
  // port is open, then flip into the regular status-polling flow.
  useEffect(() => {
    const bridge = window.amvBridge;
    if (!bridge) return;

    // Resume from whatever state main.ts already reached before this effect ran.
    bridge.getBootstrapState?.().then((s) => {
      if (s.phase === 'ready' && s.port > 0) {
        (window as any).__AMV_API_PORT__ = s.port;
        setBootstrapPhase('ready');
      } else if (s.phase === 'error') {
        setBootstrapPhase('error');
      }
    }).catch(() => {});

    const offLine = bridge.onBootstrap?.((evt) => {
      setBootstrapLines((prev) => {
        const next = prev.concat(evt);
        return next.length > 500 ? next.slice(next.length - 500) : next;
      });
    });
    const offReady = bridge.onBootstrapReady?.((port) => {
      (window as any).__AMV_API_PORT__ = port;
      setBootstrapPhase('ready');
    });
    const offError = bridge.onBootstrapError?.((message) => {
      setBootstrapError(message);
      setBootstrapPhase('error');
    });
    return () => { offLine?.(); offReady?.(); offError?.(); };
  }, []);

  // Poll backend status (only once the sidecar has opened its port)
  useEffect(() => {
    if (bootstrapPhase !== 'ready') return;
    let active = true;
    const tick = async () => {
      try {
        const s = await api.status();
        if (active) { setStatus(s); setStatusError(null); }
      } catch (err) {
        if (active) setStatusError((err as Error).message);
      }
    };
    tick();
    const id = setInterval(tick, 4000);
    return () => { active = false; clearInterval(id); };
  }, [bootstrapPhase]);

  // Settings (for hover_delay_ms etc.)
  useEffect(() => {
    if (!status?.ready) return;
    api.getSettings().then(setSettings).catch(() => {});
  }, [status?.ready]);

  // WebSocket progress
  useEffect(() => {
    if (!status?.ready) return;
    let ws: WebSocket | null = null;
    let alive = true;
    const open = () => {
      try {
        ws = api.progressSocket();
        ws.onmessage = (msg) => {
          try {
            const evt = JSON.parse(msg.data) as ProgressEvent;
            if (evt.type !== 'install') setProgress(evt);
          } catch {}
        };
        ws.onclose = () => {
          if (alive) setTimeout(open, 1500);
        };
      } catch {}
    };
    open();
    return () => { alive = false; ws?.close(); };
  }, [status?.ready]);

  // Tutorial on first ready
  useEffect(() => {
    if (status?.ready && !localStorage.getItem(TUTORIAL_KEY)) {
      setShowTutorial(true);
    }
  }, [status?.ready]);

  // Indexed count tracker
  useEffect(() => {
    if (!status?.ready) return;
    api.listVideos().then((r) => {
      const total = r.videos.reduce((acc, v) => acc + (v.scene_count || 0), 0);
      setIndexedCount(total);
    }).catch(() => {});
  }, [status?.ready, progress?.type, progress?.percent]);

  if (bootstrapPhase !== 'ready') {
    return <BootstrapScreen lines={bootstrapLines} error={bootstrapError} />;
  }

  if (statusError && !status) {
    return (
      <div className="w-screen h-screen bg-[#0F0F11] flex items-center justify-center text-[#FAFAFA]">
        <div className="max-w-md text-center">
          <div className="text-2xl font-semibold mb-3">Backend unreachable</div>
          <div className="text-[#A1A1AA] text-sm">The Python sidecar didn't respond. If you launched the app outside Electron, start the backend manually:</div>
          <pre className="mt-4 bg-[#18181B] border border-[#27272A] p-3 rounded-lg text-xs text-left text-[#A1A1AA]">uv run --extra cu130 backend/server.py --port 8731</pre>
        </div>
      </div>
    );
  }

  if (status?.setup_required) {
    return <OnboardingScreen onDone={() => api.status().then(setStatus)} />;
  }

  const hoverDelay = settings?.interface.hover_delay_ms ?? 200;

  return (
    <div className="w-screen h-screen bg-gradient-to-br from-[#0F0F11] via-[#18181B] to-[#0F0F11] text-[#FAFAFA] flex flex-col overflow-hidden" style={{ fontFamily: 'Inter, Space Grotesk, system-ui, sans-serif' }}>
      <div className="bg-gradient-to-r from-[#18181B] via-[#0F0F11] to-[#18181B] border-b-2 border-[#27272A] px-6 py-4 flex items-center justify-between backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] rounded-lg blur-lg opacity-50"></div>
            <div className="relative bg-gradient-to-br from-[#8B5CF6] to-[#EC4899] p-2 rounded-lg">
              <Database size={20} className="text-white" />
            </div>
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-[#FAFAFA] to-[#A1A1AA] bg-clip-text text-transparent">AMV Tools</h1>
            <div className="text-[#71717A] text-xs uppercase tracking-wider">Semantic Scene Browser</div>
          </div>
        </div>
        <button onClick={() => setActiveTab('settings')} className="flex items-center gap-2 bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] hover:border-[#8B5CF6]/50 px-4 py-2 rounded-xl text-sm transition-all">
          <Database size={14} className="text-[#8B5CF6]" />
          <span className="font-medium truncate max-w-[280px]">{currentDbName(settings)}</span>
          <ChevronDown size={16} className="text-[#71717A]" />
        </button>
      </div>

      <div className="bg-[#18181B]/50 backdrop-blur-sm border-b-2 border-[#27272A] flex gap-1 px-3 py-2">
        {([
          { id: 'search', label: 'Search', color: 'from-[#8B5CF6] to-[#EC4899]' },
          { id: 'tags', label: 'Tags', color: 'from-[#EC4899] to-[#8B5CF6]' },
          { id: 'settings', label: 'Settings', color: 'from-[#8B5CF6] to-[#6366F1]' },
        ] as { id: Tab; label: string; color: string }[]).map((tab) => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)} className={`relative px-6 py-2.5 rounded-lg text-sm font-semibold transition-all ${activeTab === tab.id ? 'text-[#FAFAFA]' : 'text-[#71717A] hover:text-[#A1A1AA] hover:bg-[#27272A]'}`}>
            {activeTab === tab.id && (
              <>
                <div className={`absolute inset-0 bg-gradient-to-r ${tab.color} rounded-lg blur opacity-30`}></div>
                <div className={`absolute inset-0 bg-gradient-to-r ${tab.color} rounded-lg opacity-80`}></div>
              </>
            )}
            <span className="relative z-10">{tab.label}</span>
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-hidden relative">
        {activeTab === 'search' && (
          <SearchTab hoverDelayMs={hoverDelay} onOpenScene={(_s, list, idx) => setEditor({ scenes: list, index: idx })} />
        )}
        {activeTab === 'tags' && (
          <TagsTab hoverDelayMs={hoverDelay} onOpenScene={(_s, list, idx) => setEditor({ scenes: list, index: idx })} />
        )}
        {activeTab === 'settings' && <SettingsTab runtimeDevice={status?.device} />}
      </div>

      <StatusBar
        progress={progress}
        indexedCount={indexedCount}
        device={status?.gpu_name ?? status?.device ?? '...'}
        taggerProvider={status?.tagger_provider ?? null}
        warnings={status?.warnings ?? []}
      />

      {editor && (
        <MiniEditor scenes={editor.scenes} index={editor.index}
                    onIndexChange={(i) => setEditor({ ...editor, index: i })}
                    onJumpTo={(scenes, idx) => setEditor({ scenes, index: idx })}
                    onClose={() => setEditor(null)} />
      )}

      {showTutorial && (
        <TutorialOverlay onDone={() => { localStorage.setItem(TUTORIAL_KEY, '1'); setShowTutorial(false); }} />
      )}
    </div>
  );
}

function currentDbName(settings: AppSettings | null): string {
  if (!settings) return 'No library';
  const p = settings.databases.primary;
  if (!p) return 'No library';
  const seg = p.split(/[\\/]/).pop() || p;
  return seg.replace(/\.(db|sqlite|sqlite3)$/i, '');
}

function StatusBar({
  progress,
  indexedCount,
  device,
  taggerProvider,
  warnings,
}: {
  progress: ProgressEvent | null;
  indexedCount: number;
  device: string;
  taggerProvider: string | null;
  warnings: string[];
}) {
  const isIndexing = progress?.type === 'indexing';
  const isProxy = progress?.type === 'proxy';
  const percent = progress?.percent ?? (progress?.current && progress?.total ? Math.round((progress.current / progress.total) * 100) : 0);
  return (
    <div className="bg-gradient-to-r from-[#18181B] via-[#0F0F11] to-[#18181B] border-t-2 border-[#27272A] px-6 py-3 backdrop-blur-sm">
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-3 min-w-0">
          {isIndexing ? (
            <>
              <div className="relative">
                <div className="w-3 h-3 bg-[#8B5CF6] rounded-full animate-pulse shadow-lg shadow-[#8B5CF6]/50"></div>
                <div className="absolute inset-0 w-3 h-3 bg-[#8B5CF6] rounded-full animate-ping opacity-75"></div>
              </div>
              <span className="text-[#A1A1AA]">Indexing</span>
              <span className="text-[#FAFAFA] font-semibold truncate max-w-[280px]">{progress?.video ?? '...'}</span>
              <div className="w-32 h-1.5 bg-[#27272A] rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] rounded-full transition-all" style={{ width: `${percent}%` }}></div>
              </div>
              <span className="text-[#8B5CF6] font-mono font-bold text-xs">{percent}%</span>
              <button onClick={() => api.stopIndexing()} title="Stop indexing" className="ml-2 p-1 rounded-md text-[#71717A] hover:text-red-400 hover:bg-[#27272A]">
                <Square size={12} fill="currentColor" />
              </button>
            </>
          ) : isProxy ? (
            <>
              <div className="w-3 h-3 bg-[#EC4899] rounded-full animate-pulse"></div>
              <span className="text-[#A1A1AA]">Generating proxies</span>
              <span className="text-[#EC4899] font-mono font-bold text-xs">{percent}%</span>
            </>
          ) : (
            <>
              <div className="w-3 h-3 bg-[#27272A] rounded-full"></div>
              <span className="text-[#71717A]">Idle</span>
            </>
          )}
        </div>
        <div className="flex items-center gap-6 text-[#71717A]">
          <div className="flex items-center gap-2">
            <Zap size={14} className="text-[#EC4899]" />
            <span className="text-[#A1A1AA]">Device:</span>
            <span className="text-[#FAFAFA] font-mono font-medium text-xs truncate max-w-[200px]">{device}</span>
          </div>
          {taggerProvider && (
            <div className="flex items-center gap-2">
              <span className="text-[#A1A1AA]">Tagger:</span>
              <span className="text-[#FAFAFA] font-mono font-medium text-xs truncate max-w-[180px]">{taggerProvider.replace('ExecutionProvider', '')}</span>
            </div>
          )}
          {warnings.length > 0 && (
            <div className="flex items-center gap-1 text-amber-300 max-w-[360px]" title={warnings.join('\n')}>
              <AlertTriangle size={14} />
              <span className="truncate text-xs">{warnings[0]}</span>
            </div>
          )}
          <div className="w-px h-4 bg-[#27272A]"></div>
          <div className="flex items-center gap-2">
            <span className="text-[#A1A1AA]">Indexed scenes:</span>
            <span className="text-[#FAFAFA] font-mono font-bold">{indexedCount.toLocaleString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
