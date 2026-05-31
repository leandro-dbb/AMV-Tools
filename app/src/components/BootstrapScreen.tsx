import { useEffect, useRef } from 'react';
import { Sparkles, Loader2, AlertCircle } from 'lucide-react';

interface BootstrapLine {
  line: string;
  stream: 'stdout' | 'stderr';
}

interface Props {
  lines: BootstrapLine[];
  error: string | null;
}

export default function BootstrapScreen({ lines, error }: Props) {
  const preRef = useRef<HTMLPreElement | null>(null);

  useEffect(() => {
    if (preRef.current) preRef.current.scrollTop = preRef.current.scrollHeight;
  }, [lines.length]);

  const subtitle = error
    ? 'Something went wrong while starting the Python backend.'
    : detectStage(lines);

  return (
    <div
      className="w-screen h-screen bg-gradient-to-br from-[#0F0F11] via-[#18181B] to-[#0F0F11] text-[#FAFAFA] flex items-center justify-center p-8 overflow-auto"
      style={{ fontFamily: 'Inter, Space Grotesk, system-ui, sans-serif' }}
    >
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-[#8B5CF6]/10 rounded-full blur-[150px] pointer-events-none" />
      <div className="relative max-w-3xl w-full">
        <div className="flex items-center gap-3 mb-2">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] rounded-lg blur-lg opacity-50" />
            <div className="relative bg-gradient-to-br from-[#8B5CF6] to-[#EC4899] p-3 rounded-xl">
              <Sparkles size={24} className="text-white" />
            </div>
          </div>
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-[#FAFAFA] to-[#A1A1AA] bg-clip-text text-transparent">
              AMV Tools
            </h1>
            <div className="text-[#71717A] text-sm uppercase tracking-wider">
              {error ? 'Backend failure' : 'Starting backend'}
            </div>
          </div>
        </div>

        <p className="text-[#A1A1AA] mb-6 max-w-2xl">{subtitle}</p>

        <div
          className={`bg-[#18181B] border-2 rounded-xl p-6 ${
            error ? 'border-red-500/50' : 'border-[#27272A]'
          }`}
        >
          <div className="flex items-center gap-3 mb-4">
            {error ? (
              <AlertCircle size={20} className="text-red-400" />
            ) : (
              <Loader2 size={20} className="text-[#8B5CF6] animate-spin" />
            )}
            <div className="text-sm">
              {error ? (
                <span className="text-red-300 font-semibold">{error}</span>
              ) : (
                <>
                  <div className="font-semibold">
                    First launch — fetching the Python sidecar
                  </div>
                  <div className="text-xs text-[#71717A]">
                    This usually takes 1-3 minutes the first time. The app will
                    open automatically when the backend is ready.
                  </div>
                </>
              )}
            </div>
          </div>
          <pre
            ref={preRef}
            className="bg-[#0F0F11] border border-[#27272A] rounded-lg p-3 text-xs font-mono max-h-[420px] overflow-auto whitespace-pre-wrap"
          >
            {lines.length === 0 ? (
              <span className="text-[#71717A]">Waiting for sidecar output…</span>
            ) : (
              lines.map((l, i) => (
                <div
                  key={i}
                  className={l.stream === 'stderr' ? 'text-[#A1A1AA]' : 'text-[#FAFAFA]'}
                >
                  {l.line}
                </div>
              ))
            )}
          </pre>
          {error && (
            <div className="text-xs text-[#71717A] mt-3">
              Close and relaunch AMV Tools to retry. If the problem persists,
              check the log above for clues (most often: missing network access
              for the initial wheel download, or insufficient disk space).
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function detectStage(lines: BootstrapLine[]): string {
  for (let i = lines.length - 1; i >= 0; i--) {
    const l = lines[i].line;
    if (/Uvicorn running/i.test(l)) return 'Backend is responding — opening the app.';
    if (/Application startup complete/i.test(l)) return 'FastAPI started — handshake in progress.';
    if (/Installed \d+ packages/i.test(l)) return 'Packages installed — launching server.';
    if (/Downloading /i.test(l)) return 'Downloading Python wheels…';
    if (/Resolved \d+ packages/i.test(l)) return 'Resolving dependencies…';
    if (/Creating virtual environment/i.test(l)) return 'Creating the Python environment…';
    if (/Using CPython/i.test(l)) return 'Preparing Python runtime…';
  }
  return 'Initialising — this can take a couple of minutes on first launch.';
}
