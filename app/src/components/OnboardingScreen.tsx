import { useEffect, useRef, useState } from 'react';
import { Cpu, Sparkles, Check, AlertCircle, Loader2 } from 'lucide-react';
import { api, platformIs } from '../api/client';

type GpuBackend = 'cu130' | 'cu130-trt' | 'cu126' | 'dml' | 'xpu' | 'rocm' | 'cpu';

interface Option {
  id: GpuBackend;
  title: string;
  subtitle: string;
  recommended?: boolean;
  badge?: string;
  detail: string;
  platforms?: ('win32' | 'linux' | 'darwin')[];
}

const ALL_OPTIONS: Option[] = [
  { id: 'cu130-trt', title: 'NVIDIA CUDA 13 + TensorRT', subtitle: 'RTX 20xx and newer · fastest', badge: 'GPU everywhere', detail: 'SigLIP on CUDA with TensorRT JIT acceleration + wd-tagger on CUDA via ONNX. ~3.5 GB total download (PyTorch + TensorRT + dual CUDA 12/13 runtimes). Pick this if you want maximum speed.', platforms: ['win32', 'linux'] },
  { id: 'cu130', title: 'NVIDIA CUDA 13', subtitle: 'RTX 20xx and newer', recommended: true, badge: 'GPU everywhere', detail: 'SigLIP + wd-tagger both on CUDA. ~2.5 GB total download. No TensorRT — slightly slower for SigLIP but a smaller install and faster first launch.', platforms: ['win32', 'linux'] },
  { id: 'cu126', title: 'NVIDIA CUDA 12.6', subtitle: 'GTX 10xx · older drivers', detail: 'Use this if your driver is older than 555.', platforms: ['win32', 'linux'] },
  { id: 'dml', title: 'DirectML (AMD / Intel)', subtitle: 'Windows only', detail: 'AMD Radeon, Intel Arc and any DirectX 12 GPU.', platforms: ['win32'] },
  { id: 'rocm', title: 'AMD ROCm', subtitle: 'Linux only', detail: 'Native AMD acceleration on Linux.', platforms: ['linux'] },
  { id: 'xpu', title: 'Intel Arc / Xe (XPU)', subtitle: 'oneAPI', detail: 'For Intel Arc A-series and Xe.', platforms: ['win32', 'linux'] },
  { id: 'cpu', title: 'CPU only', subtitle: 'No GPU acceleration', detail: 'Works everywhere, much slower for indexing.' },
];

export default function OnboardingScreen({ onDone }: { onDone: () => void }) {
  const [phase, setPhase] = useState<'choose' | 'installing' | 'done' | 'error'>('choose');
  const [selected, setSelected] = useState<GpuBackend>('cu130');
  const [log, setLog] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [pythonVer, setPythonVer] = useState<string>('');
  const logRef = useRef<HTMLPreElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    api.status().then((s) => setPythonVer(s.python_version)).catch(() => {});
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  const visibleOptions = ALL_OPTIONS.filter((o) => {
    if (!o.platforms) return true;
    return o.platforms.some((p) => platformIs(p));
  });

  async function startInstall() {
    setPhase('installing');
    setLog('');
    setError('');

    try {
      wsRef.current = api.progressSocket();
      wsRef.current.onmessage = (msg) => {
        try {
          const evt = JSON.parse(msg.data);
          if (evt.type === 'install') {
            if (evt.line) setLog((l) => l + evt.line);
          }
        } catch {}
      };
    } catch (err) {
      // proceed without live stream
    }

    try {
      const res = await api.setupGpu(selected);
      if (!log) setLog(res.log);
      if (res.ok) {
        setPhase('done');
        setTimeout(() => { wsRef.current?.close(); onDone(); }, 1200);
      } else {
        setPhase('error');
        setError(res.hint || 'Installation failed. Retry this backend or choose CPU to open the app.');
      }
    } catch (err) {
      setPhase('error');
      setError((err as Error).message);
    }
  }

  return (
    <div className="w-screen h-screen bg-gradient-to-br from-[#0F0F11] via-[#18181B] to-[#0F0F11] text-[#FAFAFA] flex items-center justify-center p-8 overflow-auto" style={{ fontFamily: 'Inter, Space Grotesk, system-ui, sans-serif' }}>
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-[#8B5CF6]/10 rounded-full blur-[150px] pointer-events-none"></div>

      <div className="relative max-w-4xl w-full">
        <div className="flex items-center gap-3 mb-2">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] rounded-lg blur-lg opacity-50"></div>
            <div className="relative bg-gradient-to-br from-[#8B5CF6] to-[#EC4899] p-3 rounded-xl">
              <Sparkles size={24} className="text-white" />
            </div>
          </div>
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-[#FAFAFA] to-[#A1A1AA] bg-clip-text text-transparent">Welcome to AMV Tools</h1>
            <div className="text-[#71717A] text-sm uppercase tracking-wider">One-time setup</div>
          </div>
        </div>

        <p className="text-[#A1A1AA] mb-8 max-w-2xl">
          Pick the acceleration backend that matches your hardware. This downloads ~1-3 GB of PyTorch wheels.
          You can change it later from <span className="text-[#FAFAFA]">Settings → Models</span>.
        </p>

        {phase === 'choose' && (
          <>
            <div className="grid grid-cols-2 gap-3 mb-6">
              {visibleOptions.map((opt) => {
                const isSel = selected === opt.id;
                return (
                  <button key={opt.id} onClick={() => setSelected(opt.id)}
                    className={`relative text-left p-5 rounded-xl border-2 transition-all ${isSel ? 'border-[#8B5CF6] bg-gradient-to-br from-[#8B5CF6]/10 to-[#EC4899]/10 shadow-lg shadow-[#8B5CF6]/20' : 'border-[#27272A] bg-[#18181B]/50 hover:border-[#3f3f46]'}`}>
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <Cpu size={16} className={isSel ? 'text-[#8B5CF6]' : 'text-[#71717A]'} />
                          <span className="font-semibold text-[#FAFAFA]">{opt.title}</span>
                          {opt.recommended && <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#8B5CF6]/20 text-[#8B5CF6] font-bold uppercase tracking-wider">Recommended</span>}
                          {opt.badge && <span className="text-[10px] px-2 py-0.5 rounded-full bg-[#EC4899]/20 text-[#EC4899] font-bold uppercase tracking-wider">{opt.badge}</span>}
                        </div>
                        <div className="text-xs text-[#71717A] mb-2">{opt.subtitle}</div>
                        <div className="text-xs text-[#A1A1AA]">{opt.detail}</div>
                      </div>
                      {isSel && <Check size={20} className="text-[#8B5CF6] flex-shrink-0" />}
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="flex justify-end gap-3">
              <button onClick={startInstall} className="bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-[#FAFAFA] px-8 py-3 rounded-xl text-sm font-semibold hover:shadow-lg hover:shadow-[#8B5CF6]/50 transition-all">
                Install and continue
              </button>
            </div>
          </>
        )}

        {phase === 'installing' && (
          <div className="bg-[#18181B] border-2 border-[#27272A] rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <Loader2 size={20} className="text-[#8B5CF6] animate-spin" />
              <div>
                <div className="font-semibold">Installing {selected.toUpperCase()} backend</div>
                <div className="text-xs text-[#71717A]">This usually takes 2-8 minutes depending on your connection.</div>
              </div>
            </div>
            <pre ref={logRef} className="bg-[#0F0F11] border border-[#27272A] rounded-lg p-3 text-xs text-[#A1A1AA] font-mono max-h-[400px] overflow-auto whitespace-pre-wrap">
              {log || 'Resolving PyTorch wheels...'}
            </pre>
          </div>
        )}

        {phase === 'done' && (
          <div className="bg-gradient-to-br from-[#8B5CF6]/20 to-[#EC4899]/20 border-2 border-[#8B5CF6]/50 rounded-xl p-6 flex items-center gap-3">
            <Check size={24} className="text-[#8B5CF6]" />
            <div>
              <div className="font-semibold">Setup complete</div>
              <div className="text-xs text-[#A1A1AA]">Loading the app...</div>
            </div>
          </div>
        )}

        {phase === 'error' && (
          <div className="bg-red-500/10 border-2 border-red-500/50 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-3">
              <AlertCircle size={20} className="text-red-400" />
              <div className="font-semibold text-red-300">Installation failed</div>
            </div>
            <pre className="bg-[#0F0F11] border border-[#27272A] rounded-lg p-3 text-xs text-[#A1A1AA] font-mono max-h-[300px] overflow-auto whitespace-pre-wrap mb-3">{log}</pre>
            <div className="text-xs text-red-400 mb-3">{error}</div>
            <button onClick={() => setPhase('choose')} className="bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-5 py-2 rounded-lg text-sm">Try another backend</button>
          </div>
        )}

        <div className="mt-6 text-xs text-[#71717A] text-center">
          {pythonVer ? `Python ${pythonVer} - ` : ''}AMV Tools v0.1 alpha
        </div>
      </div>
    </div>
  );
}
