// Pick / drop a music file, or reopen one analysed earlier.
import { useEffect, useState } from 'react';
import { AlertTriangle, Loader2, Music2, Upload } from 'lucide-react';
import type { MusicTrackSummary } from '../api/music';
import { useT } from '../i18n';
import { formatTime } from './musicUtils';

export const AUDIO_EXTENSIONS = ['mp3', 'wav', 'flac', 'm4a', 'ogg', 'aac', 'opus', 'mp4', 'mkv', 'mov', 'webm'];

interface Props {
  analyzing: boolean;
  error: string | null;
  tracks: MusicTrackSummary[];
  currentHash: string | null;
  onAnalyze: (path: string) => void;
  onOpenTrack: (hash: string) => void;
  onError: (message: string) => void;
  compact?: boolean;
}

export default function MusicImport({ analyzing, error, tracks, currentHash, onAnalyze, onOpenTrack, onError, compact }: Props) {
  const t = useT();
  const [dragOver, setDragOver] = useState(false);
  const [path, setPath] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const hasBridge = !!window.amvBridge?.openFileDialog;

  useEffect(() => {
    if (!analyzing) { setElapsed(0); return; }
    const started = Date.now();
    const id = window.setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 500);
    return () => window.clearInterval(id);
  }, [analyzing]);

  async function choose() {
    if (analyzing || !window.amvBridge) return;
    try {
      const picked = await window.amvBridge.openFileDialog({ filters: [{ name: 'Audio', extensions: AUDIO_EXTENSIONS }] });
      if (picked && picked[0]) onAnalyze(picked[0]);
    } catch (e) {
      onError((e as Error).message);
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    if (analyzing) return;
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    const p = (file as any).path as string | undefined;
    if (p) onAnalyze(p); else onError(t('music.import.dropNoPath'));
  }

  const others = tracks.filter((tr) => tr.hash !== currentHash);

  return (
    <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5 flex flex-col gap-4">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={`rounded-xl border-2 border-dashed transition-colors flex flex-col items-center justify-center text-center px-6 ${compact ? 'py-5' : 'py-10'} ${
          dragOver ? 'border-[#EC4899] bg-[#EC4899]/10' : 'border-[#3F3F46] bg-[#0F0F11]/50'
        }`}
      >
        {analyzing ? (
          <>
            <Loader2 size={34} className="animate-spin text-[#EC4899]" />
            <div className="text-[#FAFAFA] font-semibold mt-3">{t('music.import.analyzing')}</div>
            <div className="text-xs text-[#A1A1AA] mt-1 max-w-md">{t('music.import.analyzingHint', { seconds: elapsed })}</div>
          </>
        ) : (
          <>
            <Music2 size={compact ? 26 : 36} className={dragOver ? 'text-[#EC4899]' : 'text-[#71717A]'} />
            <div className="text-[#FAFAFA] font-semibold mt-3">{t('music.import.drop')}</div>
            <div className="text-[11px] text-[#71717A] mt-1">{t('music.import.formats')}</div>
            <div className="text-xs text-[#71717A] my-2">{t('music.import.or')}</div>
            {hasBridge ? (
              <button type="button" onClick={choose}
                      className="bg-gradient-to-r from-[#EC4899] to-[#8B5CF6] text-[#FAFAFA] px-6 py-2.5 rounded-lg text-sm font-semibold hover:shadow-lg hover:shadow-[#EC4899]/30 transition-all flex items-center gap-2">
                <Upload size={14} /> {t('music.import.choose')}
              </button>
            ) : (
              <form className="flex items-center gap-2 w-full max-w-xl"
                    onSubmit={(e) => { e.preventDefault(); if (path.trim()) onAnalyze(path.trim()); }}>
                <input value={path} onChange={(e) => setPath(e.target.value)} placeholder={t('music.import.pathPlaceholder')}
                       className="flex-1 bg-[#0F0F11] border-2 border-[#27272A] focus:border-[#EC4899] rounded-lg px-3 py-2 text-sm font-mono text-[#FAFAFA] focus:outline-none" />
                <button type="submit" disabled={!path.trim()}
                        className="bg-gradient-to-r from-[#EC4899] to-[#8B5CF6] text-[#FAFAFA] px-4 py-2 rounded-lg text-sm font-semibold disabled:opacity-40">
                  {t('music.import.analyze')}
                </button>
              </form>
            )}
          </>
        )}
      </div>

      {error && (
        <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/40 text-red-300 text-xs rounded-lg px-3 py-2 break-words">
          <AlertTriangle size={14} className="shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {!compact || others.length > 0 ? (
        <div className="flex flex-col gap-1.5">
          <div className="text-[11px] uppercase tracking-wider text-[#71717A]">{t('music.import.recent')}</div>
          {others.length === 0 ? (
            <div className="text-xs text-[#52525B]">{t('music.import.recentEmpty')}</div>
          ) : (
            <div className="flex flex-col gap-1 max-h-48 overflow-y-auto">
              {others.map((tr) => (
                <button key={tr.hash} type="button" disabled={analyzing} onClick={() => onOpenTrack(tr.hash)}
                        title={tr.path}
                        className="flex items-center gap-3 text-left px-3 py-2 rounded-lg bg-[#0F0F11]/60 border border-[#27272A] hover:border-[#EC4899]/60 disabled:opacity-40 transition-colors">
                  <Music2 size={14} className="text-[#EC4899] shrink-0" />
                  <span className="text-sm text-[#FAFAFA] truncate flex-1">{tr.name}</span>
                  <span className="text-xs text-[#A1A1AA] font-mono shrink-0">{tr.bpm ? `${Math.round(tr.bpm)} BPM` : ''}</span>
                  <span className="text-xs text-[#71717A] font-mono shrink-0 w-12 text-right">{formatTime(tr.duration_ms)}</span>
                  <span className="text-xs text-[#8B5CF6] shrink-0">{t('music.import.open')}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
