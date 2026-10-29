// Status-bar block shown while indexing: file, rank in the queue, progress,
// current stage in plain words, time left, and a labelled Stop button.
import { useEffect, useRef, useState } from 'react';
import { Square } from 'lucide-react';
import { api } from '../api/client';
import { formatDuration } from '../api/finish';
import type { ProgressEvent } from '../api/types';
import { useT } from '../i18n';
import { etaSeconds, stageLabel } from '../indexProgress';

export default function IndexingStatus({ progress, percent }: { progress: ProgressEvent; percent: number }) {
  const t = useT();
  // Anchor of the time-left estimate: reset per file, then moved to the
  // moment the bar reaches the tagging stage (cut detection is not linear).
  const started = useRef<{ video?: string; at: number; pct: number }>({ at: Date.now(), pct: 0 });
  const [now, setNow] = useState(Date.now());
  const [stopping, setStopping] = useState(false);

  if (started.current.video !== progress.video) started.current = { video: progress.video, at: Date.now(), pct: 0 };
  if (started.current.pct < 25 && percent >= 25) started.current = { video: progress.video, at: Date.now(), pct: percent };
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const eta = etaSeconds(percent, (now - started.current.at) / 1000, started.current.pct);
  const stage = stageLabel(progress.message, t);
  return (
    <>
      <div className="relative">
        <div className="w-3 h-3 bg-[#8B5CF6] rounded-full animate-pulse shadow-lg shadow-[#8B5CF6]/50"></div>
        <div className="absolute inset-0 w-3 h-3 bg-[#8B5CF6] rounded-full animate-ping opacity-75"></div>
      </div>
      <span className="text-[#A1A1AA]">{t('app.statusbar.indexing')}</span>
      <span className="text-[#FAFAFA] font-semibold truncate max-w-[240px]" title={progress.video}>{progress.video ?? '...'}</span>
      {progress.file_count && progress.file_count > 1 && (
        <span className="text-[#71717A] text-xs whitespace-nowrap">{t('journey.index.file', { i: progress.file_index ?? 1, n: progress.file_count })}</span>
      )}
      <div className="w-32 h-1.5 bg-[#27272A] rounded-full overflow-hidden flex-shrink-0">
        <div className="h-full bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] rounded-full transition-all" style={{ width: `${percent}%` }}></div>
      </div>
      <span className="text-[#8B5CF6] font-mono font-bold text-xs">{percent}%</span>
      {stage && <span data-testid="index-stage" className="text-[#A1A1AA] text-xs truncate max-w-[300px]" title={stage}>{stage}</span>}
      {eta != null && <span className="text-[#71717A] text-xs whitespace-nowrap">{t('journey.index.eta', { eta: formatDuration(eta) })}</span>}
      <button
        disabled={stopping}
        onClick={() => { setStopping(true); api.stopIndexing().finally(() => setTimeout(() => setStopping(false), 3000)); }}
        title={t('app.statusbar.stopIndexing')}
        className="ml-2 flex items-center gap-1 px-2 py-0.5 rounded-md text-xs text-[#A1A1AA] border border-[#27272A] hover:text-red-400 hover:border-red-400/50 disabled:opacity-40"
      >
        <Square size={10} fill="currentColor" /> {t('journey.index.stop')}
      </button>
    </>
  );
}
