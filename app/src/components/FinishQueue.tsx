import { Ban, Check, ChevronDown, FolderOpen, Loader2, Trash2, Wand2, X } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useT } from '../i18n';
import { api } from '../api/client';
import { finishApi, formatDuration, subscribeFinish, type FinishJob } from '../api/finish';

/** Floating finishing-job queue (bottom-right). Hidden until the first job
 * exists; opens itself when a job is queued from a FinishDialog. */
export default function FinishQueue() {
  const t = useT();
  const [jobs, setJobs] = useState<FinishJob[]>([]);
  const [open, setOpen] = useState(false);

  const refresh = useCallback(() => {
    finishApi.jobs().then((r) => setJobs(r.jobs)).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const off = subscribeFinish((e) => {
      if (e.type !== 'finish_job' || !e.job) return;
      const job = e.job;
      setJobs((prev) => {
        const i = prev.findIndex((j) => j.id === job.id);
        if (i === -1) return [...prev, job];
        const next = prev.slice();
        next[i] = job;
        return next;
      });
    });
    const onChanged = (ev: Event) => {
      refresh();
      if ((ev as CustomEvent).detail?.open) setOpen(true);
    };
    window.addEventListener('amv-finish-jobs-changed', onChanged);
    return () => { off(); window.removeEventListener('amv-finish-jobs-changed', onChanged); };
  }, [refresh]);

  const active = useMemo(() => jobs.filter((j) => j.status === 'queued' || j.status === 'running'), [jobs]);
  const running = jobs.find((j) => j.status === 'running');
  if (jobs.length === 0) return null;

  const statusIcon = (j: FinishJob) => {
    if (j.status === 'running') return <Loader2 size={14} className="animate-spin text-[#8B5CF6]" />;
    if (j.status === 'done') return <Check size={14} className="text-emerald-400" />;
    if (j.status === 'failed') return <X size={14} className="text-red-400" />;
    if (j.status === 'cancelled') return <Ban size={14} className="text-[#71717A]" />;
    return <span className="w-3.5 h-3.5 rounded-full border-2 border-[#3F3F46] inline-block" />;
  };

  return (
    <div className="fixed bottom-4 right-4 z-[65] flex flex-col items-end gap-2">
      {open && (
        <div className="w-[420px] max-h-[60vh] flex flex-col bg-[#18181B] border-2 border-[#27272A] rounded-2xl shadow-2xl shadow-black/70">
          <div className="flex items-center gap-2 px-4 py-3 border-b-2 border-[#27272A]">
            <Wand2 size={15} className="text-[#A78BFA]" />
            <span className="text-sm font-semibold text-[#FAFAFA] flex-1">{t('finish.queue.title')}</span>
            <button onClick={() => finishApi.clearJobs().then(refresh).catch(() => {})} className="text-[11px] text-[#71717A] hover:text-[#FAFAFA] px-2 py-1 rounded hover:bg-[#27272A]">
              {t('finish.queue.clear')}
            </button>
            <button onClick={() => setOpen(false)} className="p-1 rounded hover:bg-[#27272A] text-[#A1A1AA]"><ChevronDown size={15} /></button>
          </div>
          <div className="overflow-y-auto p-2 space-y-1.5">
            {jobs.slice().reverse().map((j) => (
              <div key={j.id} className="bg-[#0F0F11] border border-[#27272A] rounded-xl px-3 py-2.5">
                <div className="flex items-center gap-2">
                  {statusIcon(j)}
                  <span className="text-xs font-semibold text-[#FAFAFA]">{t(`finish.kind.${j.kind}`)}</span>
                  <span className="text-xs text-[#71717A] truncate flex-1">{j.label}</span>
                  {(j.status === 'queued' || j.status === 'running') && (
                    <button onClick={() => finishApi.cancelJob(j.id).catch(() => {})} title={t('finish.queue.cancel')} className="text-[11px] text-[#A1A1AA] hover:text-red-300 px-1.5 py-0.5 rounded hover:bg-[#27272A]">
                      {t('finish.queue.cancel')}
                    </button>
                  )}
                  {j.status === 'done' && j.output && (
                    <button onClick={() => api.revealOutput(j.output!)} title={t('finish.queue.open')} className="p-1 text-[#A1A1AA] hover:text-[#FAFAFA] rounded hover:bg-[#27272A]">
                      <FolderOpen size={13} />
                    </button>
                  )}
                  {(j.status === 'done' || j.status === 'failed' || j.status === 'cancelled') && (
                    <button onClick={() => finishApi.removeJob(j.id).then(refresh).catch(() => {})} title={t('finish.queue.remove')} className="p-1 text-[#71717A] hover:text-[#FAFAFA] rounded hover:bg-[#27272A]">
                      <Trash2 size={12} />
                    </button>
                  )}
                </div>
                {j.status === 'running' && (
                  <>
                    <div className="h-1.5 bg-[#27272A] rounded-full overflow-hidden mt-2">
                      <div className="h-full bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] transition-all" style={{ width: `${j.percent}%` }} />
                    </div>
                    <div className="flex justify-between text-[10px] text-[#71717A] mt-1 font-mono">
                      <span className="truncate">{j.message}</span>
                      <span>{j.percent}%{j.eta_seconds != null ? ` · ${t('finish.queue.eta', { eta: formatDuration(j.eta_seconds) })}` : ''}</span>
                    </div>
                  </>
                )}
                {j.status !== 'running' && (
                  <div className="text-[10px] text-[#71717A] mt-1 truncate font-mono" title={j.error ?? j.output ?? ''}>
                    {t(`finish.status.${j.status}`)}
                    {j.status === 'done' && j.output ? ` · ${j.output.split(/[\\/]/).pop()}` : ''}
                    {j.status === 'failed' && j.error ? ` · ${j.error}` : ''}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 px-4 py-2 rounded-full bg-[#18181B] border-2 border-[#8B5CF6]/50 hover:border-[#8B5CF6] text-[#FAFAFA] text-xs font-semibold shadow-lg shadow-black/50"
      >
        {running ? <Loader2 size={14} className="animate-spin text-[#8B5CF6]" /> : <Wand2 size={14} className="text-[#A78BFA]" />}
        {t('finish.queue.badge', { n: active.length > 0 ? `(${active.length})` : '' })}
        {running && <span className="font-mono text-[#A1A1AA]">{running.percent}%</span>}
      </button>
    </div>
  );
}
