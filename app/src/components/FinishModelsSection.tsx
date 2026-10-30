import { Download, Loader2, Trash2, Wand2 } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { useT } from '../i18n';
import { finishApi, formatBytes, subscribeFinish, type FinishModel } from '../api/finish';

/** Settings > Models: download / delete the on-demand finishing models. */
export default function FinishModelsSection() {
  const t = useT();
  const [models, setModels] = useState<FinishModel[]>([]);
  const [root, setRoot] = useState('');
  const [progress, setProgress] = useState<Record<string, number>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  const refresh = useCallback(() => {
    finishApi.models().then((r) => { setModels(r.models); setRoot(r.root); }).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    return subscribeFinish((e) => {
      if (e.type !== 'finish_model' || !e.model_id) return;
      const id = e.model_id;
      if (e.done) {
        setProgress((p) => { const n = { ...p }; delete n[id]; return n; });
        if (e.ok === false) setErrors((x) => ({ ...x, [id]: e.message ?? '' }));
        refresh();
      } else {
        setProgress((p) => ({ ...p, [id]: e.percent ?? 0 }));
      }
    });
  }, [refresh]);

  async function download(id: string) {
    setErrors((x) => { const n = { ...x }; delete n[id]; return n; });
    setProgress((p) => ({ ...p, [id]: 0 }));
    try { await finishApi.downloadModel(id); } catch (err) {
      setProgress((p) => { const n = { ...p }; delete n[id]; return n; });
      setErrors((x) => ({ ...x, [id]: (err as Error).message }));
    }
  }

  async function remove(id: string) {
    try { await finishApi.deleteModel(id); } catch (err) {
      // 409 while it downloads or a finishing job is queued
      setErrors((x) => ({ ...x, [id]: (err as Error).message }));
    } finally { refresh(); }
  }

  return (
    <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5 hover:border-[#8B5CF6]/30 transition-all">
      <div className="flex items-center gap-2 mb-1">
        <Wand2 size={16} className="text-[#A78BFA]" />
        <div className="text-[#FAFAFA] font-medium">{t('finish.models.title')}</div>
      </div>
      <div className="text-[#71717A] text-sm mb-3">{t('finish.models.hint', { root: root || '…' })}</div>
      <div className="space-y-2">
        {models.map((m) => {
          const pct = progress[m.id] ?? (m.downloading ? m.download_percent ?? 0 : undefined);
          return (
            <div key={m.id} className="flex items-center gap-3 bg-[#0F0F11] border-2 border-[#27272A] rounded-xl px-4 py-3">
              <div className="flex-1 min-w-0">
                <div className="text-sm text-[#FAFAFA] font-medium truncate">{m.label}</div>
                <div className="text-[11px] text-[#71717A]">
                  {m.installed
                    ? t('finish.models.installed', { size: formatBytes(m.disk_size) })
                    : t('finish.models.notInstalled', { size: formatBytes(m.size) })}
                  {' · '}
                  <a href={m.source} target="_blank" rel="noreferrer" className="hover:text-[#A1A1AA] underline">{t('finish.models.license', { license: m.license })}</a>
                  {m.non_commercial && <span className="ml-1.5 px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300">{t('finish.models.nonCommercial')}</span>}
                </div>
                {errors[m.id] && <div className="text-[11px] text-red-300 mt-1 break-words">{t('finish.models.error', { message: errors[m.id] })}</div>}
                {pct !== undefined && (
                  <div className="h-1 bg-[#27272A] rounded-full overflow-hidden mt-2">
                    <div className="h-full bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] transition-all" style={{ width: `${pct}%` }} />
                  </div>
                )}
              </div>
              {pct !== undefined ? (
                <span className="text-xs text-[#8B5CF6] flex items-center gap-1.5"><Loader2 size={13} className="animate-spin" />{pct}%</span>
              ) : m.installed ? (
                <button onClick={() => remove(m.id)} className="px-3 py-1.5 rounded-lg bg-[#27272A] hover:bg-red-500/20 text-[#A1A1AA] hover:text-red-300 text-xs flex items-center gap-1.5">
                  <Trash2 size={13} />{t('finish.models.delete')}
                </button>
              ) : (
                <button onClick={() => download(m.id)} className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white text-xs font-semibold flex items-center gap-1.5">
                  <Download size={13} />{t('finish.download')}
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
