import { useCallback, useEffect, useState } from 'react';
import { Gauge, Loader2, RotateCcw } from 'lucide-react';
import { api } from '../api/client';
import type { AutotuneSetting, AutotuneStatus } from '../api/types';
import { useT, type TFunc } from '../i18n';

// Settings > Diagnostic: the settings the automatic tuning chose on this
// machine (backend/autotune), their measures, and a re-run button.
export default function AutotuneSection() {
  const t = useT();
  const [status, setStatus] = useState<AutotuneStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setStatus(await api.autotune());
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const busy = status?.state === 'running' || status?.state === 'waiting';
  useEffect(() => {
    if (!busy) return;
    const id = window.setInterval(load, 2000);
    return () => window.clearInterval(id);
  }, [busy, load]);

  async function rerun() {
    try {
      setStatus(await api.rerunAutotune());
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  const entry = status?.entry ?? null;
  const settings = entry ? Object.entries(entry.settings) : [];
  const skipped = entry ? Object.entries(entry.skipped) : [];
  const errors = entry ? Object.entries(entry.errors) : [];

  return (
    <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-[#FAFAFA] font-semibold">
            <Gauge size={16} className="text-[#8B5CF6]" />
            {t('autotune.title')}
          </div>
          <div className="text-[#71717A] text-xs mt-1 max-w-2xl">{t('autotune.hint')}</div>
        </div>
        <button
          onClick={rerun}
          disabled={busy || !status}
          className="flex-shrink-0 bg-[#27272A] hover:bg-[#3f3f46] disabled:opacity-50 text-[#FAFAFA] px-3 py-2 rounded-lg text-xs flex items-center gap-2"
        >
          {busy ? <Loader2 size={13} className="animate-spin" /> : <RotateCcw size={13} />}
          {t('autotune.rerun')}
        </button>
      </div>

      {error && <div className="text-red-300 text-xs mt-3">{error}</div>}
      {status && <StateLine status={status} t={t} />}

      {entry && (
        <div className="text-[#71717A] text-xs mt-2">
          {t('autotune.measuredAt', {
            date: new Date(entry.measured_at).toLocaleString(),
            seconds: entry.duration_s,
            source: t(`autotune.source.${entry.source}`),
          })}
        </div>
      )}
      {status && !entry && !busy && <div className="text-[#A1A1AA] text-sm mt-3">{t('autotune.none')}</div>}
      {entry && settings.length === 0 && skipped.length === 0 && errors.length === 0 && (
        <div className="text-[#A1A1AA] text-sm mt-3">{t('autotune.nothing')}</div>
      )}

      {(settings.length > 0 || skipped.length > 0 || errors.length > 0) && (
        <div className="flex flex-col gap-2 mt-3">
          {settings.map(([name, rec]) => (
            <SettingRow key={name} name={name} rec={rec} t={t} />
          ))}
          {skipped.map(([name, reason]) => (
            <div key={name} className="border border-[#27272A] rounded-lg px-3 py-2 text-xs">
              <span className="text-[#FAFAFA] text-sm">{t(`autotune.setting.${name}`)}</span>
              <span className="text-[#71717A] ml-2">{t('autotune.skipped')} — {reason}</span>
            </div>
          ))}
          {errors.map(([name, reason]) => (
            <div key={name} className="border border-amber-500/40 rounded-lg px-3 py-2 text-xs">
              <span className="text-[#FAFAFA] text-sm">{t(`autotune.setting.${name}`)}</span>
              <span className="text-amber-300 ml-2">{t('autotune.errorSetting')}</span>
              <div className="text-[#71717A] font-mono mt-1 break-words">{reason}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function StateLine({ status, t }: { status: AutotuneStatus; t: TFunc }) {
  if (status.state === 'done' || (status.state === 'idle' && status.entry)) return null;
  const text = status.state === 'running' && status.step
    ? t('autotune.state.runningStep', { step: t(`autotune.setting.${status.step}`) })
    : status.state === 'failed'
      ? t('autotune.state.failed', { error: status.error ?? '?' })
      : t(`autotune.state.${status.state}`);
  const tone = status.state === 'failed' ? 'text-amber-300' : 'text-[#A78BFA]';
  return <div className={`${tone} text-xs mt-3`}>{text}</div>;
}

function SettingRow({ name, rec, t }: { name: string; rec: AutotuneSetting; t: TFunc }) {
  const measures = Object.entries(rec.measures);
  return (
    <div className="border border-[#27272A] rounded-lg px-3 py-2">
      <div className="flex items-center justify-between gap-3">
        <span className="text-[#FAFAFA] text-sm">{t(`autotune.setting.${name}`)}</span>
        <span className="text-sm font-mono text-[#FAFAFA]">
          {String(rec.value)}
          {rec.value !== rec.default && (
            <span className="text-[#71717A] text-xs ml-2">({t('autotune.default', { value: String(rec.default) })})</span>
          )}
        </span>
      </div>
      {measures.length > 0 && (
        <div className="text-[#71717A] text-xs font-mono mt-1 flex flex-wrap gap-x-3">
          {measures.map(([cand, v]) => (
            <span key={cand} className={cand === String(rec.value) ? 'text-[#A78BFA]' : undefined}>
              {cand}: {v == null ? t('autotune.failedCandidate') : `${v} ${rec.unit}`}
            </span>
          ))}
        </div>
      )}
      {rec.note && <div className="text-[#52525B] text-xs mt-1">{rec.note}</div>}
    </div>
  );
}
