import { AlertTriangle, CheckCircle2, Gauge, Loader2, RefreshCw, X } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { api, modelTiersApi } from '../api/client';
import type { AppSettings, ModelTiersInfo, ReembedEvent, ReembedJob } from '../api/types';
import { useI18n } from '../i18n';
import {
  TIER_ROLES, clampPercent, formatSizeMb, isNotFound, normalizeJob, parseApiError, tierLabel, tierOptions,
} from '../modelTiers';

type Result =
  | { kind: 'done' }
  | { kind: 'busy' }
  | { kind: 'unsupported' }
  | { kind: 'error'; message: string; videos_missing: number };

/**
 * Settings > Models, top block: the model tier picked for the GPU memory, the
 * models it implies, and the embeddings-only re-index when the database was
 * built with another embedding model. Renders nothing on an older sidecar
 * that has no /api/models/tiers.
 */
export default function ModelTiersSection({ settings, patch }: {
  settings: AppSettings;
  patch: (section: 'models', partial: Partial<AppSettings['models']>) => unknown;
}) {
  const { t, lang } = useI18n();
  const [info, setInfo] = useState<ModelTiersInfo | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [job, setJob] = useState<ReembedJob | null>(null);
  const [starting, setStarting] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const alive = useRef(true);
  const firstLoad = useRef(true);

  const refresh = useCallback(() => {
    modelTiersApi.tiers().then((r) => {
      if (!alive.current) return;
      setInfo(r);
      setLoadError(null);
      setUnavailable(false);
      if (r.embedding?.job?.running) setJob(r.embedding.job);
    }).catch((err) => {
      if (!alive.current) return;
      if (isNotFound(err)) setUnavailable(true);
      else setLoadError(parseApiError(err).message);
    });
  }, []);

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);

  // Reload on mount and whenever a setting that changes the effective models
  // moves (debounced so the settings save lands first).
  const m = settings.models;
  const settingsKey = [m.tier, m.wd_tagger_variant, m.siglip_variant, m.siglip_max_num_patches].join('|');
  useEffect(() => {
    if (firstLoad.current) { firstLoad.current = false; refresh(); return; }
    const id = setTimeout(refresh, 500);
    return () => clearTimeout(id);
  }, [settingsKey, refresh]);

  const finish = useCallback((ok: boolean, message: string, videosMissing: number) => {
    setJob(null);
    setResult(ok ? { kind: 'done' } : { kind: 'error', message, videos_missing: videosMissing });
    refresh();
  }, [refresh]);

  // Live progress from the shared progress socket.
  useEffect(() => {
    if (unavailable) return;
    let ws: WebSocket | null = null;
    let closed = false;
    let retry: ReturnType<typeof setTimeout> | null = null;
    const open = () => {
      try { ws = api.progressSocket(); } catch { return; }
      ws.onmessage = (msg) => {
        let e: ReembedEvent;
        try { e = JSON.parse(msg.data); } catch { return; }
        if (!e || e.type !== 'reembed' || !alive.current) return;
        if (e.done) finish(!!e.ok, e.message ?? '', e.videos_missing ?? 0);
        else setJob({ running: true, percent: e.percent ?? 0, message: e.message ?? '', error: null });
      };
      ws.onclose = () => { if (!closed) retry = setTimeout(open, 2000); };
    };
    open();
    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      ws?.close();
    };
  }, [unavailable, finish]);

  // Polling fallback every 2 s while a job runs.
  const running = !!job?.running;
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      modelTiersApi.reembedStatus().then((r) => {
        if (!alive.current) return;
        const j = normalizeJob(r);
        if (!j) return;
        if (j.running) setJob(j);
        else finish(!j.error && j.ok !== false, j.error ?? j.message ?? '', j.videos_missing ?? 0);
      }).catch(() => { /* the socket may still report progress */ });
    }, 2000);
    return () => clearInterval(id);
  }, [running, finish]);

  async function start(force: boolean) {
    if (!info) return;
    setStarting(true);
    setResult(null);
    try {
      const r = await modelTiersApi.reembed({ model: info.embedding.target_model, ...(force ? { force: true } : {}) });
      if (r.ok) setJob({ running: true, percent: 0, message: '', error: null });
      else setResult({ kind: 'error', message: r.error ?? '', videos_missing: r.videos_missing ?? 0 });
    } catch (err) {
      const e = parseApiError(err);
      if (e.status === 409) setResult({ kind: 'busy' });
      else if (e.status === 404) setResult({ kind: 'unsupported' });
      else setResult({ kind: 'error', message: e.message, videos_missing: e.videos_missing });
    } finally {
      if (alive.current) setStarting(false);
    }
  }

  async function cancel() {
    try { await modelTiersApi.reembedCancel(); } catch { /* polling reports the final state */ }
  }

  async function selectTier(tier: string) {
    setResult(null);
    try { await patch('models', { tier }); } finally { refresh(); }
  }

  if (unavailable) return null;
  if (!info) {
    if (!loadError) return null;
    return <div className="text-xs text-[#71717A]">{t('models.unavailable', { message: loadError })}</div>;
  }

  const selected = m.tier ?? info.selected ?? 'auto';
  // Same rule as the legacy dropdowns in SettingsTab: locked unless Custom.
  const locked = m.tier !== undefined && m.tier !== 'custom';
  const options = tierOptions(info, t, lang);
  if (!options.some((o) => o.value === selected)) options.push({ value: selected, label: tierLabel(selected, t) });
  const effective = info.tiers.find((x) => x.id === info.effective_tier);
  const emb = info.embedding;
  const showBanner = !!emb?.reindex_needed || running || result !== null;

  return (
    <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5 hover:border-[#8B5CF6]/30 transition-all">
      <div className="flex items-center justify-between gap-4">
        <div className="flex flex-col flex-1">
          <span className="text-[#FAFAFA] font-semibold mb-1 flex items-center gap-2">
            <Gauge size={16} className="text-[#A78BFA]" />{t('models.tier.label')}
          </span>
          <span className="text-[#71717A] text-sm">{t('models.tier.hint')}</span>
        </div>
        <select
          value={selected}
          onChange={(e) => selectTier(e.target.value)}
          className="bg-[#0F0F11] border-2 border-[#8B5CF6]/30 rounded-lg px-5 py-2.5 text-[#FAFAFA] focus:outline-none focus:border-[#8B5CF6] max-w-xs"
        >
          {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      </div>

      {selected === 'custom' || !effective ? (
        selected === 'custom' && <div className="text-xs text-[#71717A] mt-3">{t('models.table.custom')}</div>
      ) : (
        <>
          <table className="w-full mt-4 text-xs">
            <thead>
              <tr className="text-left text-[#71717A]">
                <th className="font-medium py-1 pr-3">{t('models.table.function')}</th>
                <th className="font-medium py-1 pr-3">{t('models.table.model')}</th>
                <th className="font-medium py-1 pr-3 text-right">{t('models.table.size')}</th>
                <th className="font-medium py-1">{t('models.table.license')}</th>
              </tr>
            </thead>
            <tbody>
              {TIER_ROLES.map((role) => {
                const model = effective.models[role];
                if (!model) return null;
                return (
                  <tr key={role} className="border-t border-[#27272A] text-[#D4D4D8]">
                    <td className="py-1.5 pr-3 text-[#A1A1AA] whitespace-nowrap">{t(`models.role.${role}`)}</td>
                    <td className="py-1.5 pr-3 break-words" title={model.id}>
                      {model.label}
                      {role === 'embedding' && model.patches ? <span className="text-[#71717A]"> · {t('models.table.patches', { n: model.patches })}</span> : null}
                    </td>
                    <td className="py-1.5 pr-3 text-right whitespace-nowrap">{formatSizeMb(model.size_mb, t, lang)}</td>
                    <td className="py-1.5 text-[#A1A1AA]">{model.license}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {locked && <div className="text-[11px] text-[#71717A] mt-2">{t('models.lockedNote')}</div>}
        </>
      )}

      {showBanner && (
        <div className="mt-4 bg-[#8B5CF6]/10 border border-[#8B5CF6]/40 rounded-lg px-4 py-3 text-sm text-[#D4D4D8] space-y-2">
          {emb?.reindex_needed && (
            <div className="flex gap-2">
              <AlertTriangle size={16} className="text-[#A78BFA] shrink-0 mt-0.5" />
              <span>{t('models.reindex.needed', { db_model: emb.db_model, target_model: emb.target_model })}</span>
            </div>
          )}

          {running && job && (
            <div>
              <div className="flex items-center justify-between text-xs text-[#A1A1AA] gap-3">
                <span className="flex items-center gap-1.5 min-w-0">
                  <Loader2 size={12} className="animate-spin shrink-0" />
                  <span className="truncate">{t('models.reindex.running', { percent: clampPercent(job.percent) })}{job.message ? ` — ${job.message}` : ''}</span>
                </span>
                <button onClick={cancel} className="px-2.5 py-1 rounded-lg bg-[#27272A] hover:bg-red-500/20 text-[#A1A1AA] hover:text-red-300 flex items-center gap-1 shrink-0">
                  <X size={12} />{t('models.reindex.cancel')}
                </button>
              </div>
              <div className="h-1 bg-[#27272A] rounded-full overflow-hidden mt-2">
                <div className="h-full bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] transition-all" style={{ width: `${clampPercent(job.percent)}%` }} />
              </div>
            </div>
          )}

          {result?.kind === 'done' && (
            <div className="text-xs text-emerald-300 flex items-center gap-1.5"><CheckCircle2 size={13} />{t('models.reindex.done')}</div>
          )}
          {result?.kind === 'busy' && <div className="text-xs text-amber-300">{t('models.reindex.busy')}</div>}
          {result?.kind === 'unsupported' && <div className="text-xs text-amber-300">{t('models.reindex.unsupported')}</div>}
          {result?.kind === 'error' && (
            <div className="text-xs text-red-300 break-words space-y-1">
              {result.message && <div>{t('models.reindex.error', { message: result.message })}</div>}
              {result.videos_missing > 0 && <div>{t('models.reindex.missing', { n: result.videos_missing })}</div>}
            </div>
          )}

          {!running && emb?.reindex_needed && (
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => start(false)}
                disabled={starting}
                className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white text-xs font-semibold flex items-center gap-1.5 disabled:opacity-50"
              >
                {starting ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                {starting ? t('models.reindex.starting') : t('models.reindex.start')}
              </button>
              {result?.kind === 'error' && result.videos_missing > 0 && (
                <button
                  onClick={() => start(true)}
                  disabled={starting}
                  className="px-3 py-1.5 rounded-lg bg-[#27272A] hover:bg-[#3F3F46] text-[#FAFAFA] text-xs font-semibold disabled:opacity-50"
                >
                  {t('models.reindex.force')}
                </button>
              )}
            </div>
          )}

          {(emb?.archived_models?.length ?? 0) > 0 && (
            <div className="text-[11px] text-[#71717A]">{t('models.reindex.archived', { list: emb.archived_models.join(', ') })}</div>
          )}
        </div>
      )}
    </div>
  );
}
