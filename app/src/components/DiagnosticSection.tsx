import { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Check,
  Copy,
  Download,
  FolderOpen,
  Loader2,
  RefreshCw,
} from 'lucide-react';
import { api } from '../api/client';
import type { DiagnosticsReport, ModelDownloadResult } from '../api/types';
import { formatBytes, reportMarkdown } from '../diagnostics';
import { useT } from '../i18n';
import AutotuneSection from './AutotuneSection';

export default function DiagnosticSection() {
  const t = useT();
  const [report, setReport] = useState<DiagnosticsReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const [logsError, setLogsError] = useState(false);
  const [retrying, setRetrying] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<{ key: string; error: string; hint?: string } | null>(null);
  const [downloadOk, setDownloadOk] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setError(null);
      setReport(await api.diagnostics());
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function copyReport() {
    if (!report) return;
    try {
      await navigator.clipboard.writeText(reportMarkdown(report));
      setCopied(true);
      setCopyError(false);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopyError(true);
    }
  }

  async function openLogs() {
    const dir = report?.logs.dir;
    if (!dir || !window.amvBridge) {
      setLogsError(true);
      return;
    }
    setLogsError(false);
    if (window.amvBridge.openPath) {
      await window.amvBridge.openPath(dir);
    } else {
      window.amvBridge.reveal(dir);
    }
  }

  async function retryModel(key: string) {
    setRetrying(key);
    setDownloadError(null);
    setDownloadOk(null);
    try {
      const result: ModelDownloadResult = await api.downloadModel(key);
      if (result.ok) {
        setDownloadOk(key);
        await load();
      } else {
        setDownloadError({ key, error: result.error ?? 'unknown error', hint: result.hint });
      }
    } catch (err) {
      setDownloadError({ key, error: (err as Error).message });
    } finally {
      setRetrying(null);
    }
  }

  if (!report && !error) {
    return (
      <div className="text-[#71717A] flex items-center gap-2">
        <Loader2 size={14} className="animate-spin" /> {t('settings.diagnostic.loading')}
      </div>
    );
  }

  if (error && !report) {
    return (
      <div className="bg-red-500/10 border border-red-500/40 text-red-300 text-sm rounded-lg px-4 py-3">
        {t('settings.diagnostic.error', { message: error })}
        <button onClick={load} className="ml-3 underline">{t('settings.diagnostic.refresh')}</button>
      </div>
    );
  }

  if (!report) return null;

  const torch = report.torch.version ?? t('settings.diagnostic.notInstalled');
  const ffmpeg = report.ffmpeg.version ?? t('settings.diagnostic.notInstalled');
  const disk = report.data_dir;

  return (
    <div className="flex flex-col gap-4">
      <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-[#FAFAFA] font-semibold">
              <Activity size={17} className="text-[#8B5CF6]" />
              {t('settings.diagnostic.title')}
            </div>
            <div className="text-[#71717A] text-sm mt-1 max-w-2xl">{t('settings.diagnostic.hint')}</div>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              onClick={load}
              className="bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-3 py-2 rounded-lg text-xs flex items-center gap-2"
            >
              <RefreshCw size={13} /> {t('settings.diagnostic.refresh')}
            </button>
            <button
              onClick={copyReport}
              className="bg-gradient-to-r from-[#8B5CF6] to-[#EC4899] text-white px-3 py-2 rounded-lg text-xs font-semibold flex items-center gap-2"
            >
              {copied ? <Check size={13} /> : <Copy size={13} />}
              {copied ? t('settings.diagnostic.copied') : t('settings.diagnostic.copy')}
            </button>
            <button
              onClick={openLogs}
              className="bg-[#27272A] hover:bg-[#3f3f46] text-[#FAFAFA] px-3 py-2 rounded-lg text-xs flex items-center gap-2"
            >
              <FolderOpen size={13} /> {t('settings.diagnostic.openLogs')}
            </button>
          </div>
        </div>
        {copyError && <div className="text-amber-300 text-xs mt-3">{t('settings.diagnostic.copyFailed')}</div>}
        {logsError && <div className="text-amber-300 text-xs mt-3">{t('settings.diagnostic.logsUnavailable')}</div>}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <Info label={t('settings.diagnostic.appVersion')} value={report.app.version} />
        <Info label={t('settings.diagnostic.python')} value={report.python} />
        <Info label={t('settings.diagnostic.platform')} value={report.platform} />
        <Info label={t('settings.diagnostic.torch')} value={torch} />
        <Info label={t('settings.diagnostic.ffmpeg')} value={ffmpeg} />
        <Info label={t('settings.diagnostic.requested')} value={report.device.requested} />
        <Info
          label={t('settings.diagnostic.active')}
          value={`${report.device.active}${report.device.name ? ` — ${report.device.name}` : ''}`}
        />
        <Info
          label={t('settings.diagnostic.dataDir')}
          value={disk.error
            ? disk.error
            : disk.free_gb != null
              ? `${disk.path} · ${t('settings.diagnostic.diskFree', {
                  free: disk.free_gb,
                  total: disk.total_gb ?? '?',
                  used: disk.used_percent ?? '?',
                })}`
              : disk.path}
        />
      </div>

      {report.device.fallback_reason && (
        <div className="bg-amber-500/10 border border-amber-500/40 text-amber-200 text-sm rounded-xl px-4 py-3 flex items-start gap-2">
          <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
          <div>
            <div className="font-semibold">{t('settings.diagnostic.fallback')}</div>
            <div className="text-amber-200/90">{report.device.fallback_reason}</div>
          </div>
        </div>
      )}

      <AutotuneSection />

      <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
        <div className="text-[#FAFAFA] font-semibold mb-3">{t('settings.diagnostic.models')}</div>
        <div className="flex flex-col gap-2">
          {report.models.map((model) => (
            <div key={model.key} className="border border-[#27272A] rounded-lg px-3 py-2.5">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[#FAFAFA] text-sm font-medium">{model.label}</span>
                    <span className={`text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded ${model.present ? 'bg-emerald-500/15 text-emerald-300' : 'bg-[#27272A] text-[#A1A1AA]'}`}>
                      {model.present ? t('settings.diagnostic.model.present') : t('settings.diagnostic.model.absent')}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#27272A] text-[#71717A]">
                      {model.enabled ? t('settings.diagnostic.model.enabled') : t('settings.diagnostic.model.disabled')}
                    </span>
                    {model.loaded && (
                      <span className="text-[10px] text-[#A78BFA]">{t('settings.diagnostic.model.loaded')}</span>
                    )}
                  </div>
                  <div className="text-[#71717A] text-xs font-mono truncate mt-1">
                    {model.repo_id}{model.size_bytes != null ? ` · ${formatBytes(model.size_bytes)}` : ''}
                  </div>
                </div>
                <button
                  onClick={() => retryModel(model.key)}
                  disabled={retrying !== null}
                  className="flex-shrink-0 bg-[#27272A] hover:bg-[#3f3f46] disabled:opacity-50 text-[#FAFAFA] px-3 py-1.5 rounded-lg text-xs flex items-center gap-1.5"
                >
                  {retrying === model.key
                    ? <Loader2 size={12} className="animate-spin" />
                    : <Download size={12} />}
                  {retrying === model.key
                    ? t('settings.diagnostic.model.retrying')
                    : t('settings.diagnostic.model.retry')}
                </button>
              </div>
              {downloadOk === model.key && (
                <div className="text-emerald-300 text-xs mt-2">{t('settings.diagnostic.model.downloaded')}</div>
              )}
              {downloadError?.key === model.key && (
                <div className="text-red-300 text-xs mt-2">
                  {t('settings.diagnostic.model.error', { error: downloadError.error })}
                  {downloadError.hint && <div className="text-[#A1A1AA] mt-1">{downloadError.hint}</div>}
                </div>
              )}
              {model.error && downloadError?.key !== model.key && (
                <div className="text-amber-300 text-xs mt-2 break-words">{model.error}</div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border-2 border-[#27272A] rounded-xl p-5">
        <div className="text-[#FAFAFA] font-semibold mb-2">{t('settings.diagnostic.recentErrors')}</div>
        {report.logs.recent_errors.length === 0 ? (
          <div className="text-[#71717A] text-sm">{t('settings.diagnostic.noErrors')}</div>
        ) : (
          <pre className="text-xs text-[#A1A1AA] bg-[#0F0F11] border border-[#27272A] rounded-lg p-3 max-h-64 overflow-auto whitespace-pre-wrap">
            {report.logs.recent_errors.map((e) => `[${e.time}] ${e.level} ${e.logger}: ${e.message}`).join('\n')}
          </pre>
        )}
        <div className="text-[#71717A] text-xs font-mono mt-2 break-all">{report.logs.path}</div>
      </div>
    </div>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gradient-to-br from-[#18181B] to-[#0F0F11] border border-[#27272A] rounded-lg px-4 py-3 min-w-0">
      <div className="text-[#71717A] text-[10px] uppercase tracking-wider">{label}</div>
      <div className="text-[#FAFAFA] text-sm font-mono truncate" title={value}>{value}</div>
    </div>
  );
}
