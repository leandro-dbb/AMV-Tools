// Pure helpers for the model tiers block of Settings > Models
// (ModelTiersSection): tier labels, size formatting, sidecar error parsing.
import type { ModelTierPlatform, ModelTierRole, ModelTiersInfo, ReembedJob } from './api/types';
import type { Lang, TFunc } from './i18n';

export const TIER_ROLES: ModelTierRole[] = ['embedding', 'tagger', 'upscale', 'interpolate', 'depth', 'mask_selector', 'mask_detector', 'mask_tracker'];

/** 16 → "16", 23.64 → "23.6" (FR: "23,6"). */
export function formatGb(gb: number, lang: Lang): string {
  const rounded = Math.round(gb * 10) / 10;
  const s = Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
  return lang === 'fr' ? s.replace('.', ',') : s;
}

/** Model size in MB → "2.5 MB" / "350 MB" / "1.2 GB" (FR: "2,5 Mo" / "350 Mo" / "1,2 Go"). */
export function formatSizeMb(mb: number | null | undefined, t: TFunc, lang: Lang): string {
  if (mb == null || !isFinite(mb) || mb <= 0) return '—';
  if (mb >= 1000) return t('models.size.gb', { n: formatGb(mb / 1000, lang) });
  return t('models.size.mb', { n: mb < 10 ? formatGb(mb, lang) : String(Math.round(mb)) });
}

/** Human label of a tier id: "fast", "8gb", "mac16", "max", "custom", "auto". */
export function tierLabel(id: string, t: TFunc): string {
  if (id === 'fast') return t('models.tier.fast');
  if (id === 'max') return t('models.tier.max');
  if (id === 'custom') return t('models.tier.custom');
  if (id === 'auto') return 'Auto';
  const nvidia = /^(\d+)gb$/i.exec(id);
  if (nvidia) return t('models.tier.gb', { gb: nvidia[1] });
  const mac = /^mac(\d+)$/i.exec(id);
  if (mac) return t(mac[1] === '32' ? 'models.tier.macGbPlus' : 'models.tier.macGb', { gb: mac[1] });
  return id;
}

/** "16 GB VRAM" / "16 GB unified memory" / "16 GB" for the detected memory. */
export function detectedLabel(platform: ModelTierPlatform, gb: number, t: TFunc, lang: Lang): string {
  const v = formatGb(gb, lang);
  if (platform === 'nvidia') return t('models.detected.vram', { gb: v });
  if (platform === 'mac') return t('models.detected.unified', { gb: v });
  return t('models.detected.ram', { gb: v });
}

/** Label of the "auto" option: "Auto (detected: 16 GB VRAM → 16 GB)". */
export function autoLabel(info: Pick<ModelTiersInfo, 'platform' | 'detected_gb' | 'auto_tier'>, t: TFunc, lang: Lang): string {
  const tier = tierLabel(info.auto_tier, t);
  if (info.detected_gb == null || !isFinite(info.detected_gb)) return t('models.tier.autoUnknown', { tier });
  return t('models.tier.auto', { detected: detectedLabel(info.platform, info.detected_gb, t, lang), tier });
}

export interface TierOption { value: string; label: string }

/** Options of the tier select: auto, every tier the sidecar lists, custom. */
export function tierOptions(info: ModelTiersInfo, t: TFunc, lang: Lang): TierOption[] {
  const opts: TierOption[] = [{ value: 'auto', label: autoLabel(info, t, lang) }];
  for (const tier of info.tiers) {
    if (tier.id === 'auto' || tier.id === 'custom') continue;
    let label = tierLabel(tier.id, t);
    if (tier.min_gb != null && info.detected_gb != null && tier.min_gb > info.detected_gb) {
      label = t('models.tier.overMemory', { tier: label });
    }
    opts.push({ value: tier.id, label });
  }
  opts.push({ value: 'custom', label: t('models.tier.custom') });
  return opts;
}

/** Parsed error thrown by the API client ("API 409: {...}"). */
export interface ApiErrorInfo {
  status: number | null;
  message: string;
  videos_missing: number;
  code?: string;
}

export function parseApiError(err: unknown): ApiErrorInfo {
  const raw = err instanceof Error ? err.message : String(err ?? '');
  const m = /^API (\d{3}):\s*([\s\S]*)$/.exec(raw);
  if (!m) return { status: null, message: raw, videos_missing: 0 };
  const status = Number(m[1]);
  let message = m[2].trim();
  let videosMissing = 0;
  let code: string | undefined;
  try {
    const body = JSON.parse(message);
    const detail = body && typeof body === 'object' && 'detail' in body ? body.detail : body;
    if (typeof detail === 'string') {
      message = detail;
    } else if (detail && typeof detail === 'object') {
      if (typeof detail.error === 'string') message = detail.error;
      else if (typeof detail.message === 'string') message = detail.message;
      if (typeof detail.videos_missing === 'number') videosMissing = detail.videos_missing;
      if (typeof detail.code === 'string') code = detail.code;
    }
  } catch { /* plain-text body */ }
  return code ? { status, message, videos_missing: videosMissing, code } : { status, message, videos_missing: videosMissing };
}

export function isNotFound(err: unknown): boolean {
  return parseApiError(err).status === 404;
}

/** GET /api/models/reembed may return the job itself or { job }. */
export function normalizeJob(x: unknown): ReembedJob | null {
  if (!x || typeof x !== 'object') return null;
  const o = x as Record<string, unknown>;
  if ('job' in o && !('running' in o)) return normalizeJob(o.job);
  if (typeof o.running !== 'boolean') return null;
  return o as unknown as ReembedJob;
}

/** Percent clamped to 0-100 and rounded, for the progress bar. */
export function clampPercent(p: number | null | undefined): number {
  if (p == null || !isFinite(p)) return 0;
  return Math.max(0, Math.min(100, Math.round(p)));
}
