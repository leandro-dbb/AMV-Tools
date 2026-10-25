import { afterEach, describe, expect, it, vi } from 'vitest';
import { modelTiersApi } from './api/client';
import type { ModelTiersInfo } from './api/types';
import { DICT, type Lang, type TFunc } from './i18n';
import { modelsStrings } from './i18n/strings.models';
import {
  TIER_ROLES, autoLabel, clampPercent, formatGb, formatSizeMb, isNotFound, normalizeJob, parseApiError,
  tierLabel, tierOptions,
} from './modelTiers';

// Same lookup as the real t() in i18n/index.tsx, without React.
function makeT(lang: Lang): TFunc {
  return (key, vars) => {
    const entry = DICT[key];
    let s = entry ? (entry[lang] || entry.en) : key;
    if (vars) for (const [k, v] of Object.entries(vars)) s = s.split(`{${k}}`).join(String(v));
    return s;
  };
}
const en = makeT('en');
const fr = makeT('fr');

function info(partial: Partial<ModelTiersInfo> = {}): ModelTiersInfo {
  return {
    platform: 'nvidia',
    detected_gb: 16,
    auto_tier: '16gb',
    selected: 'auto',
    effective_tier: '16gb',
    tiers: ['fast', '8gb', '12gb', '16gb', '24gb', 'max'].map((id) => ({
      id,
      min_gb: id === 'fast' ? null : id === 'max' ? 32 : Number(id.replace('gb', '')),
      models: {},
    })),
    embedding: { db_model: 'a', target_model: 'b', reindex_needed: false, archived_models: [], job: null },
    ...partial,
  };
}

describe('model tiers strings', () => {
  it('are merged into the i18n dictionary with FR and EN text', () => {
    for (const [key, entry] of Object.entries(modelsStrings)) {
      expect(DICT[key]).toBe(entry);
      expect(entry.en.trim()).not.toBe('');
      expect(entry.fr.trim()).not.toBe('');
    }
  });

  it('cover every role and the fixed tier labels', () => {
    for (const role of TIER_ROLES) expect(modelsStrings[`models.role.${role}`]).toBeDefined();
    for (const id of ['fast', 'max', 'custom']) expect(modelsStrings[`models.tier.${id}`]).toBeDefined();
  });
});

describe('tierLabel', () => {
  it('formats NVIDIA, Mac and named tiers', () => {
    expect(tierLabel('8gb', en)).toBe('8 GB');
    expect(tierLabel('24gb', fr)).toBe('24 Go');
    expect(tierLabel('mac16', en)).toBe('Mac 16 GB');
    expect(tierLabel('mac32', fr)).toBe('Mac 32 Go et plus');
    expect(tierLabel('fast', en)).toBe('Fast');
    expect(tierLabel('max', fr)).toBe('Qualité max');
    expect(tierLabel('custom', en)).toBe('Custom');
    expect(tierLabel('weird', en)).toBe('weird');
  });
});

describe('autoLabel', () => {
  it('shows the detected VRAM and the tier it maps to', () => {
    expect(autoLabel(info(), en, 'en')).toBe('Auto (detected: 16 GB VRAM → 16 GB)');
  });

  it('uses unified memory wording on a Mac and a decimal comma in French', () => {
    const mac = info({ platform: 'mac', detected_gb: 18.5, auto_tier: 'mac16' });
    expect(autoLabel(mac, en, 'en')).toBe('Auto (detected: 18.5 GB unified memory → Mac 16 GB)');
    expect(autoLabel(mac, fr, 'fr')).toBe('Auto (détecté : 18,5 Go de mémoire unifiée → Mac 16 Go)');
  });

  it('falls back when no memory was detected', () => {
    expect(autoLabel(info({ platform: 'cpu', detected_gb: null, auto_tier: 'fast' }), en, 'en')).toBe('Auto (Fast)');
  });
});

describe('tierOptions', () => {
  it('lists auto, every tier and custom, flagging tiers above the detected memory', () => {
    const opts = tierOptions(info(), en, 'en');
    expect(opts.map((o) => o.value)).toEqual(['auto', 'fast', '8gb', '12gb', '16gb', '24gb', 'max', 'custom']);
    expect(opts.find((o) => o.value === '16gb')?.label).toBe('16 GB');
    expect(opts.find((o) => o.value === '24gb')?.label).toBe('24 GB (more than detected)');
    expect(opts.find((o) => o.value === 'fast')?.label).toBe('Fast');
  });

  it('does not duplicate auto / custom if the sidecar lists them', () => {
    const i = info({ tiers: [{ id: 'auto', min_gb: null, models: {} }, { id: 'fast', min_gb: null, models: {} }, { id: 'custom', min_gb: null, models: {} }] });
    expect(tierOptions(i, en, 'en').map((o) => o.value)).toEqual(['auto', 'fast', 'custom']);
  });
});

describe('formatting', () => {
  it('formats GB and model sizes', () => {
    expect(formatGb(16, 'en')).toBe('16');
    expect(formatGb(23.64, 'en')).toBe('23.6');
    expect(formatGb(23.64, 'fr')).toBe('23,6');
    expect(formatSizeMb(350.4, en, 'en')).toBe('350 MB');
    expect(formatSizeMb(2.5, fr, 'fr')).toBe('2,5 Mo');
    expect(formatSizeMb(18, en, 'en')).toBe('18 MB');
    expect(formatSizeMb(1200, fr, 'fr')).toBe('1,2 Go');
    expect(formatSizeMb(0, en, 'en')).toBe('—');
    expect(formatSizeMb(undefined, en, 'en')).toBe('—');
  });

  it('clamps percentages', () => {
    expect(clampPercent(42.4)).toBe(42);
    expect(clampPercent(150)).toBe(100);
    expect(clampPercent(-3)).toBe(0);
    expect(clampPercent(undefined)).toBe(0);
  });
});

describe('sidecar errors and job status', () => {
  it('detects a missing endpoint on an older sidecar', () => {
    expect(isNotFound(new Error('API 404: {"detail":"Not Found"}'))).toBe(true);
    expect(isNotFound(new Error('API 500: boom'))).toBe(false);
    expect(isNotFound(new Error('Failed to fetch'))).toBe(false);
  });

  it('extracts the message and missing videos from the response body', () => {
    expect(parseApiError(new Error('API 409: {"detail":"indexing is running"}'))).toEqual({
      status: 409, message: 'indexing is running', videos_missing: 0,
    });
    expect(parseApiError(new Error('API 400: {"detail":{"error":"videos missing","videos_missing":3}}'))).toEqual({
      status: 400, message: 'videos missing', videos_missing: 3,
    });
    expect(parseApiError(new Error('API 400: {"ok":false,"error":"nope","videos_missing":2}'))).toEqual({
      status: 400, message: 'nope', videos_missing: 2,
    });
    expect(parseApiError(new Error('API 502: Bad gateway'))).toEqual({ status: 502, message: 'Bad gateway', videos_missing: 0 });
    expect(parseApiError(new Error('offline'))).toEqual({ status: null, message: 'offline', videos_missing: 0 });
  });

  it('accepts the job directly or wrapped in { job }', () => {
    const job = { running: true, percent: 10, message: 'x', error: null };
    expect(normalizeJob(job)).toEqual(job);
    expect(normalizeJob({ job })).toEqual(job);
    expect(normalizeJob({ job: null })).toBeNull();
    expect(normalizeJob(null)).toBeNull();
    expect(normalizeJob({ foo: 1 })).toBeNull();
  });
});

describe('modelTiersApi', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('posts the re-index request and surfaces a 404 as not found', async () => {
    vi.stubGlobal('window', { __AMV_API_PORT__: 9123 });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }))
      .mockResolvedValueOnce(new Response('{"detail":"Not Found"}', { status: 404 }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(modelTiersApi.reembed({ model: 'm', force: true })).resolves.toEqual({ ok: true });
    expect(fetchMock).toHaveBeenCalledWith(
      'http://127.0.0.1:9123/api/models/reembed',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ model: 'm', force: true }) }),
    );
    const err = await modelTiersApi.tiers().catch((e) => e);
    expect(isNotFound(err)).toBe(true);
  });
});
