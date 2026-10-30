import { describe, expect, it } from 'vitest';
import { exportDoneLabel } from './derushExport';
import { DICT } from './i18n';
import type { TFunc } from './i18n';

const tFr: TFunc = (key, vars) => {
  let s = DICT[key]?.fr ?? key;
  for (const [k, v] of Object.entries(vars ?? {})) s = s.replace(`{${k}}`, String(v));
  return s;
};

describe('exportDoneLabel', () => {
  it('is translated (was the raw English "2 exported")', () => {
    expect(exportDoneLabel({ type: 'derush_export', done: true, ok: true, message: '2 exported', exported_count: 2, failed_count: 0 }, tFr)).toBe('Export terminé : 2 clip(s)');
    expect(exportDoneLabel({ type: 'derush_export', done: true, ok: false, exported_count: 1, failed_count: 1 }, tFr)).toBe('Export terminé : 1 clip(s) · 1 en échec');
  });
  it('keeps other messages', () => {
    expect(exportDoneLabel({ type: 'derush_export', done: true, ok: false, message: 'disk full' }, tFr)).toBe('disk full');
    expect(exportDoneLabel({ type: 'derush_export', done: true, ok: true }, tFr)).toBe('Export terminé');
  });
});
