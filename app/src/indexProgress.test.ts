import { describe, expect, it } from 'vitest';
import { etaSeconds, fallbackLabel, nextFailures, stageLabel } from './indexProgress';
import { DICT } from './i18n';
import type { TFunc } from './i18n';

const tFr: TFunc = (key, vars) => {
  let s = DICT[key]?.fr ?? key;
  for (const [k, v] of Object.entries(vars ?? {})) s = s.replace(`{${k}}`, String(v));
  return s;
};

describe('stageLabel', () => {
  it('translates the pipeline stage messages', () => {
    expect(stageLabel('Detecting cuts (pass 1)', tFr)).toBe('Détection des coupes');
    expect(stageLabel('Sub-segmenting (decoding 412 samples in one pass)', tFr)).toBe('Découpage des longues scènes (lecture de 412 images)');
    expect(stageLabel('Tagged 16/68', tFr)).toBe('Tags 16/68');
    expect(stageLabel('Embedded 8/68', tFr)).toBe('Index de recherche 8/68');
    expect(stageLabel('Failed: moov atom not found', tFr)).toBe('Échec : moov atom not found');
  });

  it('passes unknown messages through and ignores empty ones', () => {
    expect(stageLabel('Something new', tFr)).toBe('Something new');
    expect(stageLabel(undefined, tFr)).toBe('');
  });

  it('has every stage key in both languages', () => {
    for (const k of Object.keys(DICT).filter((k) => k.startsWith('journey.'))) {
      expect(DICT[k].fr, k).toBeTruthy();
      expect(DICT[k].en, k).toBeTruthy();
    }
  });
});

describe('etaSeconds', () => {
  it('waits for some signal, then extrapolates', () => {
    expect(etaSeconds(1, 60)).toBeNull();
    expect(etaSeconds(50, 2)).toBeNull();
    expect(etaSeconds(25, 60)).toBe(180);
    expect(etaSeconds(100, 60)).toBeNull();
  });
});

describe('nextFailures', () => {
  const fail = { type: 'indexing', failed: true, video: 'bad.mkv', path: '/m/bad.mkv', message: 'Failed: invalid data' };
  it('collects failed files once, and resets on a new run', () => {
    let list = nextFailures([], { type: 'indexing', video: 'a.mkv', percent: 1 } as never, 'idle');
    list = nextFailures(list, fail, 'indexing');
    list = nextFailures(list, fail, 'indexing');
    expect(list).toEqual([{ video: 'bad.mkv', path: '/m/bad.mkv', reason: 'invalid data' }]);
    list = nextFailures(list, { type: 'idle' }, 'indexing');
    expect(list).toHaveLength(1);
    list = nextFailures(list, { type: 'indexing', video: 'a.mkv' }, 'idle');
    expect(list).toEqual([]);
  });
  it('keeps a failure reported as the very first event of a run', () => {
    expect(nextFailures([], fail, 'idle')).toHaveLength(1);
  });
});

describe('fallbackLabel', () => {
  it('localises the sidecar CPU-fallback reasons', () => {
    expect(fallbackLabel('No GPU runtime detected (torch is not installed); running on CPU.', tFr)).toBe("Aucun moteur GPU n'est installé.");
    expect(fallbackLabel('No supported GPU detected; running on CPU.', tFr)).toBe("Aucune carte graphique compatible n'a été détectée.");
    expect(fallbackLabel('Apple MPS was requested, but it is unavailable; fell back to CPU.', tFr)).toBe('Apple MPS a été choisi, mais est indisponible.');
    expect(fallbackLabel('something else', tFr)).toBe('something else');
  });
});
