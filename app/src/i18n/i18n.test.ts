import { describe, expect, it } from 'vitest';
import { DICT } from './index';

describe('i18n dictionary', () => {
  const entries = Object.entries(DICT);

  it('contains strings', () => {
    expect(entries.length).toBeGreaterThan(100);
  });

  it('has a non-empty FR and EN translation for every key', () => {
    const missing: string[] = [];
    for (const [key, entry] of entries) {
      if (!entry || typeof entry.en !== 'string' || entry.en.trim() === '') {
        missing.push(`${key} (en)`);
      }
      if (!entry || typeof entry.fr !== 'string' || entry.fr.trim() === '') {
        missing.push(`${key} (fr)`);
      }
    }
    expect(missing).toEqual([]);
  });

  it('keeps placeholders consistent between FR and EN', () => {
    const placeholders = (text: string) =>
      [...text.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();
    const mismatched = entries
      .filter(([, entry]) => {
        const en = placeholders(entry.en);
        const fr = placeholders(entry.fr);
        return en.length !== fr.length || en.some((p, i) => p !== fr[i]);
      })
      .map(([key]) => key);
    expect(mismatched).toEqual([]);
  });
});
