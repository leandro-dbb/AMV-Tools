import { describe, expect, it } from 'vitest';
import { searchDefaults } from './searchplus';

describe('searchDefaults (Settings > Search used by the Search tab)', () => {
  it('maps the saved settings', () => {
    expect(searchDefaults({ threshold: 0.22, max_results: 1000, sort: 'duration' })).toEqual({ threshold: 22, sort: 'duration', maxResults: 1000 });
  });
  it('falls back to the historical defaults on missing or odd values', () => {
    expect(searchDefaults(undefined)).toEqual({ threshold: 10, sort: 'relevance', maxResults: 200 });
    expect(searchDefaults({ threshold: Number.NaN, max_results: 0, sort: 'weird' })).toEqual({ threshold: 10, sort: 'relevance', maxResults: 200 });
  });
});
