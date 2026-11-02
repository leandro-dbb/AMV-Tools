import { describe, expect, it } from 'vitest';
import { extraPerEpisode } from './PreciseCutsHint';

describe('extraPerEpisode', () => {
  it('says seconds under a minute instead of rounding up to 1 min', () => {
    expect(extraPerEpisode(13.5)).toBe('14 s');   // the M1 Pro measure
    expect(extraPerEpisode(0.2)).toBe('1 s');
    expect(extraPerEpisode(75)).toBe('1 min');
    expect(extraPerEpisode(200)).toBe('3 min');
  });
});
