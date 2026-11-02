import { describe, expect, it } from 'vitest';
import { requiredModel } from './api/finish';

describe('requiredModel', () => {
  it('follows the tier default the sidecar will run (slow motion: RIFE 4.26)', () => {
    expect(requiredModel('interpolate', {}, { interpolate: 'rife-4.26' })).toBe('rife-4.26');
    expect(requiredModel('depth', {}, { depth: 'video-depth-anything-s' })).toBe('video-depth-anything-s');
  });
  it('keeps an explicit model and the old defaults without capabilities', () => {
    expect(requiredModel('upscale', { model: 'realesrgan-x4plus-anime' }, { upscale: 'x' })).toBe('realesrgan-x4plus-anime');
    expect(requiredModel('interpolate', {})).toBe('rife-4.25');
    expect(requiredModel('matte', {}, { interpolate: 'rife-4.26' })).toBeNull();
  });
});
