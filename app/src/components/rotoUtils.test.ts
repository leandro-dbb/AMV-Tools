import { describe, expect, it } from 'vitest';
import type { RotoSuspect } from '../api/roto';
import { DICT } from '../i18n';
import {
  REASON_KEYS, fitXform, frameToScreen, isTypingTarget, lockedFrames, lumToAlpha, matteVersion, nextSuspect,
  pendingSuspects, preloadOrder, prevSuspect, screenToFrame, settingsKey, stepRadius, thinPoints, timelineFrameAt,
  zoomAround,
} from './rotoUtils';

const S = (frame: number, score = 0.5): RotoSuspect => ({ frame, score, reasons: ['lost'] });

describe('suspect navigation', () => {
  const suspects = [S(30), S(5), S(12)];

  it('goes to the next suspect after the current frame, wrapping around', () => {
    expect(nextSuspect(suspects, 0)).toBe(5);
    expect(nextSuspect(suspects, 5)).toBe(12);
    expect(nextSuspect(suspects, 12)).toBe(30);
    expect(nextSuspect(suspects, 30)).toBe(5);
  });

  it('goes to the previous suspect, wrapping around', () => {
    expect(prevSuspect(suspects, 31)).toBe(30);
    expect(prevSuspect(suspects, 12)).toBe(5);
    expect(prevSuspect(suspects, 5)).toBe(30);
  });

  it('returns null without suspects', () => {
    expect(nextSuspect([], 3)).toBeNull();
    expect(prevSuspect([], 3)).toBeNull();
  });

  it('drops the frames the user locked on the active layer', () => {
    expect(pendingSuspects(suspects, [12]).map((s) => s.frame)).toEqual([30, 5]);
    expect(pendingSuspects(suspects, [])).toBe(suspects);
    expect(lockedFrames({ a: [1, 2] }, 'a')).toEqual([1, 2]);
    expect(lockedFrames({ a: [1, 2] }, 'b')).toEqual([]);
    expect(lockedFrames({ a: [1] }, null)).toEqual([]);
  });

  it('has a FR + EN label for every reason', () => {
    for (const key of Object.values(REASON_KEYS)) {
      expect(DICT[key]?.fr).toBeTruthy();
      expect(DICT[key]?.en).toBeTruthy();
    }
  });
});

describe('viewer transform', () => {
  it('fits the frame centred in the view', () => {
    const xf = fitXform(1000, 500, 1920, 1080, 1);
    expect(xf.scale).toBeCloseTo(500 / 1080);
    const [x0, y0] = frameToScreen(0, 0, xf);
    const [x1, y1] = frameToScreen(1920, 1080, xf);
    expect(y0).toBeCloseTo(0);
    expect(y1).toBeCloseTo(500);
    expect(x0 + x1).toBeCloseTo(1000); // centred horizontally
  });

  it('round-trips screen <-> frame coordinates', () => {
    const xf = { scale: 0.37, tx: 12.5, ty: -40 };
    const [sx, sy] = frameToScreen(640, 360, xf);
    const [fx, fy] = screenToFrame(sx, sy, xf);
    expect(fx).toBeCloseTo(640);
    expect(fy).toBeCloseTo(360);
  });

  it('zooms around the cursor (the point under it stays put)', () => {
    const xf = { scale: 0.5, tx: 10, ty: 20 };
    const before = screenToFrame(300, 200, xf);
    const z = zoomAround(xf, 300, 200, 2);
    expect(z.scale).toBeCloseTo(1);
    const after = screenToFrame(300, 200, z);
    expect(after[0]).toBeCloseTo(before[0]);
    expect(after[1]).toBeCloseTo(before[1]);
  });

  it('clamps the zoom', () => {
    expect(zoomAround({ scale: 30, tx: 0, ty: 0 }, 0, 0, 10).scale).toBe(32);
    expect(zoomAround({ scale: 0.06, tx: 0, ty: 0 }, 0, 0, 0.1).scale).toBe(0.05);
  });

  it('maps a timeline position to a frame', () => {
    expect(timelineFrameAt(0, 200, 10)).toBe(0);
    expect(timelineFrameAt(199.9, 200, 10)).toBe(9);
    expect(timelineFrameAt(500, 200, 10)).toBe(9);
    expect(timelineFrameAt(-5, 200, 10)).toBe(0);
    expect(timelineFrameAt(45, 200, 10)).toBe(2);
  });
});

describe('cache keys', () => {
  const base = { version: 9, frame_versions: { '3': 4 }, settings: { keep_ink: true, softness: 0, grow_px: 0, temporal: 0 } };

  it('uses the per-frame version, else the global one', () => {
    expect(matteVersion(base, 3)).toBe(4);
    expect(matteVersion(base, 7)).toBe(9);
  });

  it('uses the global version when smoothing over time is on', () => {
    expect(matteVersion({ ...base, settings: { ...base.settings, temporal: 0.5 } }, 3)).toBe(9);
  });

  it('changes the settings key with any edge setting', () => {
    const k = settingsKey(base.settings);
    expect(settingsKey({ ...base.settings, keep_ink: false })).not.toBe(k);
    expect(settingsKey({ ...base.settings, softness: 0.25 })).not.toBe(k);
    expect(settingsKey({ ...base.settings, grow_px: -2 })).not.toBe(k);
    expect(settingsKey({ ...base.settings, temporal: 0.1 })).not.toBe(k);
    expect(settingsKey(null)).toBe('0');
  });
});

describe('misc helpers', () => {
  it('knows when the user is typing', () => {
    expect(isTypingTarget({ tagName: 'INPUT', type: 'text' })).toBe(true);
    expect(isTypingTarget({ tagName: 'INPUT' })).toBe(true);
    expect(isTypingTarget({ tagName: 'TEXTAREA' })).toBe(true);
    expect(isTypingTarget({ tagName: 'DIV', isContentEditable: true })).toBe(true);
    expect(isTypingTarget({ tagName: 'INPUT', type: 'range' })).toBe(false);
    expect(isTypingTarget({ tagName: 'INPUT', type: 'checkbox' })).toBe(false);
    expect(isTypingTarget({ tagName: 'BUTTON' })).toBe(false);
    expect(isTypingTarget(null)).toBe(false);
  });

  it('thins stroke points but keeps both ends', () => {
    const pts: [number, number][] = [[0, 0], [0.2, 0], [0.4, 0], [3, 0], [3.1, 0]];
    expect(thinPoints(pts, 1)).toEqual([[0, 0], [3, 0], [3.1, 0]]);
    expect(thinPoints([[1, 1]], 1)).toEqual([[1, 1]]);
  });

  it('steps the brush size', () => {
    expect(stepRadius(5, 1)).toBe(6);
    expect(stepRadius(5, -1)).toBe(4);
    expect(stepRadius(1, -1)).toBe(1);
    expect(stepRadius(40, 1)).toBe(46);
    expect(stepRadius(200, 1)).toBe(200);
  });

  it('turns a gray matte into an alpha mask', () => {
    const d = new Uint8ClampedArray([200, 200, 200, 255, 0, 0, 0, 255]);
    lumToAlpha(d);
    expect(Array.from(d)).toEqual([255, 255, 255, 200, 255, 255, 255, 0]);
  });

  it('preloads the nearest frames first, within bounds', () => {
    expect(preloadOrder(5, 10, 2, 3)).toEqual([6, 4, 7, 3, 8]);
    expect(preloadOrder(0, 3, 4, 4)).toEqual([1, 2]);
  });
});
