import { afterEach, describe, expect, it, vi } from 'vitest';
import { droppedPaths } from './dropPaths';

afterEach(() => vi.unstubAllGlobals());

describe('droppedPaths', () => {
  it('asks the preload bridge (File.path is gone since Electron 32)', () => {
    const pathForFile = vi.fn((f: File) => `/media/Séries test/${f.name}`);
    vi.stubGlobal('window', { amvBridge: { pathForFile } });
    const files = [new File([], 'DBS 01.mkv'), new File([], 'BC 04.mkv')];
    expect(droppedPaths(files)).toEqual(['/media/Séries test/DBS 01.mkv', '/media/Séries test/BC 04.mkv']);
  });

  it('falls back to File.path and skips files without a path', () => {
    vi.stubGlobal('window', {});
    const withPath = Object.assign(new File([], 'a.mkv'), { path: '/x/a.mkv' });
    expect(droppedPaths([withPath, new File([], 'b.mkv')])).toEqual(['/x/a.mkv']);
  });

  it('ignores a bridge that throws (not a real file)', () => {
    vi.stubGlobal('window', { amvBridge: { pathForFile: () => { throw new Error('nope'); } } });
    expect(droppedPaths([new File([], 'a.mkv')])).toEqual([]);
  });
});
