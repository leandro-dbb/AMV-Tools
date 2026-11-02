import { describe, expect, it } from 'vitest';
import { finalCutUnsupported } from './timelineSources';

describe('finalCutUnsupported', () => {
  it('flags the containers Final Cut Pro cannot import', () => {
    // the M1 test library: two .mkv episodes, both offline in Final Cut
    expect(finalCutUnsupported([
      '/Users/x/media/dbs-beerus-01.mkv',
      'C:\\Anime\\Show 01.MKV',
      '/Users/x/clip.webm',
      '/Users/x/ok.mov',
      '/Users/x/ok.mp4',
      '/Users/x/ok.m4v',
    ])).toEqual(['/Users/x/media/dbs-beerus-01.mkv', 'C:\\Anime\\Show 01.MKV', '/Users/x/clip.webm']);
  });
});
