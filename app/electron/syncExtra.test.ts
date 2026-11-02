import { describe, expect, it } from 'vitest';
import { syncExtraArgs } from './syncExtra';

describe('syncExtraArgs', () => {
  it('keeps the installed backend on every platform', () => {
    expect(syncExtraArgs('linux', 'cu126')).toEqual(['--extra', 'cu126']);
    expect(syncExtraArgs('win32', 'cu130')).toEqual(['--extra', 'cu130']);
    expect(syncExtraArgs('darwin', 'mps')).toEqual(['--extra', 'mps']);
    expect(syncExtraArgs('win32', 'dml')).toEqual(['--extra', 'dml', '--python', '3.12']);
  });

  it('bootstraps Linux with the CPU extra (base deps resolve torch 2.4.1 there, no cp313 wheel)', () => {
    expect(syncExtraArgs('linux', null)).toEqual(['--extra', 'cpu']);
  });

  it('bootstraps macOS and Windows with the base deps', () => {
    expect(syncExtraArgs('darwin', null)).toEqual([]);
    expect(syncExtraArgs('win32', null)).toEqual([]);
  });
});
