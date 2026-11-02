import { describe, expect, it } from 'vitest';
import { claimSearchRequest } from './searchplus';

describe('claimSearchRequest', () => {
  it('runs a cross-tab request once, even when the Search tab remounts', () => {
    expect(claimSearchRequest(101)).toBe(true);
    // Back to Search later: App hands the same request again.
    expect(claimSearchRequest(101)).toBe(false);
    expect(claimSearchRequest(101)).toBe(false);
    expect(claimSearchRequest(102)).toBe(true);
  });
});
