import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api, authHeaders, getBaseUrl, withToken } from './client';

describe('API client', () => {
  beforeEach(() => {
    vi.stubGlobal('window', { __AMV_API_PORT__: 9123 });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('targets the sidecar port exposed on window', () => {
    expect(getBaseUrl()).toBe('http://127.0.0.1:9123');
  });

  it('falls back to the default sidecar port', () => {
    vi.stubGlobal('window', {});
    expect(getBaseUrl()).toBe('http://127.0.0.1:8731');
  });

  it('parses a successful JSON response', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ready: true, device: 'cpu' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const status = await api.status();

    expect(status.ready).toBe(true);
    expect(fetchMock).toHaveBeenCalledWith(
      'http://127.0.0.1:9123/api/status',
      expect.objectContaining({ headers: expect.objectContaining({ 'Content-Type': 'application/json' }) }),
    );
  });

  it('surfaces the HTTP status and response body on errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('sidecar crashed', { status: 503 })));
    await expect(api.status()).rejects.toThrow('API 503: sidecar crashed');
  });

  it('posts model download requests with the model key', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await api.downloadModel('wd_tagger');

    expect(fetchMock).toHaveBeenCalledWith(
      'http://127.0.0.1:9123/api/models/download',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ key: 'wd_tagger' }),
      }),
    );
  });
});

describe('sidecar token', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('sends X-AMV-Token on fetches and ?token= on media/socket URLs', async () => {
    vi.stubGlobal('window', { __AMV_API_PORT__: 9123, amvBridge: { apiToken: 'abc' } });
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ready: true }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await api.status();

    expect(fetchMock).toHaveBeenCalledWith(
      'http://127.0.0.1:9123/api/status',
      expect.objectContaining({ headers: expect.objectContaining({ 'X-AMV-Token': 'abc' }) }),
    );
    expect(api.thumbnailUrl(7)).toBe('http://127.0.0.1:9123/api/scene/7/thumbnail?token=abc');
    expect(withToken('http://127.0.0.1:9123/api/videos/1/stream?variant=enc')).toBe(
      'http://127.0.0.1:9123/api/videos/1/stream?variant=enc&token=abc',
    );
  });

  it('leaves URLs and headers untouched outside Electron', () => {
    vi.stubGlobal('window', { __AMV_API_PORT__: 9123 });
    expect(api.thumbnailUrl(7)).toBe('http://127.0.0.1:9123/api/scene/7/thumbnail');
    expect(authHeaders()).toEqual({});
  });
});
