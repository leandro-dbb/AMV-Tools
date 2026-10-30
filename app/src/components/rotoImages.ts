// Image cache for the roto editor viewer. Frames and mattes are fetched with
// the sidecar auth header (fetch + createImageBitmap), so the canvas is never
// tainted and mattes can be turned into alpha masks client-side.
import { authHeaders } from '../api/client';
import { lumToAlpha } from './rotoUtils';

export type RotoImage = ImageBitmap | HTMLCanvasElement;

interface Entry<V> { promise: Promise<V>; value?: V; settled: boolean }

/** Small LRU of decoded images keyed by URL (the URL carries the version). */
class Lru<V> {
  private map = new Map<string, Entry<V>>();
  constructor(private max: number) {}
  get(key: string, load: () => Promise<V>): Promise<V> {
    const hit = this.map.get(key);
    if (hit) {
      this.map.delete(key);
      this.map.set(key, hit);
      return hit.promise;
    }
    const entry: Entry<V> = { promise: load(), settled: false };
    entry.promise.then(
      (v) => { entry.value = v; entry.settled = true; },
      () => { this.map.delete(key); },
    );
    this.map.set(key, entry);
    while (this.map.size > this.max) {
      const oldest = this.map.keys().next().value as string;
      this.map.delete(oldest);
    }
    return entry.promise;
  }
  /** Decoded value without waiting: undefined when not loaded (yet). */
  peek(key: string): V | undefined {
    const e = this.map.get(key);
    return e && e.settled ? e.value : undefined;
  }
  clear() { this.map.clear(); }
}

async function fetchBitmap(url: string): Promise<ImageBitmap | null> {
  const res = await fetch(url, { headers: authHeaders() });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const blob = await res.blob();
  return createImageBitmap(blob);
}

/** Grayscale matte -> white canvas whose alpha is the matte. */
function toAlphaCanvas(bmp: ImageBitmap): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = bmp.width;
  c.height = bmp.height;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  const img = ctx.getImageData(0, 0, c.width, c.height);
  lumToAlpha(img.data);
  ctx.putImageData(img, 0, 0);
  bmp.close();
  return c;
}

export class RotoImageCache {
  private frames = new Lru<ImageBitmap | null>(60);
  private mattes = new Lru<HTMLCanvasElement | null>(90);

  frame(url: string): Promise<ImageBitmap | null> {
    return this.frames.get(url, () => fetchBitmap(url));
  }

  /** Alpha mask of a matte, or null when the server has none (404). */
  matte(url: string): Promise<HTMLCanvasElement | null> {
    return this.mattes.get(url, () => fetchBitmap(url).then((b) => (b ? toAlphaCanvas(b) : null)));
  }

  peekFrame(url: string) { return this.frames.peek(url); }
  peekMatte(url: string) { return this.mattes.peek(url); }

  clear() {
    this.frames.clear();
    this.mattes.clear();
  }
}
