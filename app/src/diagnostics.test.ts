import { describe, expect, it } from 'vitest';
import type { DiagnosticsReport } from './api/types';
import { formatBytes, isGpuFallback, reportMarkdown } from './diagnostics';

const baseReport = (overrides: Partial<DiagnosticsReport> = {}): DiagnosticsReport => ({
  app: { name: 'AMV Tools', version: '0.4.0', frozen: false },
  python: '3.12.0',
  platform: 'Linux',
  torch: { version: '2.11.0', cuda_available: false },
  ffmpeg: { path: '/usr/bin/ffmpeg', version: 'ffmpeg version 7' },
  device: {
    requested: 'cuda',
    active: 'cpu',
    name: 'CPU',
    vram_gb: null,
    gpu_detected: false,
    fallback_reason: 'CUDA was requested, but no CUDA device is available',
    error: null,
  },
  data_dir: { path: '/data', total_gb: 100, free_gb: 25, used_percent: 75 },
  models: [],
  logs: { path: '/data/logs/backend.log', dir: '/data/logs', recent_errors: [] },
  settings: { indexing_device: 'cuda', mask_enabled: false },
  markdown: '## AMV Tools diagnostic report\n',
  ...overrides,
});

describe('diagnostic helpers', () => {
  it('formats byte sizes', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(1536)).toBe('1.5 KB');
    expect(formatBytes(5 * 1024 ** 3)).toBe('5.0 GB');
    expect(formatBytes(null)).toBe('—');
  });

  it('prefers the backend markdown when present', () => {
    expect(reportMarkdown(baseReport())).toContain('## AMV Tools diagnostic report');
  });

  it('builds a fallback markdown report for older sidecars', () => {
    const md = reportMarkdown(baseReport({ markdown: '' }));
    expect(md).toContain('App version: 0.4.0');
    expect(md).toContain('Device: cuda -> cpu');
  });

  it('detects a GPU fallback', () => {
    expect(isGpuFallback(baseReport())).toBe(true);
    expect(isGpuFallback(baseReport({
      device: { ...baseReport().device, active: 'cuda', fallback_reason: null },
    }))).toBe(false);
    expect(isGpuFallback(null)).toBe(false);
  });
});
