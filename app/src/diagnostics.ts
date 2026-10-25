import type { DiagnosticsReport } from './api/types';

/** Human-readable byte size used by the diagnostic model list. */
export function formatBytes(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value) || value < 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(size >= 100 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

/** The backend already renders the paste-ready Markdown; keep a fallback so a
 *  stale sidecar can never leave the Copy button empty. */
export function reportMarkdown(report: DiagnosticsReport): string {
  if (report.markdown && report.markdown.trim()) return report.markdown;
  return [
    '## AMV Tools diagnostic report',
    '',
    `- App version: ${report.app.version}`,
    `- Python: ${report.python}`,
    `- Platform: ${report.platform}`,
    `- Torch: ${report.torch.version ?? 'not installed'}`,
    `- FFmpeg: ${report.ffmpeg.version ?? 'not found'}`,
    `- Device: ${report.device.requested} -> ${report.device.active}`,
  ].join('\n');
}

/** True when the user asked for a GPU backend but the app is on CPU. */
export function isGpuFallback(report: Pick<DiagnosticsReport, 'device'> | null | undefined): boolean {
  if (!report) return false;
  return Boolean(report.device.fallback_reason) && report.device.active !== 'cuda' &&
    report.device.active !== 'mps' && report.device.active !== 'dml' && report.device.active !== 'xpu';
}
