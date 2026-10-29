import type { ProgressEvent } from './api/types';
import type { TFunc } from './i18n';

/** End-of-export line of the derush basket. The sidecar's summary ("2
 *  exported, 1 failed") is English: rebuild it from the counts it sends;
 *  older messages pass through. */
export function exportDoneLabel(e: ProgressEvent & { exported_count?: number; failed_count?: number }, t: TFunc): string {
  if (typeof e.exported_count === 'number') {
    const done = t('journey.derush.exported', { n: e.exported_count });
    return e.failed_count ? `${done} · ${t('journey.derush.exportFailed', { n: e.failed_count })}` : done;
  }
  return e.message ?? t('derush.library.exportFinished');
}
