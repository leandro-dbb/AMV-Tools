// Indexing progress, as a video editor reads it: what the sidecar is doing
// right now (its English stage messages, translated), and how long is left.
import type { TFunc } from './i18n';

const STAGES: [RegExp, string][] = [
  [/^Probing video/, 'journey.stage.probe'],
  [/^Detecting cuts/, 'journey.stage.cuts'],
  [/^Sub-segmenting \(decoding (\d+) samples/, 'journey.stage.subsegDecode'],
  [/^Sub-segmenting (\d+)\/(\d+)/, 'journey.stage.subseg'],
  [/^Extracting (\d+) median frames/, 'journey.stage.frames'],
  [/^Inserting (\d+) scenes/, 'journey.stage.insert'],
  [/^Tagging (\d+) scenes/, 'journey.stage.tagStart'],
  [/^Tagged (\d+)\/(\d+)/, 'journey.stage.tag'],
  [/^Embedding (\d+) scenes/, 'journey.stage.embedStart'],
  [/^Embedded (\d+)\/(\d+)/, 'journey.stage.embed'],
  [/^Resuming: (\d+) scenes tagged, (\d+) embedded/, 'fix.stage.resume'],
  [/^Already indexed/, 'journey.stage.skipped'],
  [/^Done/, 'journey.stage.done'],
];

/** Localised label of a pipeline stage message; unknown messages pass through. */
export function stageLabel(message: string | undefined, t: TFunc): string {
  if (!message) return '';
  for (const [re, key] of STAGES) {
    const m = re.exec(message);
    if (m) return t(key, { a: m[1] ?? '', b: m[2] ?? '' });
  }
  if (message.startsWith('Failed: ')) return t('journey.stage.failed', { reason: message.slice(8) });
  return message;
}

/** Seconds left for the current file, extrapolated from the progress made
 *  since `fromPercent` (when the bar passed the tagging/embedding stages, the
 *  bulk of the work). null until there is enough signal: during cut
 *  detection the bar sits on 3-5 % and the estimate climbed to "23 min". */
export function etaSeconds(percent: number, elapsedS: number, fromPercent = 0): number | null {
  const done = percent - fromPercent;
  if (percent < 25 || percent >= 100 || elapsedS < 5 || done < 3) return null;
  return (elapsedS * (100 - percent)) / done;
}

export interface IndexFailure { video: string; path?: string; reason: string }

/** Failed files of the current run, from the progress stream. A new run
 *  (indexing after idle) starts a fresh list. */
export function nextFailures(
  prev: IndexFailure[],
  evt: { type: string; failed?: boolean; video?: string; path?: string; message?: string },
  prevType: string | null,
): IndexFailure[] {
  let list = prev;
  if (evt.type === 'indexing' && prevType !== 'indexing' && !evt.failed) list = [];
  if (evt.type === 'indexing' && evt.failed) {
    const reason = (evt.message ?? '').replace(/^Failed: /, '');
    const key = evt.path ?? evt.video;
    list = [...list.filter((f) => (f.path ?? f.video) !== key), { video: evt.video ?? '?', path: evt.path, reason }];
  }
  return list;
}

const FALLBACKS: [RegExp, string][] = [
  [/torch is not installed/, 'journey.fallback.noTorch'],
  [/^No supported GPU detected/, 'journey.fallback.noGpu'],
  [/^CUDA was requested, but this PyTorch build has no CUDA support/, 'journey.fallback.cudaBuild'],
  [/^CUDA was requested, but no CUDA device is available/, 'journey.fallback.cudaDevice'],
  [/^(.+) was requested, but it is unavailable/, 'journey.fallback.unavailable'],
];

/** The sidecar explains a CPU fallback in English: localise the known reasons. */
export function fallbackLabel(reason: string, t: TFunc): string {
  for (const [re, key] of FALLBACKS) {
    const m = re.exec(reason);
    if (m) return t(key, { label: m[1] ?? '' });
  }
  return reason;
}
