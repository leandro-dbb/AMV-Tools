// Containers Final Cut Pro cannot import (it offers no MKV/WebM reader): a
// timeline pointing at them opens with every clip offline.
const FINAL_CUT_UNSUPPORTED = /\.(mkv|webm|avi|flv|wmv|ogv|ogm)$/i;

/** The sources of a timeline that Final Cut Pro will not open. */
export function finalCutUnsupported(sources: string[]): string[] {
  return sources.filter((p) => FINAL_CUT_UNSUPPORTED.test(p));
}
