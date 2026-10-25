// Cross-tab search+ actions. Any component (Tags, Derush, MiniEditor…) can
// ask for "similar shots" or "search this character" without prop drilling:
// App listens, switches to the Search tab and hands the request to SearchTab.
import type { ColorBucket, MotionType, SearchFilters } from './api/types';

export type SearchPlusRequest =
  | { kind: 'similar'; sceneId: number }
  | { kind: 'character'; characterId: number };

export const SEARCHPLUS_EVENT = 'amv:searchplus';

export function requestSimilar(sceneId: number) {
  window.dispatchEvent(new CustomEvent<SearchPlusRequest>(SEARCHPLUS_EVENT, { detail: { kind: 'similar', sceneId } }));
}

export function requestCharacterSearch(characterId: number) {
  window.dispatchEvent(new CustomEvent<SearchPlusRequest>(SEARCHPLUS_EVENT, { detail: { kind: 'character', characterId } }));
}

export const MOTION_TYPES: MotionType[] = [
  'static', 'pan_left', 'pan_right', 'tilt_up', 'tilt_down', 'zoom_in', 'zoom_out', 'shake',
];

// Swatch colour per bucket (UI only — the backend buckets by HSV).
export const COLOR_SWATCHES: Record<ColorBucket, string> = {
  red: '#EF4444', orange: '#F97316', yellow: '#EAB308', green: '#22C55E', cyan: '#06B6D4',
  blue: '#3B82F6', purple: '#8B5CF6', pink: '#EC4899', brown: '#92400E', white: '#F4F4F5',
  gray: '#71717A', black: '#09090B',
};

export function filtersActive(f: SearchFilters | null | undefined): boolean {
  if (!f) return false;
  return Boolean(
    f.characters?.length || f.motion?.length || f.colors?.length || f.groups?.length || f.favorites
    || f.video_ids?.length || f.exclude_video_ids?.length
    || f.motion_min != null || f.intensity_min != null || f.intensity_max != null
    || f.duration_min_ms != null || f.duration_max_ms != null,
  );
}

export function countActiveFilters(f: SearchFilters): number {
  let n = 0;
  if (f.characters?.length) n++;
  if (f.motion?.length) n++;
  if (f.colors?.length) n++;
  if (f.groups?.length) n++;
  if (f.favorites) n++;
  if (f.intensity_min != null || f.intensity_max != null) n++;
  if (f.duration_min_ms != null || f.duration_max_ms != null) n++;
  return n;
}
