// Derush player key bindings — shared between the player (DerushTab) and the
// remapping UI (Settings > Derush). Values are KeyboardEvent.key, lowercased.

export const DERUSH_DEFAULT_KEYS: Record<string, string> = {
  level_up: 'arrowup',
  level_down: 'arrowdown',
  prev_scene: 'arrowleft',
  next_scene: 'arrowright',
  shuttle_slower: 'j',
  pause: 'k',
  shuttle_faster: 'l',
  play_pause: ' ',
  mute: 'm',
  merge: 'g',
  frame_back: ',',
  frame_forward: '.',
  prev_episode: 'p',
  next_episode: 'n',
  toggle_keep: 'h',
  toggle_favorite: 'ù',
};

// Labels and hints live in the i18n dictionary under
// `derush.action.<id>.label` / `derush.action.<id>.hint`.
export const DERUSH_ACTIONS: { id: keyof typeof DERUSH_DEFAULT_KEYS & string }[] = [
  { id: 'level_up' },
  { id: 'level_down' },
  { id: 'prev_scene' },
  { id: 'next_scene' },
  { id: 'shuttle_slower' },
  { id: 'pause' },
  { id: 'shuttle_faster' },
  { id: 'play_pause' },
  { id: 'mute' },
  { id: 'merge' },
  { id: 'frame_back' },
  { id: 'frame_forward' },
  { id: 'prev_episode' },
  { id: 'next_episode' },
  { id: 'toggle_keep' },
  { id: 'toggle_favorite' },
];

const KEY_LABELS: Record<string, string> = {
  ' ': 'Space',
  arrowup: '↑',
  arrowdown: '↓',
  arrowleft: '←',
  arrowright: '→',
  escape: 'Esc',
  enter: 'Enter',
};

export function displayKey(k: string): string {
  return KEY_LABELS[k] ?? (k.length === 1 ? k.toUpperCase() : k);
}
