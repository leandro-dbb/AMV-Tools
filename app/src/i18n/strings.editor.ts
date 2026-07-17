import type { Dict } from './index';

// Strings for MiniEditor.tsx.
export const editorStrings: Dict = {
  // Header / breadcrumb
  'editor.query': { en: 'Query', fr: 'Requête' },
  'editor.noTags': { en: '(no tags)', fr: '(aucun tag)' },
  'editor.untagged': { en: 'untagged', fr: 'sans tag' },
  'editor.prevOccurrence': { en: 'Prev occurrence (←)', fr: 'Occurrence précédente (←)' },
  'editor.nextOccurrence': { en: 'Next occurrence (→)', fr: 'Occurrence suivante (→)' },

  // Mask toggle
  'editor.mask': { en: 'Mask', fr: 'Masque' },
  'editor.maskOpenTitle': {
    en: 'Open mask mode (click→mask, alpha export)',
    fr: 'Ouvrir le mode masque (clic→masque, export alpha)',
  },
  'editor.maskCloseTitle': { en: 'Close mask mode', fr: 'Fermer le mode masque' },

  // Export button
  'editor.exportTitle': {
    en: 'Export {seconds}s clip via ffmpeg',
    fr: 'Exporter un clip de {seconds} s via ffmpeg',
  },
  'editor.exporting': { en: 'Exporting…', fr: 'Export en cours…' },
  'editor.exportBtn': { en: 'Export {seconds}s', fr: 'Exporter {seconds} s' },

  // Playback errors
  'editor.playbackFailed': {
    en: 'Playback failed: {message}',
    fr: 'Lecture impossible : {message}',
  },
  'editor.videoErrorCode': { en: 'Code {code}: {message}', fr: 'Code {code} : {message}' },
  'editor.codecUnsupported': {
    en: 'codec unsupported or source unreachable',
    fr: 'codec non pris en charge ou source inaccessible',
  },
  'editor.unknownVideoError': { en: 'unknown video error', fr: 'erreur vidéo inconnue' },
  'editor.playbackError': { en: 'Playback error', fr: 'Erreur de lecture' },
  'editor.playbackTip': {
    en: 'Tip: Settings → Indexing → Generate proxies = ON for unsupported codecs.',
    fr: 'Astuce : Paramètres → Indexation → Générer des proxys = ON pour les codecs non pris en charge.',
  },

  // Timeline / trim
  'editor.detectedBounds': { en: 'Detected scene bounds', fr: 'Limites de scène détectées' },
  'editor.dragStart': {
    en: 'Drag to adjust scene start',
    fr: 'Glisser pour ajuster le début de la scène (point d’entrée)',
  },
  'editor.dragEnd': {
    en: 'Drag to adjust scene end',
    fr: 'Glisser pour ajuster la fin de la scène (point de sortie)',
  },

  // Transport / keyboard hints
  'editor.pauseTitle': { en: 'Pause (Space / K)', fr: 'Pause (Espace / K)' },
  'editor.playTitle': { en: 'Play (Space / K)', fr: 'Lecture (Espace / K)' },
  'editor.kbdPlay': { en: 'play ·', fr: 'lecture ·' },
  'editor.kbdScrub': { en: 'scrub ·', fr: 'défilement ·' },
  'editor.kbdTrim': { en: 'trim', fr: 'découpe' },

  // Export result
  'editor.exportedSuccess': { en: '✓ Exported: {file}', fr: '✓ Exporté : {file}' },
  'editor.exportFailed': { en: 'Export failed', fr: 'Échec de l’export' },
  'editor.exportTipPrefix': {
    en: 'Tip: check Settings → Export → Codec. ',
    fr: 'Astuce : vérifiez Paramètres → Export → Codec. ',
  },
  'editor.exportTipSuffix': {
    en: ' is the most compatible.',
    fr: ' est le plus compatible.',
  },
};
