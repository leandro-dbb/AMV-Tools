import type { Dict } from './index';

// Strings for TagsTab.tsx.
export const tagsStrings: Dict = {
  'tags.noVideos': { en: 'No videos', fr: 'Aucune vidéo' },
  'tags.videosSelected': { en: '{count} videos selected', fr: '{count} vidéos sélectionnées' },
  'tags.noVideosIndexed': { en: 'No videos indexed yet.', fr: 'Aucune vidéo indexée pour le moment.' },
  'tags.sceneCount': { en: '{count} scenes', fr: '{count} scènes' },
  'tags.allScenes': { en: 'All scenes', fr: 'Toutes les scènes' },
  'tags.noTagFilter': { en: '(no tag filter)', fr: '(aucun filtre de tag)' },
  'tags.filterTags': { en: 'Filter tags…', fr: 'Filtrer les tags…' },
  'tags.noTagMatches': { en: 'No tag matches «{query}».', fr: 'Aucun tag ne correspond à « {query} ».' },
  'tags.noTagsInSelection': { en: 'No tags found in this selection.', fr: 'Aucun tag trouvé dans cette sélection.' },
  'tags.threshold': { en: 'Threshold', fr: 'Seuil' },
  'tags.sort.timecode': { en: 'timecode', fr: 'timecode' },
  'tags.sort.confidence': { en: 'confidence', fr: 'confiance' },
  'tags.clear': { en: 'Clear', fr: 'Effacer' },
  'tags.selectAll': { en: 'Select all', fr: 'Tout sélectionner' },
  'tags.exportSelected': { en: 'Export Selected', fr: 'Exporter la sélection' },
  'tags.exportSelectedCount': { en: 'Export Selected ({count})', fr: 'Exporter la sélection ({count})' },
  'tags.loadingScenes': { en: 'Loading scenes...', fr: 'Chargement des scènes…' },
  'tags.exported': { en: 'Exported {count}', fr: '{count} exportées' },
  'tags.exportedFailed': { en: 'Exported {count}, {failed} failed', fr: '{count} exportées, {failed} en échec' },
  'tags.noScenesMatch': {
    en: 'No scenes match tag "{tag}" at threshold {threshold}%.',
    fr: 'Aucune scène ne correspond au tag « {tag} » au seuil de {threshold} %.',
  },
  'tags.noScenesYet': {
    en: 'No scenes in this video selection yet — index it first from the Library tab.',
    fr: 'Aucune scène dans cette sélection de vidéos pour l\'instant — indexez-la d\'abord depuis l\'onglet Bibliothèque.',
  },
  'tags.sceneAlt': { en: 'Scene', fr: 'Scène' },
};
