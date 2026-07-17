import type { Dict } from './index';

// Strings for LibraryTab.tsx (top-level import/index/organise tab).
// The episode-folders panel and database rows reuse the settings.folders.* /
// settings.db.* keys, which live in strings.settings.ts.
export const libraryStrings: Dict = {
  'library.title': { en: 'Library', fr: 'Bibliothèque' },
  'library.subtitle': { en: 'Add your episodes, index them, organise them — everything starts here.', fr: 'Ajoute tes épisodes, indexe-les, organise-les — tout commence ici.' },

  // Import card
  'library.import.title': { en: 'Import episodes', fr: 'Importer des épisodes' },
  'library.drop.big': { en: 'Drag & drop your episode folders here', fr: 'Glisse tes dossiers d\'épisodes ici' },
  'library.drop.or': { en: 'or', fr: 'ou' },
  'library.group.label': { en: 'Group (optional)', fr: 'Groupe (optionnel)' },
  'library.queue.title': { en: 'Ready to index', fr: 'Prêt à indexer' },
  'library.index.start': { en: 'Start indexing ({n})', fr: 'Lancer l\'indexation ({n})' },
  'library.index.explain': { en: 'Detects cuts, tags every scene and builds the semantic search index. You can keep using the app while it runs.', fr: 'Détecte les cuts, tague chaque scène et construit l\'index de recherche sémantique. Tu peux continuer à utiliser l\'app pendant que ça tourne.' },
  'library.index.advanced': { en: 'Advanced: ', fr: 'Avancé : ' },

  // First-run empty state
  'library.empty.title': { en: 'Your library is empty', fr: 'Ta bibliothèque est vide' },
  'library.empty.body': { en: 'Add a folder of anime episodes above to unlock search, tags and derush.', fr: 'Ajoute un dossier d\'épisodes d\'anime ci-dessus pour débloquer la recherche, les tags et le dérushage.' },

  // Episodes section
  'library.videos.title': { en: 'Indexed episodes', fr: 'Épisodes indexés' },
  'library.videos.count': { en: '{videos} episodes · {scenes} scenes', fr: '{videos} épisodes · {scenes} scènes' },

  // Databases (advanced)
  'library.db.title': { en: 'Databases (advanced)', fr: 'Bases de données (avancé)' },
  'library.db.hint': { en: 'One database = one independent library (e.g. one per project). Import a database coming from another PC here — missing episodes can be relinked in one click.', fr: 'Une base = une bibliothèque indépendante (ex. une par projet). C\'est ici qu\'on importe une base venant d\'un autre PC — les épisodes manquants se relient en un clic.' },
};
