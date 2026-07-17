import type { Dict } from './index';

// Strings for App.tsx (tab bar, status bar, backend-unreachable screen).
export const appStrings: Dict = {
  'app.unreachable.title': { en: 'Backend unreachable', fr: 'Backend injoignable' },
  'app.unreachable.body': {
    en: "The Python sidecar didn't respond. If you launched the app outside Electron, start the backend manually:",
    fr: "Le sidecar Python n'a pas répondu. Si vous avez lancé l'application en dehors d'Electron, démarrez le backend manuellement :",
  },
  'app.header.subtitle': { en: 'Semantic Scene Browser', fr: 'Navigateur sémantique de scènes' },
  'app.header.noLibrary': { en: 'No library', fr: 'Aucune bibliothèque' },
  'app.tab.library': { en: 'Library', fr: 'Bibliothèque' },
  'app.tab.search': { en: 'Search', fr: 'Recherche' },
  'app.tab.tags': { en: 'Tags', fr: 'Tags' },
  'app.tab.derush': { en: 'Derush', fr: 'Dérushage' },
  'app.tab.settings': { en: 'Settings', fr: 'Paramètres' },
  'app.statusbar.indexing': { en: 'Indexing', fr: 'Indexation' },
  'app.statusbar.stopIndexing': { en: 'Stop indexing', fr: "Arrêter l'indexation" },
  'app.statusbar.generatingProxies': { en: 'Generating proxies', fr: 'Génération des proxys' },
  'app.statusbar.idle': { en: 'Idle', fr: 'Inactif' },
  'app.statusbar.device': { en: 'Device:', fr: 'Périphérique :' },
  'app.statusbar.tagger': { en: 'Tagger:', fr: 'Tagger :' },
  'app.statusbar.indexedScenes': { en: 'Indexed scenes:', fr: 'Scènes indexées :' },
};
