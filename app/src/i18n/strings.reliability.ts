import type { Dict } from './index';

// Strings for reliability surfaces: GPU fallback banner, Settings > Diagnostic,
// model download errors/retry.
export const reliabilityStrings: Dict = {
  // Non-blocking GPU fallback banner
  'app.fallback.banner': {
    en: 'GPU unavailable — running on {active}. {reason} Indexing will be slower.',
    fr: 'GPU indisponible — exécution sur {active}. {reason} L\'indexation sera plus lente.',
  },
  'app.fallback.openSettings': { en: 'Open Settings', fr: 'Ouvrir les paramètres' },

  // Settings sidebar
  'settings.section.diagnostic': { en: 'Diagnostic', fr: 'Diagnostic' },

  // Panel chrome
  'settings.diagnostic.title': { en: 'Runtime diagnostic', fr: 'Diagnostic d\'exécution' },
  'settings.diagnostic.hint': {
    en: 'Copy this report into a bug report when something goes wrong. It contains no video files, only environment details.',
    fr: 'Collez ce rapport dans un rapport de bug en cas de problème. Il ne contient aucune vidéo, uniquement des détails d\'environnement.',
  },
  'settings.diagnostic.loading': { en: 'Collecting diagnostic…', fr: 'Collecte du diagnostic…' },
  'settings.diagnostic.error': { en: 'Could not collect diagnostic: {message}', fr: 'Impossible de collecter le diagnostic : {message}' },
  'settings.diagnostic.copy': { en: 'Copy report', fr: 'Copier le rapport' },
  'settings.diagnostic.copied': { en: 'Copied!', fr: 'Copié !' },
  'settings.diagnostic.copyFailed': { en: 'Clipboard unavailable — select the report text manually.', fr: 'Presse-papiers indisponible — sélectionnez le texte du rapport manuellement.' },
  'settings.diagnostic.openLogs': { en: 'Open logs folder', fr: 'Ouvrir le dossier des logs' },
  'settings.diagnostic.refresh': { en: 'Refresh', fr: 'Actualiser' },
  'settings.diagnostic.logsUnavailable': { en: 'The logs folder is only available in the desktop app.', fr: 'Le dossier des logs n\'est disponible que dans l\'application de bureau.' },

  // Summary rows
  'settings.diagnostic.appVersion': { en: 'App version', fr: 'Version de l\'application' },
  'settings.diagnostic.python': { en: 'Python', fr: 'Python' },
  'settings.diagnostic.platform': { en: 'Platform', fr: 'Plateforme' },
  'settings.diagnostic.torch': { en: 'PyTorch', fr: 'PyTorch' },
  'settings.diagnostic.ffmpeg': { en: 'FFmpeg', fr: 'FFmpeg' },
  'settings.diagnostic.requested': { en: 'Requested acceleration', fr: 'Accélération demandée' },
  'settings.diagnostic.active': { en: 'Active device', fr: 'Périphérique actif' },
  'settings.diagnostic.fallback': { en: 'Fallback', fr: 'Repli' },
  'settings.diagnostic.dataDir': { en: 'Data folder', fr: 'Dossier de données' },
  'settings.diagnostic.diskFree': { en: '{free} GB free of {total} GB ({used}% used)', fr: '{free} Go libres sur {total} Go ({used} % utilisés)' },
  'settings.diagnostic.logs': { en: 'Backend log', fr: 'Log backend' },
  'settings.diagnostic.notInstalled': { en: 'not installed', fr: 'non installé' },
  'settings.diagnostic.unknown': { en: 'unknown', fr: 'inconnu' },

  // Models
  'settings.diagnostic.models': { en: 'Model weights', fr: 'Poids des modèles' },
  'settings.diagnostic.model.present': { en: 'Present', fr: 'Présent' },
  'settings.diagnostic.model.absent': { en: 'Absent', fr: 'Absent' },
  'settings.diagnostic.model.loaded': { en: 'loaded in memory', fr: 'chargé en mémoire' },
  'settings.diagnostic.model.enabled': { en: 'enabled', fr: 'activé' },
  'settings.diagnostic.model.disabled': { en: 'disabled', fr: 'désactivé' },
  'settings.diagnostic.model.retry': { en: 'Retry download', fr: 'Réessayer le téléchargement' },
  'settings.diagnostic.model.retrying': { en: 'Downloading…', fr: 'Téléchargement…' },
  'settings.diagnostic.model.downloaded': { en: 'Download complete.', fr: 'Téléchargement terminé.' },
  'settings.diagnostic.model.error': { en: 'Download failed: {error}', fr: 'Échec du téléchargement : {error}' },

  // Errors
  'settings.diagnostic.recentErrors': { en: 'Recent errors', fr: 'Erreurs récentes' },
  'settings.diagnostic.noErrors': { en: 'No recent warnings or errors.', fr: 'Aucun avertissement ni erreur récente.' },
  'settings.diagnostic.report': { en: 'Report preview', fr: 'Aperçu du rapport' },
};
