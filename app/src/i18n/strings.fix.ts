import type { Dict } from './index';

// Strings added by the v06 bug-fix pass: resumable indexing.
export const fixStrings: Dict = {
  'fix.stage.resume': {
    en: 'Resuming where it stopped ({a} scenes already tagged, {b} analysed)',
    fr: 'Reprise là où elle s\'était arrêtée ({a} scènes déjà taguées, {b} analysées)',
  },
  'fix.library.incomplete': {
    en: '{n} episode(s) not fully indexed (indexing was interrupted or failed). They are not searchable yet; the next indexing picks up where it stopped.',
    fr: '{n} épisode(s) incomplet(s) : l\'indexation a été interrompue ou a échoué. Ils ne sont pas encore cherchables ; la prochaine indexation reprend là où elle s\'est arrêtée.',
  },
  'fix.library.busy.indexing': {
    en: 'An indexing run is still finishing (hover previews): wait for it to end.',
    fr: 'Une indexation se termine encore (aperçus au survol) : attends qu’elle finisse.',
  },
  'fix.library.busy.reembed': {
    en: 'The library is being re-analysed for the new model: indexing will be possible when it ends.',
    fr: 'La bibliothèque est en cours de ré-analyse pour le nouveau modèle : l’indexation sera possible ensuite.',
  },
  'fix.library.busy.merge': {
    en: 'A library merge is running: indexing will be possible when it ends.',
    fr: 'Une fusion de bibliothèques est en cours : l’indexation sera possible ensuite.',
  },
  'fix.settings.importFailed': {
    en: 'Could not import these settings: {reason}',
    fr: 'Impossible d’importer ces réglages : {reason}',
  },
  'fix.stage.download': { en: 'Downloading a model', fr: 'Téléchargement d’un modèle' },
  'fix.music.barsHint': {
    en: 'Bar lines (“1”) are estimated. For more reliable ones, download “Beat This!” in Settings > Models (8.5 MB).',
    fr: 'Les premiers temps (« 1 ») sont estimés. Pour plus de fiabilité, télécharge « Beat This! » dans Paramètres > Modèles (8,5 Mo).',
  },
  'fix.library.resume': { en: 'Resume indexing', fr: 'Reprendre l\'indexation' },
  'fix.library.incompleteChip': { en: 'incomplete', fr: 'incomplet' },
  'fix.library.incompleteChip.title': {
    en: 'Indexing did not finish: resume it from the Library',
    fr: 'Indexation non terminée : reprenez-la depuis la Bibliothèque',
  },
};
