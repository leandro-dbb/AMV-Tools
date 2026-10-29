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
  'fix.library.resume': { en: 'Resume indexing', fr: 'Reprendre l\'indexation' },
  'fix.library.incompleteChip': { en: 'incomplete', fr: 'incomplet' },
  'fix.library.incompleteChip.title': {
    en: 'Indexing did not finish: resume it from the Library',
    fr: 'Indexation non terminée : reprenez-la depuis la Bibliothèque',
  },
};
