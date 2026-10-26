import type { Dict } from './index';

// Strings added while driving the UI end to end (chantier v06 B): indexing
// progress, failure reporting.
export const journeyStrings: Dict = {
  'journey.stage.probe': { en: 'Reading the file', fr: 'Lecture du fichier' },
  'journey.stage.cuts': { en: 'Detecting cuts', fr: 'Détection des coupes' },
  'journey.stage.subsegDecode': { en: 'Splitting long scenes (reading {a} frames)', fr: 'Découpage des longues scènes (lecture de {a} images)' },
  'journey.stage.subseg': { en: 'Splitting long scenes {a}/{b}', fr: 'Découpage des longues scènes {a}/{b}' },
  'journey.stage.frames': { en: 'Extracting {a} key frames', fr: 'Extraction de {a} images clés' },
  'journey.stage.insert': { en: 'Saving {a} scenes and thumbnails', fr: 'Enregistrement de {a} scènes et miniatures' },
  'journey.stage.tagStart': { en: 'Tagging {a} scenes', fr: 'Ajout des tags ({a} scènes)' },
  'journey.stage.tag': { en: 'Tagging {a}/{b}', fr: 'Tags {a}/{b}' },
  'journey.stage.embedStart': { en: 'Building the search index ({a} scenes)', fr: "Construction de l'index de recherche ({a} scènes)" },
  'journey.stage.embed': { en: 'Search index {a}/{b}', fr: 'Index de recherche {a}/{b}' },
  'journey.stage.skipped': { en: 'Already indexed, skipped', fr: 'Déjà indexé, ignoré' },
  'journey.stage.done': { en: 'Done', fr: 'Terminé' },
  'journey.stage.failed': { en: 'Failed: {reason}', fr: 'Échec : {reason}' },
  'journey.index.file': { en: 'file {i}/{n}', fr: 'fichier {i}/{n}' },
  'journey.index.eta': { en: '~{eta} left', fr: '~{eta} restantes' },
  'journey.index.stop': { en: 'Stop', fr: 'Arrêter' },
  'journey.library.running': { en: 'Indexing in progress', fr: 'Indexation en cours' },
  'journey.library.runningHint': {
    en: 'You can keep using the app. If you close it, click "Start indexing" next time: finished episodes are skipped.',
    fr: "Tu peux continuer à utiliser l'app. Si tu la fermes, relance « Lancer l'indexation » au prochain démarrage : les épisodes déjà terminés sont ignorés.",
  },
  'journey.library.failedTitle': { en: '{n} file(s) could not be indexed', fr: "{n} fichier(s) n'ont pas pu être indexés" },
  'journey.library.failedHint': {
    en: 'Check that the file plays in a video player and is not still being copied, then start the indexing again.',
    fr: "Vérifie que le fichier se lit dans un lecteur vidéo et n'est pas en cours de copie, puis relance l'indexation.",
  },
  'journey.onboarding.cpuFast': {
    en: 'Without a GPU, "Fast" is recommended: "Accurate" can take several times longer per episode.',
    fr: "Sans GPU, « Rapide » est recommandé : « Précis » peut prendre plusieurs fois plus de temps par épisode.",
  },
  'journey.settings.folderNotWritable': {
    en: 'AMV Tools cannot write to "{path}". Pick another export folder (a folder of yours, not a read-only or system one).',
    fr: "AMV Tools ne peut pas écrire dans « {path} ». Choisis un autre dossier d'export (un dossier à toi, pas un dossier en lecture seule ou système).",
  },
  'journey.fallback.noTorch': { en: 'No GPU runtime is installed.', fr: "Aucun moteur GPU n'est installé." },
  'journey.fallback.noGpu': { en: 'No supported graphics card was detected.', fr: "Aucune carte graphique compatible n'a été détectée." },
  'journey.fallback.cudaBuild': {
    en: 'NVIDIA (CUDA) was chosen, but the installed engine has no CUDA support.',
    fr: "NVIDIA (CUDA) a été choisi, mais le moteur installé n'a pas le support CUDA.",
  },
  'journey.fallback.cudaDevice': {
    en: 'NVIDIA (CUDA) was chosen, but no NVIDIA card is available (driver?).',
    fr: "NVIDIA (CUDA) a été choisi, mais aucune carte NVIDIA n'est disponible (pilote ?).",
  },
  'journey.fallback.unavailable': { en: '{label} was chosen, but it is unavailable.', fr: '{label} a été choisi, mais est indisponible.' },
  'journey.library.modelsDownload': {
    en: 'First indexing: AMV Tools first downloads its analysis models ({models}, about 2 GB, once). The bar stays at the start while they download.',
    fr: "Première indexation : AMV Tools télécharge d'abord ses modèles d'analyse ({models}, environ 2 Go, une seule fois). La barre reste au début pendant ce téléchargement.",
  },
  'journey.derush.exported': { en: 'Export finished: {n} clip(s)', fr: 'Export terminé : {n} clip(s)' },
  'journey.derush.exportFailed': { en: '{n} failed', fr: '{n} en échec' },
  'journey.library.removeQueued': { en: 'Remove from the queue', fr: 'Retirer de la file' },
};
