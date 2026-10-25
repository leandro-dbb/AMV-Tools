import type { Dict } from './index';

// Strings for the automatic tuning panel (Settings > Diagnostic).
export const autotuneStrings: Dict = {
  'autotune.title': { en: 'Automatic tuning', fr: 'Réglage automatique' },
  'autotune.hint': {
    en: 'Measured once on this machine (and again when the GPU, driver, PyTorch, FFmpeg or app version changes), in the background while nothing is indexing. Unmeasured settings keep their defaults.',
    fr: 'Mesuré une fois sur cette machine (et à nouveau quand le GPU, le pilote, PyTorch, FFmpeg ou la version de l\'app change), en tâche de fond quand rien n\'est en cours d\'indexation. Les réglages non mesurés gardent leur valeur par défaut.',
  },
  'autotune.rerun': { en: 'Run the automatic tuning again', fr: 'Relancer le réglage automatique' },
  'autotune.state.idle': { en: 'Not measured yet', fr: 'Pas encore mesuré' },
  'autotune.state.waiting': { en: 'Waiting for indexing to finish…', fr: 'En attente de la fin de l\'indexation…' },
  'autotune.state.running': { en: 'Measuring…', fr: 'Mesure en cours…' },
  'autotune.state.runningStep': { en: 'Measuring: {step}…', fr: 'Mesure en cours : {step}…' },
  'autotune.state.done': { en: 'Measured', fr: 'Mesuré' },
  'autotune.state.failed': { en: 'Tuning failed — defaults in use: {error}', fr: 'Échec du réglage — valeurs par défaut utilisées : {error}' },
  'autotune.state.postponed': { en: 'Interrupted by indexing — will measure again later', fr: 'Interrompu par une indexation — nouvelle mesure plus tard' },
  'autotune.state.disabled': { en: 'Disabled (AMV_AUTOTUNE=0)', fr: 'Désactivé (AMV_AUTOTUNE=0)' },
  'autotune.none': { en: 'No measurement for this machine: the default settings are in use.', fr: 'Aucune mesure pour cette machine : les réglages par défaut sont utilisés.' },
  'autotune.nothing': { en: 'Nothing to tune on this machine (no GPU decoding).', fr: 'Rien à régler sur cette machine (pas de décodage GPU).' },
  'autotune.measuredAt': {
    en: 'Measured {date} in {seconds} s on a {source} clip',
    fr: 'Mesuré le {date} en {seconds} s sur un extrait {source}',
  },
  'autotune.source.library': { en: 'library', fr: 'de la bibliothèque' },
  'autotune.source.synthetic': { en: 'synthetic', fr: 'synthétique' },
  'autotune.default': { en: 'default {value}', fr: 'défaut {value}' },
  'autotune.failedCandidate': { en: 'failed', fr: 'échec' },
  'autotune.skipped': { en: 'Not measured', fr: 'Non mesuré' },
  'autotune.errorSetting': { en: 'Measurement failed, default kept', fr: 'Mesure en échec, valeur par défaut conservée' },
  'autotune.setting.parallel_decodes': { en: 'Parallel NVDEC decodes', fr: 'Décodages NVDEC parallèles' },
  'autotune.setting.chunk_workers': { en: 'VideoToolbox sessions', fr: 'Sessions VideoToolbox' },
  'autotune.setting.tagger_engine': { en: 'Tagger engine', fr: 'Moteur du tagger' },
  'autotune.setting.batch_size': { en: 'Tagger / SigLIP batch size', fr: 'Taille de lot tagger / SigLIP' },
  'autotune.setting.tensorrt': { en: 'PyTorch vs TensorRT', fr: 'PyTorch ou TensorRT' },
};
