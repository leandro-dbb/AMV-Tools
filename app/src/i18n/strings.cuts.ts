import type { Dict } from './index';

// Strings for the cut network (TransNetV2) setting (Settings > Indexing).
export const cutsStrings: Dict = {
  'settings.indexing.preciseCuts.label': { en: 'Precise cuts (slower)', fr: 'Coupes précises (plus lent)' },
  'settings.indexing.preciseCuts.base': {
    en: 'The TransNetV2 network next to the cut detector: about 15 % fewer missed cuts on anime. Needs TransNetV2 (29 MB: Settings > Diagnostic > Model weights).',
    fr: 'Le réseau TransNetV2 en plus du détecteur de coupes : environ 15 % de coupes ratées en moins sur l\'anime. Nécessite TransNetV2 (29 Mo : Paramètres › Diagnostic › Poids des modèles).',
  },
  'settings.indexing.preciseCuts.auto': {
    en: 'On by itself on this machine: it costs {pct} % on the decoding pass (measured).',
    fr: 'Activé d\'office sur cette machine : il coûte {pct} % sur la passe de décodage (mesuré).',
  },
  'settings.indexing.preciseCuts.manual': {
    en: 'On this machine: about +{extra} per 24-min episode (measured).',
    fr: 'Sur cette machine : environ +{extra} par épisode de 24 min (mesuré).',
  },
  'settings.indexing.preciseCuts.unmeasured': {
    en: 'Extra time not measured yet on this machine (Settings > Diagnostic > Automatic tuning).',
    fr: 'Temps supplémentaire pas encore mesuré sur cette machine (Paramètres › Diagnostic › Réglage automatique).',
  },
  'autotune.setting.cutnet': { en: 'Cut network (TransNetV2)', fr: 'Réseau de coupes (TransNetV2)' },
  'autotune.setting.cutnet_device': { en: 'Cut network device', fr: 'Processeur du réseau de coupes' },
};
