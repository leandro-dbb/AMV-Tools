import type { Dict } from './index';

// Strings for MaskMode.tsx (SAM 2 / BiRefNet mask + alpha-export panel).
export const maskStrings: Dict = {
  // toolbar
  'mask.badge': { en: 'Mask mode', fr: 'Mode masque' },
  'mask.engine.auto': { en: 'Auto', fr: 'Auto' },
  'mask.engine.autoTitle': {
    en: 'BiRefNet - automatic, recommended for anime',
    fr: 'BiRefNet - automatique, recommandé pour l\'anime',
  },
  'mask.engine.manual': { en: 'Manual', fr: 'Manuel' },
  'mask.engine.manualTitle': {
    en: 'SAM 2 - click to pick the subject, useful when there are multiple characters',
    fr: 'SAM 2 - cliquez pour choisir le sujet, utile quand il y a plusieurs personnages',
  },

  // status line
  'mask.status.detecting': {
    en: 'Detecting foreground automatically...',
    fr: 'Détection automatique du premier plan...',
  },
  'mask.status.ready': {
    en: 'Ready: {positive} subject, {negative} background{box} - click "Generate mask"',
    fr: 'Prêt : {positive} sujet, {negative} arrière-plan{box} - cliquez sur « Générer le masque »',
  },
  'mask.status.boxPart': { en: ', 1 box', fr: ', 1 boîte' },
  'mask.status.step1': {
    en: 'Step 1 - click on the character you want to isolate',
    fr: 'Étape 1 - cliquez sur le personnage à détourer',
  },
  'mask.status.previewingSam2': {
    en: 'Running SAM 2 on the reference frame…',
    fr: 'Exécution de SAM 2 sur l\'image de référence…',
  },
  'mask.status.previewingBirefnet': {
    en: 'Running BiRefNet on the reference frame…',
    fr: 'Exécution de BiRefNet sur l\'image de référence…',
  },
  'mask.status.reviewSam2': {
    en: 'Step 2 - happy with the mask? Track it across the clip. Or click more to refine.',
    fr: 'Étape 2 - satisfait du masque ? Trackez-le sur tout le clip, ou ajoutez des clics pour affiner.',
  },
  'mask.status.reviewBirefnet': {
    en: 'Step 2 - happy with the mask? Run it across the clip. Or switch to Manual to refine.',
    fr: 'Étape 2 - satisfait du masque ? Appliquez-le sur tout le clip, ou passez en Manuel pour affiner.',
  },
  'mask.status.maskingAll': {
    en: 'Masking every frame — {pct}%',
    fr: 'Masquage de chaque image — {pct}%',
  },
  'mask.status.trackingAll': {
    en: 'Tracking the mask across the clip — {pct}%',
    fr: 'Tracking du masque sur tout le clip — {pct}%',
  },
  'mask.status.step3': {
    en: 'Step 3 — scrub to check every frame, re-run if needed, then "Export with alpha"',
    fr: 'Étape 3 — parcourez les images pour tout vérifier, relancez si besoin, puis « Exporter avec alpha »',
  },

  // toolbar buttons
  'mask.undo': { en: 'Undo', fr: 'Annuler' },
  'mask.undoTitle': { en: 'Remove last prompt', fr: 'Retirer le dernier prompt' },
  'mask.clear': { en: 'Clear', fr: 'Effacer' },
  'mask.clearTitle': { en: 'Clear all prompts (Esc)', fr: 'Effacer tous les prompts (Échap)' },
  'mask.closeTitle': { en: 'Close mask mode', fr: 'Fermer le mode masque' },

  // legend (Manual mode)
  'mask.legend.clickEq': { en: 'Click =', fr: 'Clic =' },
  'mask.legend.subject': { en: 'subject', fr: 'sujet' },
  'mask.legend.keep': { en: '(keep)', fr: '(garder)' },
  'mask.legend.shiftKey': { en: 'Shift', fr: 'Maj' },
  'mask.legend.shiftClickEq': { en: '+click =', fr: '+clic =' },
  'mask.legend.background': { en: 'background', fr: 'arrière-plan' },
  'mask.legend.drop': { en: '(drop)', fr: '(exclure)' },
  'mask.legend.dragEq': { en: 'Drag =', fr: 'Glisser =' },
  'mask.legend.boundingBox': { en: 'bounding box', fr: 'boîte englobante' },
  'mask.legend.escKey': { en: 'Esc', fr: 'Échap' },
  'mask.legend.escClears': { en: 'clears all', fr: 'efface tout' },

  // legend (Auto mode) — BiRefNet / Manual are wrapped in <b> in the JSX
  'mask.auto.body1': {
    en: 'auto-detects the foreground subject. No clicks needed. If you have several characters and want a specific one, switch to',
    fr: 'détecte automatiquement le sujet au premier plan. Aucun clic nécessaire. Si vous avez plusieurs personnages et en voulez un en particulier, passez en',
  },
  'mask.auto.body2': { en: 'above.', fr: 'ci-dessus.' },

  // canvas
  'mask.alt.referenceFrame': { en: 'reference frame', fr: 'image de référence' },
  'mask.alt.maskOverlay': { en: 'mask overlay', fr: 'superposition du masque' },
  'mask.point.subject': { en: 'Subject #{num}', fr: 'Sujet n°{num}' },
  'mask.point.background': { en: 'Background #{num}', fr: 'Arrière-plan n°{num}' },
  'mask.hint.clickCharacter': {
    en: 'Click on the character you want',
    fr: 'Cliquez sur le personnage voulu',
  },
  'mask.loadingFrame': { en: 'Loading reference frame…', fr: 'Chargement de l\'image de référence…' },
  'mask.trackingProgress': { en: 'Tracking… {pct}%', fr: 'Tracking… {pct}%' },

  // footer
  'mask.frame': { en: 'Frame', fr: 'Image' },
  'mask.generate': { en: 'Generate mask', fr: 'Générer le masque' },
  'mask.repreview': { en: 'Re-preview', fr: 'Re-prévisualiser' },
  'mask.repreviewTitle': {
    en: 'Re-run preview on the reference frame',
    fr: 'Relancer l\'aperçu sur l\'image de référence',
  },
  'mask.retrack': { en: 'Re-track from prompts', fr: 'Re-tracker depuis les prompts' },
  'mask.rerunClip': { en: 'Re-run across clip', fr: 'Relancer sur tout le clip' },
  'mask.trackClip': { en: 'Track across clip', fr: 'Tracker sur tout le clip' },
  'mask.runClip': { en: 'Run across clip', fr: 'Appliquer sur tout le clip' },
  'mask.codec.prores': { en: 'ProRes 4444 (.mov)', fr: 'ProRes 4444 (.mov)' },
  'mask.codec.vp9': { en: 'VP9 alpha (.webm)', fr: 'VP9 alpha (.webm)' },
  'mask.exportTitle': {
    en: 'Encode the clip with the SAM 2 mask as the alpha channel',
    fr: 'Encoder le clip avec le masque SAM 2 comme canal alpha',
  },
  'mask.exportAlpha': { en: 'Export with alpha', fr: 'Exporter avec alpha' },

  // errors
  'mask.error.init': { en: 'Init failed: {message}', fr: 'Échec de l\'initialisation : {message}' },
  'mask.error.clickFirst': {
    en: 'Click at least one point on the subject before previewing.',
    fr: 'Cliquez au moins un point sur le sujet avant de prévisualiser.',
  },
  'mask.error.preview': { en: 'Preview failed: {message}', fr: 'Échec de l\'aperçu : {message}' },
  'mask.error.tracking': { en: 'Tracking failed: {message}', fr: 'Échec du tracking : {message}' },
  'mask.error.export': { en: 'Alpha export failed: {message}', fr: 'Échec de l\'export alpha : {message}' },
};
