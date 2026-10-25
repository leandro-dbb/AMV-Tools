import type { Dict } from './index';

// Strings for the Finishing toolbox (FinishMenu / FinishDialog / FinishQueue /
// FinishModelsSection) and the Mask Mode correction tools added in 0.4.
export const finishStrings: Dict = {
  // menu
  'finish.menu': { en: 'Finishing', fr: 'Finition' },
  'finish.menuTitle': {
    en: 'Upscale, slow-motion / 60 fps, depth map or cutout of this clip',
    fr: 'Upscale, ralenti / 60 fps, carte de profondeur ou détourage de ce clip',
  },
  'finish.menuTitleMany': {
    en: 'Apply a finishing pass to the {n} clips shown',
    fr: 'Appliquer une finition aux {n} clips affichés',
  },
  'finish.kind.upscale': { en: 'Upscale', fr: 'Upscale' },
  'finish.kind.upscale.desc': {
    en: 'Real-ESRGAN anime x2 / x4, audio kept',
    fr: 'Real-ESRGAN anime x2 / x4, audio conservé',
  },
  'finish.kind.interpolate': { en: 'Slow-mo / 60 fps', fr: 'Ralenti / 60 fps' },
  'finish.kind.interpolate.desc': {
    en: 'RIFE 4.25 frame interpolation',
    fr: 'Interpolation d\'images RIFE 4.25',
  },
  'finish.kind.depth': { en: 'Depth map', fr: 'Profondeur' },
  'finish.kind.depth.desc': {
    en: 'Depth Anything V2 — for parallax / 3D in After Effects',
    fr: 'Depth Anything V2 — pour parallaxe / 3D dans After Effects',
  },
  'finish.kind.matte': { en: 'Cutout', fr: 'Détourage' },
  'finish.kind.matte.desc': {
    en: 'Automatic alpha matte (ProRes 4444)',
    fr: 'Masque alpha automatique (ProRes 4444)',
  },
  'finish.kind.matte.disabled': {
    en: 'Enable Mask Mode in Settings > Models first',
    fr: 'Activez d\'abord le Mode masque dans Réglages > Modèles',
  },

  // dialog
  'finish.dialog.title': { en: 'Finishing — {kind}', fr: 'Finition — {kind}' },
  'finish.dialog.clips': { en: '{n} clip(s)', fr: '{n} clip(s)' },
  'finish.dialog.clipOne': { en: '1 clip · {seconds} s', fr: '1 clip · {seconds} s' },
  'finish.dialog.start': { en: 'Add to queue', fr: 'Ajouter à la file' },
  'finish.dialog.starting': { en: 'Queuing…', fr: 'Mise en file…' },
  'finish.dialog.queued': {
    en: '{n} job(s) queued — follow them in the Finishing queue (bottom right).',
    fr: '{n} job(s) en file — suivez-les dans la file de finition (en bas à droite).',
  },
  'finish.dialog.output': {
    en: 'Output: export folder / finish, with the codec from Settings > Export.',
    fr: 'Sortie : dossier d\'export / finish, avec le codec de Réglages > Export.',
  },

  // params
  'finish.param.model': { en: 'Model', fr: 'Modèle' },
  'finish.param.model.fast': { en: 'AnimeVideo v3 — fast, steady on video', fr: 'AnimeVideo v3 — rapide, stable en vidéo' },
  'finish.param.model.quality': { en: 'x4plus anime 6B — sharper, slower', fr: 'x4plus anime 6B — plus net, plus lent' },
  'finish.param.scale': { en: 'Scale', fr: 'Facteur' },
  'finish.param.tile': { en: 'Tiles (VRAM)', fr: 'Tuiles (VRAM)' },
  'finish.param.tile.auto': { en: 'Auto', fr: 'Auto' },
  'finish.param.tile.hint': {
    en: 'Smaller tiles = less memory, slightly slower. Auto picks from your GPU.',
    fr: 'Tuiles plus petites = moins de mémoire, un peu plus lent. Auto choisit selon le GPU.',
  },
  'finish.param.mode': { en: 'Mode', fr: 'Mode' },
  'finish.param.mode.slowmo': { en: 'Slow motion', fr: 'Ralenti' },
  'finish.param.mode.fps': { en: 'Higher frame rate', fr: 'Fréquence plus haute' },
  'finish.param.factor': { en: 'Slow-down', fr: 'Ralentissement' },
  'finish.param.targetFps': { en: 'Target fps', fr: 'Fps cible' },
  'finish.param.skipCuts': { en: 'Don\'t interpolate across cuts', fr: 'Ne pas interpoler à travers les coupes' },
  'finish.param.skipCuts.hint': {
    en: 'At a hard cut, frames are duplicated instead of morphed.',
    fr: 'Sur une coupe franche, les images sont dupliquées au lieu d\'être fondues.',
  },
  'finish.param.audio': { en: 'Audio', fr: 'Audio' },
  'finish.param.audio.mute': { en: 'Muted', fr: 'Coupé' },
  'finish.param.audio.stretch': { en: 'Stretched (pitch kept)', fr: 'Étiré (hauteur conservée)' },
  'finish.param.format': { en: 'Format', fr: 'Format' },
  'finish.param.format.png16': { en: '16-bit PNG sequence (best for AE)', fr: 'Séquence PNG 16 bits (idéal AE)' },
  'finish.param.format.prores': { en: 'ProRes 422 HQ (10-bit)', fr: 'ProRes 422 HQ (10 bits)' },
  'finish.param.format.h264': { en: 'H.264 8-bit (preview)', fr: 'H.264 8 bits (aperçu)' },
  'finish.param.smoothing': { en: 'Temporal smoothing', fr: 'Lissage temporel' },
  'finish.param.invert': { en: 'Black = near', fr: 'Noir = proche' },
  'finish.param.codec': { en: 'Codec', fr: 'Codec' },
  'finish.param.stabilize': { en: 'Anti-flicker', fr: 'Anti-scintillement' },

  // estimate / warnings
  'finish.estimate.run': { en: 'Estimate time', fr: 'Estimer la durée' },
  'finish.estimate.running': { en: 'Benchmarking 5 frames…', fr: 'Mesure sur 5 images…' },
  'finish.estimate.result': {
    en: '≈ {total} for this clip ({perFrame} s/frame on {device})',
    fr: '≈ {total} pour ce clip ({perFrame} s/image sur {device})',
  },
  'finish.estimate.resultMany': {
    en: '≈ {total} per clip like this one ({perFrame} s/frame on {device})',
    fr: '≈ {total} par clip similaire ({perFrame} s/image sur {device})',
  },
  'finish.cpuWarning': {
    en: 'No GPU detected: this runs on the CPU and can take several minutes per second of video.',
    fr: 'Aucun GPU détecté : le traitement tourne sur le CPU et peut prendre plusieurs minutes par seconde de vidéo.',
  },
  'finish.noTorch': {
    en: 'The AI runtime (PyTorch) is not installed. Install a backend from Settings first.',
    fr: 'Le moteur IA (PyTorch) n\'est pas installé. Installez un backend depuis les Réglages.',
  },
  'finish.modelMissing': {
    en: '{model} is not downloaded yet ({size}).',
    fr: '{model} n\'est pas encore téléchargé ({size}).',
  },
  'finish.download': { en: 'Download', fr: 'Télécharger' },
  'finish.downloading': { en: 'Downloading… {pct}%', fr: 'Téléchargement… {pct} %' },

  // queue
  'finish.queue.title': { en: 'Finishing queue', fr: 'File de finition' },
  'finish.queue.badge': { en: 'Finishing {n}', fr: 'Finition {n}' },
  'finish.queue.empty': { en: 'No finishing jobs yet.', fr: 'Aucun job de finition.' },
  'finish.queue.clear': { en: 'Clear finished', fr: 'Effacer les terminés' },
  'finish.queue.cancel': { en: 'Cancel', fr: 'Annuler' },
  'finish.queue.open': { en: 'Show file', fr: 'Ouvrir le fichier' },
  'finish.queue.remove': { en: 'Remove from list', fr: 'Retirer de la liste' },
  'finish.queue.eta': { en: '{eta} left', fr: '{eta} restantes' },
  'finish.status.queued': { en: 'Queued', fr: 'En attente' },
  'finish.status.running': { en: 'Running', fr: 'En cours' },
  'finish.status.done': { en: 'Done', fr: 'Terminé' },
  'finish.status.failed': { en: 'Failed', fr: 'Échec' },
  'finish.status.cancelled': { en: 'Cancelled', fr: 'Annulé' },

  // settings > models
  'finish.models.title': { en: 'Finishing models', fr: 'Modèles de finition' },
  'finish.models.hint': {
    en: 'Downloaded on demand (throttled to 10 MB/s) into {root}. Nothing is fetched until you click Download.',
    fr: 'Téléchargés à la demande (bridé à 10 Mo/s) dans {root}. Rien n\'est récupéré avant de cliquer sur Télécharger.',
  },
  'finish.models.installed': { en: 'Installed · {size}', fr: 'Installé · {size}' },
  'finish.models.notInstalled': { en: 'Not installed · {size}', fr: 'Non installé · {size}' },
  'finish.models.delete': { en: 'Delete', fr: 'Supprimer' },
  'finish.models.license': { en: 'License: {license}', fr: 'Licence : {license}' },
  'finish.models.nonCommercial': { en: 'non-commercial use only', fr: 'usage non commercial uniquement' },
  'finish.models.error': { en: 'Download failed: {message}', fr: 'Échec du téléchargement : {message}' },

  // Mask Mode corrections
  'finish.mask.keyframe': { en: 'Frame', fr: 'Image' },
  'finish.mask.keyframeHint': {
    en: 'Scrub to any frame and click there: each frame you click becomes a keyframe.',
    fr: 'Placez-vous sur n\'importe quelle image et cliquez : chaque image cliquée devient une keyframe.',
  },
  'finish.mask.keyframes': { en: 'Keyframes: {list}', fr: 'Keyframes : {list}' },
  'finish.mask.corrected': { en: 'Corrected: {list}', fr: 'Corrigées : {list}' },
  'finish.mask.tool.points': { en: 'Points', fr: 'Points' },
  'finish.mask.tool.add': { en: 'Brush +', fr: 'Pinceau +' },
  'finish.mask.tool.remove': { en: 'Brush −', fr: 'Pinceau −' },
  'finish.mask.tool.addTitle': { en: 'Paint to add to the matte on this frame', fr: 'Peindre pour ajouter au masque sur cette image' },
  'finish.mask.tool.removeTitle': { en: 'Paint to remove from the matte on this frame', fr: 'Peindre pour retirer du masque sur cette image' },
  'finish.mask.brushSize': { en: 'Size', fr: 'Taille' },
  'finish.mask.resetFrame': { en: 'Undo correction', fr: 'Annuler la correction' },
  'finish.mask.repropagate': { en: 'Re-propagate', fr: 'Re-propager' },
  'finish.mask.repropagateTitle': {
    en: 'Track again using every keyframe and brush correction',
    fr: 'Relancer le suivi avec toutes les keyframes et corrections au pinceau',
  },
  'finish.mask.view.overlay': { en: 'Overlay', fr: 'Superposition' },
  'finish.mask.view.checker': { en: 'Checkerboard', fr: 'Damier' },
  'finish.mask.view.color': { en: 'Color', fr: 'Couleur' },
  'finish.mask.viewTitle': { en: 'Preview background', fr: 'Fond de l\'aperçu' },
  'finish.mask.brushNeedsMask': {
    en: 'Generate a mask first, then correct it with the brush.',
    fr: 'Générez d\'abord un masque, puis corrigez-le au pinceau.',
  },
};
