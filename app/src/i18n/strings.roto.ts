import type { Dict } from './index';

// Strings for the roto editor (RotoEditor*.tsx, v0.6): the AI does a first
// cut-out of the characters, the editor fixes it with a few gestures.
export const rotoStrings: Dict = {
  // entry point (MiniEditor)
  'roto.open': { en: 'Cut out this shot', fr: 'Détourer ce plan' },
  'roto.openTitle': {
    en: 'Open the cut-out editor: the AI cuts out the characters, you fix what it missed with a few brush strokes',
    fr: 'Ouvrir l\'éditeur de détourage : l\'IA détoure les personnages, vous corrigez ce qu\'elle a raté en quelques coups de pinceau',
  },

  // top bar
  'roto.badge': { en: 'Cut-out', fr: 'Détourage' },
  'roto.opening': { en: 'Opening the shot…', fr: 'Ouverture du plan…' },
  'roto.openError': { en: 'Could not open the shot: {message}', fr: 'Impossible d\'ouvrir le plan : {message}' },
  'roto.retry': { en: 'Retry', fr: 'Réessayer' },
  'roto.close': { en: 'Close', fr: 'Fermer' },
  'roto.closeTitle': {
    en: 'Close the editor (your work is saved, reopen the shot to continue)',
    fr: 'Fermer l\'éditeur (votre travail est enregistré, rouvrez le plan pour continuer)',
  },
  'roto.fullscreen': { en: 'Full screen (F)', fr: 'Plein écran (F)' },
  'roto.exitFullscreen': { en: 'Exit full screen (F)', fr: 'Quitter le plein écran (F)' },
  'roto.help': { en: 'Keyboard shortcuts (?)', fr: 'Raccourcis clavier (?)' },
  'roto.models.characters': { en: 'Characters: {name}', fr: 'Personnages : {name}' },
  'roto.models.tracker': { en: 'Tracking: {name}', fr: 'Suivi : {name}' },
  'roto.models.noTracker': { en: 'Tracking: none (frame by frame)', fr: 'Suivi : aucun (image par image)' },
  'roto.models.clicksTitle': { en: 'Clicks answered by: {name}', fr: 'Clics traités par : {name}' },
  'roto.device.cuda': { en: 'GPU', fr: 'GPU' },
  'roto.device.mps': { en: 'Apple GPU', fr: 'GPU Apple' },
  'roto.device.cpu': { en: 'CPU', fr: 'CPU' },
  'roto.models.sam3Missing': {
    en: 'SAM 3 not available (no Hugging Face access): SAM 2.1 large used instead',
    fr: 'SAM 3 indisponible (accès Hugging Face manquant) : SAM 2.1 large utilisé à la place',
  },
  'roto.models.slow': {
    en: 'This computer has no graphics card the AI can use: each correction can take a few seconds',
    fr: 'Cet ordinateur n\'a pas de carte graphique utilisable par l\'IA : chaque correction peut prendre quelques secondes',
  },
  'roto.saved': { en: 'Saved automatically', fr: 'Enregistré automatiquement' },
  'roto.saving': { en: 'Saving…', fr: 'Enregistrement…' },

  // first AI pass
  'roto.first.title': { en: 'Who do you want to cut out?', fr: 'Qui voulez-vous détourer ?' },
  'roto.first.subtitle': {
    en: 'The AI does a first pass on the whole shot. You then fix what it missed.',
    fr: 'L\'IA fait un premier passage sur tout le plan. Vous corrigez ensuite ce qu\'elle a raté.',
  },
  'roto.target.all': { en: 'All the characters', fr: 'Tous les personnages' },
  'roto.target.main': { en: 'The main character', fr: 'Le personnage principal' },
  'roto.target.click': { en: 'Click on the character', fr: 'Cliquer sur le personnage' },
  'roto.target.allHint': { en: 'Every character in the shot', fr: 'Tous les personnages du plan' },
  'roto.target.mainHint': { en: 'The biggest, most central one', fr: 'Le plus grand, le plus au centre' },
  'roto.target.clickHint': { en: 'You choose in the image', fr: 'Vous choisissez dans l\'image' },
  'roto.detect': { en: 'Cut out', fr: 'Détourer' },
  'roto.detectClickPrompt': {
    en: 'Click on the character in the image',
    fr: 'Cliquez sur le personnage dans l\'image',
  },
  'roto.layerPick.title': { en: 'New character: {name}', fr: 'Nouveau personnage : {name}' },
  'roto.layerPick.hint': {
    en: 'Click on this character in the image (Alt+click or right click on what is not part of it), then press Cut out.',
    fr: 'Cliquez sur ce personnage dans l\'image (Alt+clic ou clic droit sur ce qui n\'en fait pas partie), puis appuyez sur Détourer.',
  },
  'roto.layerPick.points': { en: '{count} point(s) placed', fr: '{count} point(s) placé(s)' },
  'roto.layerPick.cancel': { en: 'Later', fr: 'Plus tard' },
  'roto.layerPick.clear': { en: 'Clear points', fr: 'Effacer les points' },

  // jobs
  'roto.job.detect': { en: 'The AI is cutting out the characters…', fr: 'L\'IA détoure les personnages…' },
  'roto.job.propagate': { en: 'Following the corrections through the shot…', fr: 'Propagation des corrections sur le plan…' },
  'roto.job.export': { en: 'Exporting…', fr: 'Export en cours…' },
  'roto.job.cancel': { en: 'Cancel', fr: 'Annuler' },
  'roto.job.cancelled': { en: 'Cancelled.', fr: 'Annulé.' },
  'roto.job.error': { en: 'Failed: {message}', fr: 'Échec : {message}' },
  'roto.busyJob': { en: 'Wait for the current task to finish', fr: 'Attendez la fin de la tâche en cours' },

  // gestures
  'roto.pending': { en: 'Correcting…', fr: 'Correction en cours…' },
  'roto.pendingQueue': { en: 'Correcting… ({count} waiting)', fr: 'Correction en cours… ({count} en attente)' },
  'roto.latency': { en: 'Frame fixed in {ms} ms (AI {server} ms)', fr: 'Image corrigée en {ms} ms (IA {server} ms)' },
  'roto.latencyTitle': {
    en: 'AI: {server} ms, total with the network round trip: {total} ms',
    fr: 'IA : {server} ms, total avec l\'aller-retour : {total} ms',
  },
  'roto.error': { en: 'Error: {message}', fr: 'Erreur : {message}' },
  'roto.dismiss': { en: 'Dismiss', fr: 'Fermer' },

  // views
  'roto.view.overlay': { en: 'Overlay', fr: 'Calque' },
  'roto.view.overlayTitle': {
    en: 'Character tinted over the original image (1)',
    fr: 'Personnage teinté sur l\'image d\'origine (1)',
  },
  'roto.view.black': { en: 'On black', fr: 'Sur noir' },
  'roto.view.blackTitle': { en: 'Cut-out character on black (2)', fr: 'Personnage détouré sur fond noir (2)' },
  'roto.view.white': { en: 'On white', fr: 'Sur blanc' },
  'roto.view.whiteTitle': { en: 'Cut-out character on white (3)', fr: 'Personnage détouré sur fond blanc (3)' },
  'roto.view.checker': { en: 'Transparent', fr: 'Transparent' },
  'roto.view.checkerTitle': {
    en: 'Cut-out character on a checkerboard = transparency (4)',
    fr: 'Personnage détouré sur damier = transparence (4)',
  },
  'roto.view.original': { en: 'Original', fr: 'Original' },
  'roto.view.originalTitle': { en: 'The original image only (5)', fr: 'L\'image d\'origine seule (5)' },
  'roto.fit': { en: 'Fit', fr: 'Ajuster' },
  'roto.fitTitle': {
    en: 'Fit the image to the window (0). Mouse wheel = zoom, Space+drag or middle button = move',
    fr: 'Ajuster l\'image à la fenêtre (0). Molette = zoom, Espace+glisser ou bouton du milieu = déplacer',
  },
  'roto.soloLayer': { en: 'Only this character', fr: 'Seulement ce personnage' },
  'roto.soloLayerTitle': {
    en: 'Show only the selected character instead of all of them',
    fr: 'N\'afficher que le personnage sélectionné au lieu de tous',
  },

  // tools
  'roto.tools': { en: 'Fix', fr: 'Corriger' },
  'roto.tool.brush': { en: 'Brush', fr: 'Pinceau' },
  'roto.tool.brushTitle': {
    en: 'Brush (B): paint what belongs to the character or to the background. Hold Alt to remove.',
    fr: 'Pinceau (B) : peignez ce qui fait partie du personnage ou du fond. Maintenez Alt pour retirer.',
  },
  'roto.tool.eraser': { en: 'Eraser', fr: 'Gomme' },
  'roto.tool.eraserTitle': {
    en: 'Eraser (E): undo your paint under the stroke and put back what the AI found',
    fr: 'Gomme (E) : annule votre peinture sous le trait et remet ce que l\'IA avait trouvé',
  },
  'roto.tool.click': { en: 'Clicks', fr: 'Clics' },
  'roto.tool.clickTitle': {
    en: 'Clicks (C): click a missing part to add it, Alt+click or right click a wrong part to remove it',
    fr: 'Clics (C) : cliquez sur une partie manquante pour l\'ajouter, Alt+clic ou clic droit sur une partie en trop pour la retirer',
  },
  'roto.brush.add': { en: 'Add', fr: 'Ajouter' },
  'roto.brush.addTitle': { en: 'Paint = part of the character', fr: 'Peindre = fait partie du personnage' },
  'roto.brush.remove': { en: 'Remove', fr: 'Retirer' },
  'roto.brush.removeTitle': { en: 'Paint = background (hold Alt)', fr: 'Peindre = fond (maintenir Alt)' },
  'roto.brush.size': { en: 'Size', fr: 'Taille' },
  'roto.brush.sizeTitle': { en: 'Brush size ([ and ])', fr: 'Taille du pinceau ([ et ])' },
  'roto.click.legend': {
    en: 'Click = add · Alt+click / right click = remove',
    fr: 'Clic = ajouter · Alt+clic / clic droit = retirer',
  },
  'roto.undo': { en: 'Undo (Ctrl+Z)', fr: 'Annuler (Ctrl+Z)' },
  'roto.redo': { en: 'Redo (Ctrl+Shift+Z)', fr: 'Rétablir (Ctrl+Maj+Z)' },
  'roto.undoShort': { en: 'Undo', fr: 'Annuler' },
  'roto.redoShort': { en: 'Redo', fr: 'Rétablir' },

  // propagate / validate
  'roto.propagate': { en: 'Propagate', fr: 'Propager' },
  'roto.propagateTitle': {
    en: 'Starts again from the corrected frames; the frames you corrected or validated are never overwritten. (P = both directions)',
    fr: 'Repart des images corrigées ; les images que vous avez corrigées ou validées ne sont jamais écrasées. (P = dans les deux sens)',
  },
  'roto.propagate.backward': { en: 'Propagate backward', fr: 'Propager vers l\'arrière' },
  'roto.propagate.both': { en: 'Propagate both ways (P)', fr: 'Propager dans les deux sens (P)' },
  'roto.propagate.forward': { en: 'Propagate forward', fr: 'Propager vers l\'avant' },
  'roto.validate': { en: 'Validate this frame', fr: 'Valider cette image' },
  'roto.validated': { en: 'Frame validated', fr: 'Image validée' },
  'roto.validateTitle': {
    en: 'Mark this frame as correct (V): propagation will never change it',
    fr: 'Marquer cette image comme correcte (V) : la propagation ne la modifiera jamais',
  },

  // timeline
  'roto.frameLabel': { en: 'Frame {current} / {total}', fr: 'Image {current} / {total}' },
  'roto.suspectsLeft': { en: '{count} frames to check', fr: '{count} images à vérifier' },
  'roto.suspectsLeftOne': { en: '1 frame to check', fr: '1 image à vérifier' },
  'roto.suspectsNone': { en: 'Nothing to check', fr: 'Rien à vérifier' },
  'roto.nextSuspect': { en: 'Next frame to check (N)', fr: 'Image à vérifier suivante (N)' },
  'roto.prevSuspect': { en: 'Previous frame to check (Shift+N)', fr: 'Image à vérifier précédente (Maj+N)' },
  'roto.play': { en: 'Play (Space)', fr: 'Lecture (Espace)' },
  'roto.pause': { en: 'Pause (Space)', fr: 'Pause (Espace)' },
  'roto.prevFrame': { en: 'Previous frame (←)', fr: 'Image précédente (←)' },
  'roto.nextFrame': { en: 'Next frame (→)', fr: 'Image suivante (→)' },
  'roto.firstFrame': { en: 'First frame (Home)', fr: 'Première image (Début)' },
  'roto.lastFrame': { en: 'Last frame (End)', fr: 'Dernière image (Fin)' },
  'roto.suspectTitle': { en: 'Frame {frame} to check: {reasons}', fr: 'Image {frame} à vérifier : {reasons}' },
  'roto.keyframeTitle': { en: 'Frame {frame}: corrected or validated by you', fr: 'Image {frame} : corrigée ou validée par vous' },
  'roto.reason.lost': { en: 'character lost', fr: 'personnage perdu' },
  'roto.reason.jump': { en: 'sudden jump', fr: 'saut brusque' },
  'roto.reason.specks': { en: 'stray specks', fr: 'taches parasites' },
  'roto.reason.object_lost': { en: 'object lost', fr: 'objet perdu' },
  'roto.reason.hole': { en: 'hole', fr: 'trou' },
  'roto.reason.rough': { en: 'jagged edge', fr: 'bord en escalier' },
  'roto.legend.keyframe': { en: 'fixed by you', fr: 'corrigée par vous' },
  'roto.legend.suspect': { en: 'to check', fr: 'à vérifier' },

  // edges
  'roto.edges': { en: 'Edges', fr: 'Bords' },
  'roto.edges.keepInk': { en: 'Keep the black line', fr: 'Garder le trait noir' },
  'roto.edges.keepInkTitle': {
    en: 'Keep the black outline of the drawing around the character',
    fr: 'Garder le contour noir du dessin autour du personnage',
  },
  'roto.edges.softness': { en: 'Soften ↔ Harden', fr: 'Adoucir ↔ Durcir' },
  'roto.edges.grow': { en: 'Shrink ↔ Grow', fr: 'Rétrécir ↔ Élargir' },
  'roto.edges.temporal': { en: 'Smoothing over time', fr: 'Lissage dans le temps' },
  'roto.edges.temporalTitle': {
    en: 'Reduces flickering edges from one frame to the next',
    fr: 'Réduit le scintillement des bords d\'une image à l\'autre',
  },
  'roto.edges.reset': { en: 'Reset', fr: 'Réinitialiser' },

  // layers
  'roto.layers': { en: 'Characters', fr: 'Personnages' },
  'roto.layers.add': { en: '+ Character', fr: '+ Personnage' },
  'roto.layers.addTitle': {
    en: 'Add another character to cut out separately',
    fr: 'Ajouter un autre personnage à détourer séparément',
  },
  'roto.layers.delete': { en: 'Delete this character', fr: 'Supprimer ce personnage' },
  'roto.layers.deleteConfirm': {
    en: 'Delete "{name}" and its cut-out?',
    fr: 'Supprimer « {name} » et son détourage ?',
  },
  'roto.layers.active': { en: 'Corrections go to this character', fr: 'Les corrections vont à ce personnage' },

  // export
  'roto.export': { en: 'Export', fr: 'Exporter' },
  'roto.export.prores': {
    en: 'ProRes 4444 with transparency (After Effects, Premiere, Resolve, Vegas)',
    fr: 'ProRes 4444 avec transparence (After Effects, Premiere, Resolve, Vegas)',
  },
  'roto.export.png': { en: 'PNG sequence with transparency', fr: 'Séquence PNG avec transparence' },
  'roto.export.matte': { en: 'Black and white mask only (matte)', fr: 'Masque seul noir et blanc (matte)' },
  'roto.export.selection': {
    en: 'Add the cut-out shot to my selection (and to the timeline export)',
    fr: 'Ajouter le plan détouré à ma sélection (et à l\'export timeline)',
  },
  'roto.export.none': { en: 'Choose at least one format', fr: 'Choisissez au moins un format' },
  'roto.export.done': { en: 'Exported:', fr: 'Exporté :' },
  'roto.export.addedToSelection': { en: 'Added to your selection.', fr: 'Ajouté à votre sélection.' },
  'roto.export.reveal': { en: 'Show in folder', fr: 'Afficher dans le dossier' },
  'roto.export.kind.prores': { en: 'ProRes', fr: 'ProRes' },
  'roto.export.kind.png_dir': { en: 'PNG', fr: 'PNG' },
  'roto.export.kind.matte': { en: 'Matte', fr: 'Matte' },
  'roto.export.needPass': { en: 'Cut out the shot first', fr: 'Détourez d\'abord le plan' },

  // shortcuts overlay
  'roto.keys.title': { en: 'Keyboard shortcuts', fr: 'Raccourcis clavier' },
  'roto.keys.brush': { en: 'Brush', fr: 'Pinceau' },
  'roto.keys.eraser': { en: 'Eraser (put back the AI pass)', fr: 'Gomme (remet le passage de l\'IA)' },
  'roto.keys.click': { en: 'Clicks', fr: 'Clics' },
  'roto.keys.alt': { en: 'Hold: remove / negative click', fr: 'Maintenir : retirer / clic négatif' },
  'roto.keys.size': { en: 'Smaller / bigger brush', fr: 'Pinceau plus petit / plus grand' },
  'roto.keys.undo': { en: 'Undo', fr: 'Annuler' },
  'roto.keys.redo': { en: 'Redo', fr: 'Rétablir' },
  'roto.keys.views': { en: 'Views: overlay, black, white, transparent, original', fr: 'Vues : calque, noir, blanc, transparent, original' },
  'roto.keys.fit': { en: 'Fit the image', fr: 'Ajuster l\'image' },
  'roto.keys.zoom': { en: 'Zoom', fr: 'Zoom' },
  'roto.keys.wheel': { en: 'Wheel', fr: 'Molette' },
  'roto.keys.pan': { en: 'Move the image', fr: 'Déplacer l\'image' },
  'roto.keys.spaceDrag': { en: 'Space+drag / middle button', fr: 'Espace+glisser / bouton du milieu' },
  'roto.keys.space': { en: 'Space', fr: 'Espace' },
  'roto.keys.homeEnd': { en: 'Home / End', fr: 'Début / Fin' },
  'roto.keys.shiftN': { en: 'N / Shift+N', fr: 'N / Maj+N' },
  'roto.keys.redoKeys': { en: 'Ctrl+Shift+Z / Ctrl+Y', fr: 'Ctrl+Maj+Z / Ctrl+Y' },
  'roto.keys.play': { en: 'Play / pause', fr: 'Lecture / pause' },
  'roto.keys.frames': { en: 'Previous / next frame', fr: 'Image précédente / suivante' },
  'roto.keys.ends': { en: 'First / last frame', fr: 'Première / dernière image' },
  'roto.keys.suspects': { en: 'Next / previous frame to check', fr: 'Image à vérifier suivante / précédente' },
  'roto.keys.propagate': { en: 'Propagate both ways', fr: 'Propager dans les deux sens' },
  'roto.keys.validate': { en: 'Validate this frame', fr: 'Valider cette image' },
  'roto.keys.fullscreen': { en: 'Full screen', fr: 'Plein écran' },
  'roto.keys.help': { en: 'This help', fr: 'Cette aide' },
  'roto.keys.close': { en: 'Close the help / the editor', fr: 'Fermer l\'aide / l\'éditeur' },
  // server job messages (backend/api/roto.py), shown under the progress bar
  'roto.msg.cutting': { en: 'Cutting out the characters', fr: 'Détourage des personnages' },
  'roto.msg.suspects': { en: 'Looking for frames to check', fr: 'Recherche des images à vérifier' },
  'roto.msg.following': { en: 'Following the corrections along the shot', fr: 'Suivi des corrections le long du plan' },
  'roto.msg.writing': { en: 'Writing the files', fr: 'Écriture des fichiers' },
};
