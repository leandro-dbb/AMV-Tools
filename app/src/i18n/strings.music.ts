import type { Dict } from './index';

export const musicStrings: Dict = {
  'music.tab': { en: 'Music', fr: 'Musique' },
  'music.title': { en: 'Edit on a music', fr: 'Monter sur une musique' },
  'music.subtitle': {
    en: 'Drop your music: AMV Tools finds its beats, build-ups and drops, then proposes an edit with the shots of your series, cut on the beat.',
    fr: 'Dépose ta musique : AMV Tools repère ses temps, ses montées et ses drops, puis te propose un montage avec les plans de tes séries, coupé sur les temps.',
  },

  // import
  'music.import.drop': { en: 'Drop a music file here', fr: 'Dépose un fichier musique ici' },
  'music.import.formats': {
    en: 'MP3, WAV, FLAC, M4A, OGG, AAC, Opus — or a video, its sound will be used',
    fr: 'MP3, WAV, FLAC, M4A, OGG, AAC, Opus — ou une vidéo, on prendra son son',
  },
  'music.import.or': { en: 'or', fr: 'ou' },
  'music.import.choose': { en: 'Choose a music', fr: 'Choisir une musique' },
  'music.import.pathPlaceholder': { en: 'Full path of the audio file…', fr: 'Chemin complet du fichier audio…' },
  'music.import.analyze': { en: 'Analyse', fr: 'Analyser' },
  'music.import.analyzing': { en: 'Listening to the music…', fr: 'Écoute de la musique…' },
  'music.import.analyzingHint': {
    en: 'Finding the tempo, the beats, the build-ups and the drops. Usually 5 to 15 seconds ({seconds} s).',
    fr: 'Recherche du tempo, des temps, des montées et des drops. Compte 5 à 15 secondes ({seconds} s).',
  },
  'music.import.failed': { en: 'Could not analyse this file: {message}', fr: 'Impossible d’analyser ce fichier : {message}' },
  'music.import.dropNoPath': {
    en: 'Could not read the path of the dropped file. Use “Choose a music” instead.',
    fr: 'Impossible de lire le chemin du fichier déposé. Utilise plutôt « Choisir une musique ».',
  },
  'music.import.recent': { en: 'Already analysed', fr: 'Déjà analysées' },
  'music.import.recentEmpty': { en: 'No music analysed yet.', fr: 'Aucune musique analysée pour l’instant.' },
  'music.import.open': { en: 'Open', fr: 'Ouvrir' },
  'music.import.change': { en: 'Change music', fr: 'Changer de musique' },
  'music.import.loadFailed': { en: 'Could not reopen this music: {message}', fr: 'Impossible de rouvrir cette musique : {message}' },

  // overview
  'music.overview.bpm': { en: '{bpm} BPM', fr: '{bpm} BPM' },
  'music.overview.duration': { en: 'Duration', fr: 'Durée' },
  'music.overview.tempo': { en: 'Tempo', fr: 'Tempo' },
  'music.overview.beats': { en: 'Beats', fr: 'Temps' },
  'music.overview.drops': { en: 'Drops', fr: 'Drops' },
  'music.overview.sections': { en: 'Passages', fr: 'Passages' },
  'music.overview.noBeats': {
    en: 'No clear beat found in this music: the edit will be cut on its strongest hits instead.',
    fr: 'Pas de temps net trouvé dans cette musique : le montage sera coupé sur ses coups les plus forts.',
  },
  'music.player.play': { en: 'Play', fr: 'Lecture' },
  'music.player.pause': { en: 'Pause', fr: 'Pause' },
  'music.player.toStart': { en: 'Back to start', fr: 'Revenir au début' },
  'music.player.source.music': { en: 'Playing: music only', fr: 'Lecture : musique seule' },
  'music.player.source.preview': { en: 'Playing: video preview', fr: 'Lecture : aperçu vidéo' },
  'music.player.audioError': { en: 'The music cannot be played in the app.', fr: 'La musique ne peut pas être lue dans l’app.' },
  'music.zoom': { en: 'Zoom', fr: 'Zoom' },
  'music.zoomIn': { en: 'Zoom in', fr: 'Zoomer' },
  'music.zoomOut': { en: 'Zoom out', fr: 'Dézoomer' },
  'music.wave.clickToSeek': { en: 'Click to play from here', fr: 'Clique pour lire à partir d’ici' },
  'music.wave.drop': { en: 'DROP', fr: 'DROP' },
  'music.legend.energy': { en: 'Energy', fr: 'Énergie' },
  'music.legend.beat': { en: 'Beat', fr: 'Temps' },
  'music.legend.bar': { en: 'Start of bar', fr: 'Début de mesure' },
  'music.legend.drop': { en: 'Drop', fr: 'Drop' },
  'music.sections.hint': {
    en: 'Click a passage to give it its own keywords.',
    fr: 'Clique sur un passage pour lui donner ses propres mots-clés.',
  },

  // section names
  'music.section.intro': { en: 'Intro', fr: 'Intro' },
  'music.section.verse': { en: 'Verse', fr: 'Couplet' },
  'music.section.build': { en: 'Build-up', fr: 'Montée' },
  'music.section.drop': { en: 'Drop', fr: 'Drop' },
  'music.section.chorus': { en: 'Chorus', fr: 'Refrain' },
  'music.section.bridge': { en: 'Calm passage', fr: 'Passage calme' },
  'music.section.outro': { en: 'Outro', fr: 'Outro' },
  'music.section.unknown': { en: 'Passage', fr: 'Passage' },

  // constraints
  'music.constraints.title': { en: 'What you want', fr: 'Ce que tu veux' },
  'music.constraints.series': { en: 'Series', fr: 'Séries' },
  'music.constraints.seriesAll': { en: 'All series', fr: 'Toutes les séries' },
  'music.constraints.seriesNone': {
    en: 'No series yet (set one when importing in the Library).',
    fr: 'Pas encore de série (à choisir à l’import dans la Bibliothèque).',
  },
  'music.constraints.characters': { en: 'Characters', fr: 'Personnages' },
  'music.constraints.charactersAny': { en: 'Any character', fr: 'N’importe quel personnage' },
  'music.constraints.charactersNone': {
    en: 'No character yet (create them in the Characters tab).',
    fr: 'Pas encore de personnage (crée-les dans l’onglet Personnages).',
  },
  'music.constraints.density': { en: 'Cut rhythm', fr: 'Rythme des coupes' },
  'music.constraints.densityCalm': { en: 'calm', fr: 'posé' },
  'music.constraints.densityNervous': { en: 'nervous', fr: 'nerveux' },
  'music.constraints.avoidReuse': { en: 'Never reuse a shot', fr: 'Ne pas réutiliser un plan' },
  'music.constraints.query': { en: 'Wish for the whole edit', fr: 'Envie pour tout le montage' },
  'music.constraints.queryPlaceholder': { en: 'e.g. fight, explosion', fr: 'ex : combat, explosion' },
  'music.constraints.sectionQuery': { en: 'Keywords for this passage', fr: 'Mots-clés pour ce passage' },
  'music.constraints.sectionQueryPlaceholder': { en: 'e.g. Goku', fr: 'ex : Goku' },
  'music.constraints.sectionPick': {
    en: 'Click a passage above the waveform to give it keywords (e.g. “Goku” on the drop).',
    fr: 'Clique sur un passage au-dessus de la forme d’onde pour lui donner des mots-clés (ex : « Goku » sur le drop).',
  },
  'music.constraints.sectionRange': { en: '{name} · {start} – {end}', fr: '{name} · {start} – {end}' },
  'music.constraints.clear': { en: 'Clear', fr: 'Effacer' },

  // proposal
  'music.propose': { en: 'Propose an edit', fr: 'Proposer un montage' },
  'music.proposing': { en: 'Picking the shots…', fr: 'Choix des plans…' },
  'music.regenerate': { en: 'Regenerate (except locked)', fr: 'Régénérer (sauf verrouillés)' },
  'music.regenerateHint': {
    en: 'Re-picks every shot that is not locked, with a new draw.',
    fr: 'Refait le choix de tous les plans non verrouillés, avec un nouveau tirage.',
  },
  'music.proposeFailed': { en: 'Could not build the edit: {message}', fr: 'Impossible de construire le montage : {message}' },
  'music.edit.summary': {
    en: '{count} shots · {locked} locked',
    fr: '{count} plans · {locked} verrouillés',
  },
  'music.edit.empty': {
    en: 'Choose what you want, then click “Propose an edit”: the shots will appear here, aligned on the music.',
    fr: 'Choisis ce que tu veux puis clique sur « Proposer un montage » : les plans apparaîtront ici, alignés sur la musique.',
  },
  'music.edit.noClips': {
    en: 'No shot matches these choices. Try fewer series / characters or a shorter wish.',
    fr: 'Aucun plan ne correspond à ces choix. Essaie avec moins de séries / personnages ou une envie plus courte.',
  },
  'music.edit.warnings': { en: 'To know', fr: 'À savoir' },
  'music.library.empty.title': { en: 'No indexed shots yet', fr: 'Aucun plan indexé pour l’instant' },
  'music.library.empty.body': {
    en: 'To propose an edit, AMV Tools needs your episodes. Import them in the Library tab, then come back here: your music stays loaded.',
    fr: 'Pour proposer un montage, AMV Tools a besoin de tes épisodes. Importe-les dans l’onglet Bibliothèque puis reviens ici : ta musique reste chargée.',
  },
  'music.library.open': { en: 'Go to the Library', fr: 'Aller à la Bibliothèque' },

  // clip panel
  'music.clip.title': { en: 'Shot {slot}', fr: 'Plan {slot}' },
  'music.clip.placement': { en: 'On the music', fr: 'Sur la musique' },
  'music.clip.beats': { en: '{count} beats', fr: '{count} temps' },
  'music.clip.source': { en: 'In the episode', fr: 'Dans l’épisode' },
  'music.clip.intensity': { en: 'Intensity', fr: 'Intensité' },
  'music.clip.wanted': { en: 'wanted {value}', fr: 'voulue {value}' },
  'music.clip.unknown': { en: 'not measured', fr: 'non mesurée' },
  'music.clip.alternatives': { en: 'Other shots for this spot', fr: 'Autres plans pour cet endroit' },
  'music.clip.alternativesHint': { en: 'Click one to swap it in.', fr: 'Clique sur l’un d’eux pour l’échanger.' },
  'music.clip.alternativesNone': { en: 'No other shot proposed here.', fr: 'Pas d’autre plan proposé ici.' },
  'music.clip.lock': { en: 'Lock this shot', fr: 'Verrouiller ce plan' },
  'music.clip.unlock': { en: 'Unlock', fr: 'Déverrouiller' },
  'music.clip.lockedHint': { en: 'Locked: kept when you regenerate.', fr: 'Verrouillé : gardé quand tu régénères.' },
  'music.clip.regenerateSection': { en: 'Regenerate this passage', fr: 'Régénérer ce passage' },
  'music.clip.playHere': { en: 'Play from here', fr: 'Lire à partir d’ici' },
  'music.clip.openEditor': { en: 'Open the shot', fr: 'Ouvrir le plan' },
  'music.clip.close': { en: 'Close', fr: 'Fermer' },
  'music.clip.prev': { en: 'Previous shot', fr: 'Plan précédent' },
  'music.clip.next': { en: 'Next shot', fr: 'Plan suivant' },
  'music.clip.pick': { en: 'Click a shot in the strip to change it or lock it.', fr: 'Clique sur un plan de la bande pour le changer ou le verrouiller.' },
  'music.clip.locked': { en: 'Locked', fr: 'Verrouillé' },

  // preview
  'music.preview.title': { en: 'Video preview', fr: 'Aperçu vidéo' },
  'music.preview.start': { en: 'Video preview (low resolution)', fr: 'Aperçu vidéo (basse résolution)' },
  'music.preview.redo': { en: 'Update the preview', fr: 'Mettre à jour l’aperçu' },
  'music.preview.running': { en: 'Building the preview… {percent}%', fr: 'Fabrication de l’aperçu… {percent} %' },
  'music.preview.failed': { en: 'The preview failed: {message}', fr: 'L’aperçu a échoué : {message}' },
  'music.preview.stale': {
    en: 'The edit changed since this preview: update it to see the new shots.',
    fr: 'Le montage a changé depuis cet aperçu : mets-le à jour pour voir les nouveaux plans.',
  },
  'music.preview.none': {
    en: 'Builds a small video of the edit with the music, to check it before exporting.',
    fr: 'Fabrique une petite vidéo du montage avec la musique, pour vérifier avant d’exporter.',
  },
  'music.preview.needEdit': { en: 'Propose an edit first.', fr: 'Propose d’abord un montage.' },
  'music.preview.musicOnly': { en: 'Listen to the music only', fr: 'Écouter la musique seule' },
  'music.preview.useVideo': { en: 'Watch the preview', fr: 'Regarder l’aperçu' },

  // export
  'music.export.title': { en: 'Send to your editing software', fr: 'Envoyer vers ton logiciel de montage' },
  'music.export.timeline': { en: 'Export the timeline', fr: 'Exporter la timeline' },
  'music.export.timelineHint': {
    en: 'The edit with every shot placed on the music, pointing at your original episodes (no re-encode).',
    fr: 'Le montage avec chaque plan posé sur la musique, qui pointe vers tes épisodes d’origine (sans réencodage).',
  },
  'music.export.markers': { en: 'Export only the markers', fr: 'Exporter seulement les marqueurs' },
  'music.export.markersHint': {
    en: 'Beats and passages as markers on the music, to edit by hand.',
    fr: 'Les temps et les passages en marqueurs sur la musique, pour monter à la main.',
  },
  'music.export.format': { en: 'Format', fr: 'Format' },
  'music.export.format.fcp7xml': {
    en: 'FCP 7 XML — Premiere Pro, After Effects, Vegas Pro, DaVinci Resolve',
    fr: 'FCP 7 XML — Premiere Pro, After Effects, Vegas Pro, DaVinci Resolve',
  },
  'music.export.format.fcpxml': { en: 'FCPXML — Final Cut Pro, DaVinci Resolve', fr: 'FCPXML — Final Cut Pro, DaVinci Resolve' },
  'music.export.format.edl': { en: 'EDL — simple, read everywhere', fr: 'EDL — simple, lue partout' },
  'music.export.format.otio': { en: 'OTIO — OpenTimelineIO', fr: 'OTIO — OpenTimelineIO' },
  'music.export.format.csv': { en: 'CSV — spreadsheet / scripts', fr: 'CSV — tableur / scripts' },
  'music.export.beatMarkers': { en: 'A marker on every beat', fr: 'Marqueurs sur chaque temps' },
  'music.export.sourceAudio': { en: 'Keep the episodes’ sound', fr: 'Garder le son des épisodes' },
  'music.export.outputPath': { en: 'Save as (full path)', fr: 'Enregistrer sous (chemin complet)' },
  'music.export.exporting': { en: 'Exporting…', fr: 'Export…' },
  'music.export.failed': { en: 'Export failed: {message}', fr: 'L’export a échoué : {message}' },
  'music.export.saved': { en: 'Saved: {path}', fr: 'Enregistré : {path}' },
  'music.export.counts': { en: '{clips} shots · {markers} markers · {fps} fps', fr: '{clips} plans · {markers} marqueurs · {fps} i/s' },
  'music.export.markerCount': { en: '{markers} markers', fr: '{markers} marqueurs' },
  'music.export.reveal': { en: 'Show in folder', fr: 'Afficher dans le dossier' },
  'music.export.needEdit': {
    en: 'Propose an edit first to export the timeline. The markers can be exported right away.',
    fr: 'Propose d’abord un montage pour exporter la timeline. Les marqueurs, eux, s’exportent tout de suite.',
  },

  'music.warn.reused': {
    en: '{n} cuts had to reuse a moment already used: index more episodes for more variety.',
    fr: '{n} coupes ont dû réutiliser un moment déjà pris : indexe plus d’épisodes pour plus de variété.',
  },
  'music.warn.repeats': {
    en: '{n} shots appear more than once (another moment of the same shot). More episodes = fewer repeats.',
    fr: '{n} plans reviennent plusieurs fois (un autre moment du même plan). Plus d’épisodes = moins de répétitions.',
  },
  'music.warn.unfilled': {
    en: '{n} cuts found no shot long enough.',
    fr: '{n} coupes n’ont trouvé aucun plan assez long.',
  },
  'music.warn.metrics_missing': {
    en: '{n} shots have no motion analysis yet: they are matched less precisely.',
    fr: '{n} plans n’ont pas encore d’analyse de mouvement : ils sont choisis moins finement.',
  },
  'music.warn.empty_library': {
    en: 'No indexed shot to use: index episodes in the Library tab first.',
    fr: 'Aucun plan indexé à utiliser : indexe d’abord des épisodes dans l’onglet Bibliothèque.',
  },
  'music.warn.no_shots_in_groups': {
    en: 'The chosen series have no indexed shot.',
    fr: 'Les séries choisies n’ont aucun plan indexé.',
  },
  'music.warn.no_character_matches': {
    en: 'No shot found yet for the chosen characters.',
    fr: 'Aucun plan trouvé pour l’instant avec les personnages choisis.',
  },
  'music.warn.query_tags_only': {
    en: 'Search model unavailable: keywords only matched tags.',
    fr: 'Modèle de recherche indisponible : les mots-clés n’ont utilisé que les tags.',
  },
  'music.warn.edl_no_markers': {
    en: 'EDL has no markers: export the markers separately (“markers only”).',
    fr: 'L’EDL ne porte pas de marqueurs : exporte-les à part (« marqueurs seulement »).',
  },
  'music.warn.other': { en: 'Note: {code}', fr: 'Remarque : {code}' },
};
