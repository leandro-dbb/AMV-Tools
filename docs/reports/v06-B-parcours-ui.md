# v06 B — l'interface utilisée de bout en bout, comme un monteur AMV

Branche `jarvis/v06-ui-e2e`, nuit du 26 au 27 octobre 2026, sur le VPS
(CPU seulement, 12 cœurs partagés avec les sessions A et C : charge 20 à 28
pendant toute la nuit, ce qui ralentit tout ce qui suit d'un facteur 3 à 5).

## En bref

- **L'appli Electron construite a été pilotée pour de vrai** (Playwright,
  sa propre instance Xvfb, profil et dossier de données temporaires, jamais
  ceux de Léandro) avec le **vrai backend** et les **vrais modèles en CPU**
  (SigLIP 2 base + wd-tagger v3 ; aucun mode de test, aucune maquette), sur
  deux **vrais extraits** de 2 min 30 (DBS Beerus 01, Black Clover S02E04)
  rangés dans `~/dev/amv-testdata/Séries test/` (espaces + accents).
- **22 bugs trouvés, 21 corrigés**, tous avec un test de non-régression
  (vitest, pytest ou E2E) sauf un changement de texte. Les plus gênants : le glisser-déposer de dossiers
  ne faisait **rien** (Electron 33), l'indexation restait figée sur « 5 % »
  pendant plus de 10 minutes sans dire ce qu'elle faisait, un fichier
  corrompu était ignoré **sans aucun message**, l'aperçu au survol ne
  montrait **jamais** de vidéo (réglage par défaut), le menu Finition de
  l'éditeur s'ouvrait sous le lecteur, et l'onglet Recherche disait
  « indexez un dossier » alors que la bibliothèque était pleine.
- **Pas vérifié jusqu'au bout** : aucune finition (upscale, ralenti,
  profondeur) n'a pu se terminer sur ce CPU (plus d'une heure pour 4 s de
  1080p) ; le lancement, la progression et l'annulation l'ont été.
- **Suite E2E rejouable** : `cd app && npm run build && npm run e2e`
  (parcours 1 → 4 + paramètres + cas d'erreur + relancement : 1 min 30 à
  5 min sur clips synthétiques, 12 min sur les vrais extraits), et un job
  CI Linux optionnel dans un fichier séparé (`.github/workflows/e2e-linux.yml`,
  déclenchement manuel) que la session A pourra reprendre.
- Captures avant/après : `/home/leandro/dev/amv-v06-captures/` (`avant/`,
  `apres/`, `apres/e2e/`).


## 1. Comment le parcours a été joué

| Élément | Ce qui a été utilisé |
|---|---|
| Appli | l'appli Electron **construite** (`npm run build`), lancée par Playwright `_electron.launch` |
| Affichage | Xvfb lancé exprès (`xvfb-run -a`), jamais un navigateur ni un port de débogage existant |
| Profil | dossier `userData` temporaire neuf (`/tmp/amv-e2e-profile-*`), donc base, réglages et logs jetables |
| Backend | le vrai sidecar Python (`uv run --no-sync`), sur CPU |
| Modèles | **les vrais** : SigLIP 2 base NaFlex + wd-vit-tagger v3 (déjà dans le cache HF, rien retéléchargé) |
| Médias | `~/dev/amv-testdata/Séries test/DBS Beerus 01 - extrait.mkv` (HEVC 1080p, 2 min 32) et `Black Clover S02E04 - extrait.mkv` (H.264 1080p, 2 min 35) |
| Ce qui est simulé | les boîtes de dialogue système (choix de dossier / « enregistrer sous ») sont remplacées par un chemin fixé, et le `uv sync` de l'écran d'installation est sauté (`AMV_E2E_SKIP_SYNC=1`, sinon il réinstallerait torch dans l'environnement de dev). Tout le reste de l'installation (réglages, marqueur) tourne vraiment. |

Deux façons de rejouer :

- **Exploration réelle** (`app/e2e/explore/*.mjs`) : les scripts qui ont servi
  cette nuit sur les vrais extraits, avec une capture par étape.
- **Suite rejouable** (`npm run e2e`) : le parcours 1 → 4 + paramètres + cas
  d'erreur + relancement de l'appli, sur deux petits clips synthétiques générés par ffmpeg dans un
  dossier piège `Séries test/Épisode spécial/` (+ un faux épisode corrompu),
  avec les vrais modèles. `AMV_E2E_MEDIA=<dossier>` la fait tourner sur de
  vrais épisodes.


## 2. Ce qui a été vérifié en réel, étape par étape

« Réel » = appli construite + vrai backend + vrais modèles + vrais extraits.
« Synthétique » = même chose sur des mires ffmpeg (suite E2E). Rien n'a été
testé contre un backend simulé.

| Étape | Résultat | Comment |
|---|---|---|
| 1. Premier lancement à vide | écran d'accueil compréhensible (choix du moteur, profil Rapide/Précis), tutoriel en 4 étapes, arrivée sur la Bibliothèque vide. | réel + E2E |
| 1. Téléchargement des modèles | les modèles d'analyse (SigLIP 2 ≈ 1,5 Go + wd-tagger ≈ 0,4 Go) **ne sont pas proposés au premier lancement** : ils se téléchargent en silence au début de la 1re indexation. Une note l'annonce désormais dans la Bibliothèque ; **pas de barre ni d'annulation** (non corrigé, voir § 5). Les modèles de finition, eux, ont taille, progression et bridage (upscale 2,5 Mo en 3 s, RIFE 25 Mo en 6 s, profondeur 99 Mo en 21 s à 5 Mo/s). | réel |
| 2. Ajout du dossier `Séries test` (espaces, accents) | OK, la file d'attente affiche le chemin. | réel + E2E |
| 2. Indexation | **Précis** (profil recommandé jusqu'ici) : DBS seul à 90 % après 25 min (sous-segmentation sur CPU). **Rapide** : DBS 13 min + Black Clover 13 min → 98 scènes (68 + 30). Machine saturée par les autres sessions : sur un PC normal ce serait 3 à 5 fois plus court, et sur GPU NVIDIA bien moins. | réel |
| 2. Progression / annulation | étape, fichier 1/2, temps restant, bouton Arrêter (après corrections). | réel + E2E |
| 2. Fermeture en pleine indexation puis reprise | la file est conservée ; « Lancer l'indexation » reprend ; les épisodes finis sont sautés. Mais l'épisode interrompu est refait de zéro (bug 20). | réel |
| 3. Recherche texte | « Goku qui crie » 45 plans (tous DBS, Goku SSJ qui hurle en tête), « Goku screaming » 40, « explosion » 5, « close-up face » 6, **« gros plan visage » 1 seul** au seuil de 10 % (le français trouve moins que l'anglais : conseil + bouton « baisser le seuil à 5 % » ajoutés). | réel |
| 3. Personnage | Beerus créé depuis 3 plans proposés par « purple cat god… » : 86 scènes sur 98 retenues au seuil auto, dont des plans de Goku — avec 3 exemples, c'est peu discriminant ; il faut rejeter (« Ce n'est pas lui ») pour affiner. | réel |
| 3. Plans similaires, filtres (mouvement, intensité, durée) | fonctionnent ; les badges de mouvement (« Pano gauche », « Zoom avant », « Tremblement »…) sont justes sur les captures. | réel + E2E |
| 3. Aperçu au survol | ne marchait jamais (bug 5) ; maintenant lu depuis la source (H.264 et, ici, aussi le HEVC du DBS). | réel + E2E |
| 4. Panier de dérush | garder des plans au clavier/bouton dans les deux épisodes (HEVC compris), panier « Bibliothèque (3) », export des clips, export timeline pour **chaque** format proposé (OTIO, FCPXML, FCP 7 XML, EDL) écrit sur disque. **Pas de réordonnancement possible** dans le panier (voir § 5). | réel (panier) + E2E (exports) |
| 4. Export d'un clip depuis l'éditeur | OK dans un dossier `Exports AMV été`. | E2E |
| 5. Finition | Upscale (1080p, 88 images) : lancé, mais **1 image en 8 min** sur ce CPU → arrêté (l'appli prévient « plusieurs minutes par seconde de vidéo »). Ralenti ×2 en 360p : progression et temps restant affichés, 26 % en 8 min. Profondeur : téléchargement, mise en file, **annulation** OK. Aucun résultat de finition n'a pu aller au bout en CPU cette nuit. | réel (1080p) + synthétique (360p) |
| 6. Paramètres | toutes les sections s'ouvrent ; FR ⇄ EN instantané ; dossier d'export en lecture seule refusé (après correction) ; Diagnostic complet (versions, moteur, disque, poids des modèles). | réel + E2E |
| 6. Fermeture | propre : l'appli et le sidecar se ferment en 0,4 à 2,6 s (mesuré), pas de processus orphelin. Un job de finition en cours est **perdu sans avertissement** (pas de confirmation à la fermeture). | réel |
| 7. Fichier supprimé après indexation | le Dérushage affiche « Fichier source introuvable » + lien vers la Bibliothèque. | E2E |
| 7. Dossier non inscriptible | refus clair (après correction). | E2E |
| 7. Vidéo corrompue | listée avec sa raison (après correction). | E2E |
| 7. Double clic rapide | « Lancer l'indexation » (une seule indexation), export, finition : protégés. | E2E (indexation), lecture du code (autres) |
| 7. 1280×720 et 2560×1440 | 1280×720 : accueil et éditeur cassés (corrigés), le reste tient ; 2560×1440 : tout tient, mais les textes sont petits (tailles en pixels fixes). | réel |


## 3. Bugs trouvés

Gravité : **bloquant** = la fonction ne marche pas ; **majeur** = le monteur
croit que l'appli est plantée ou perd du temps/du travail ; **gênant** =
confus ou faux ; **mineur** = texte, cosmétique.

| # | Bug | Gravité | État | Test de non-régression |
|---|---|---|---|---|
| 1 | **Glisser-déposer un dossier d'épisodes ne faisait rien** : Electron 32+ a supprimé `File.path`, l'appli est en Electron 33. | bloquant | corrigé (`webUtils.getPathForFile` via le preload) | vitest `dropPaths.test.ts` — *le geste lui-même n'a pas pu être joué sous Xvfb* |
| 2 | **Indexation figée sur « 5 % » plus de 10 min** (passe du tagger de la sous-segmentation, sans aucun retour), et rien ne disait ce que faisait l'appli. | majeur | corrigé : étape en clair (« Détection des coupes », « Ajout des tags 16/68 », « Construction de l'index de recherche »…), fichier *n/N*, temps restant, barre qui avance pendant la sous-segmentation | pytest `test_subsegment_tagger_pass_reports_progress`, vitest `indexProgress.test.ts`, E2E étape 2 |
| 3 | **Un fichier corrompu était ignoré en silence** (seul le journal technique le disait). | majeur | corrigé : encadré rouge dans la Bibliothèque avec le nom du fichier et la raison | pytest `test_index_queue_tags_events_with_file_rank_and_failures`, E2E étape 2 (faux épisode `Épisode abîmé 03.mkv`) |
| 4 | Pendant l'indexation, la Bibliothèque affichait encore « Prêt à indexer » et un bouton « Lancer l'indexation » actif ; la liste des épisodes ne se mettait jamais à jour ; bouton Arrêter = un petit carré sans texte. | gênant | corrigé (« Indexation en cours », boutons désactivés, liste rafraîchie, bouton « Arrêter ») | E2E étape 2 (y compris double clic sur « Lancer ») |
| 5 | **Survoler une scène ne montrait jamais de vidéo** : l'aperçu n'existait qu'avec des proxys, désactivés par défaut. | majeur | corrigé : lecture directe du fichier source (en boucle sur la scène), miniature si le codec n'est pas lisible | E2E étape 3 (`hover-preview`) |
| 6 | L'onglet Recherche accueillait une bibliothèque **pleine** avec « Indexez un dossier d'animes pour commencer ». | gênant | corrigé (« N scènes indexées, prêtes à être cherchées » + exemple de requête FR/EN) | E2E étape 3 |
| 7 | Paramètres > Recherche (seuil, nombre max, tri) **ignorés** par l'onglet Recherche (toujours 10 % / 200 / pertinence). | gênant | corrigé | vitest `searchDefaults.test.ts` |
| 8 | Écran de premier lancement **coupé en haut** en 1280×720 (titre inaccessible, impossible de remonter). | gênant | corrigé (Bootstrap + accueil) | E2E étape 1 (position du titre) |
| 9 | Le profil « Précis » était recommandé même en « CPU uniquement » (indexation 3 à 5 fois plus longue). | gênant | corrigé : « Rapide » présélectionné et expliqué si CPU | E2E étape 1 |
| 10 | Bandeau GPU : « No GPU runtime detected (torch is not installed) » alors que torch est installé, en anglais au milieu d'une phrase française ; « CPU (forced fallback) » quand on a **choisi** le CPU. | gênant | corrigé, raisons traduites | pytest ×3, vitest `fallbackLabel`, E2E étape 1 |
| 11 | Un dossier d'export **non inscriptible** était accepté sans broncher (tous les exports suivants échouent) ; les erreurs s'affichaient en brut (`API 400: {"detail": …}`) et hors de l'écran. | gênant | corrigé : refus clair, ancien dossier conservé, message collant visible | pytest `test_settings_refuse_a_read_only_export_folder`, E2E étape 6 |
| 12 | Fin d'export du panier en anglais dans l'interface française (« 2 exported »). | mineur | corrigé (« Export terminé : 2 clip(s) ») | vitest `derushExport.test.ts`, E2E étape 4 |
| 13 | Temps restant absurde au début de chaque épisode (« ~1 h 17 min » pour un extrait de 2 min 30, mesuré en réel). | gênant | corrigé : estimation à partir de l'étape des tags | vitest `etaSeconds` |
| 14 | Écran « Backend injoignable » qui montrait une commande de développeur (`uv run --extra cu130 …`) à l'intérieur de l'appli. | mineur | corrigé (consigne pour un monteur) | aucun (texte) |
| 15 | Textes : le Dérushage vide renvoyait vers « Paramètres » pour indexer (c'est la Bibliothèque) ; section « Derush » non traduite ; « AMV Tools v0.3 alpha » ; « 1 épisodes ». | mineur | corrigé | E2E (indirect) |
| 16 | Noms des clips exportés : les lettres accentuées et japonaises étaient supprimées (« Clip été 01 » → `Clip_t_01`, « Pokémon » → `Pokmon`), vu sur les fichiers réellement exportés. | gênant | corrigé (export et panier) | pytest `test_export_names_keep_accented_and_japanese_letters` |
| 17 | Éditeur de scène : la barre de lecture (Lecture, timecode) sortait de l'écran sous ~1000 px de haut, et le **menu Finition s'ouvrait sous le lecteur** (seul « Upscale » cliquable). | majeur | corrigé | E2E étape 4 (bouton Lecture entièrement visible, clic d'essai sur « Profondeur ») |
| 18 | Recherche : bouger le curseur Seuil ou changer le tri ne relançait pas la recherche ; première recherche (30 s+ sur CPU) sans retour visible ; suggestions « Gojo combat / Eva 01 rage / Pikachu cute » sur n'importe quelle bibliothèque. | gênant | corrigé | E2E étape 3 (curseur) |
| 19 | Export timeline : le bouton restait « Annuler » après un export réussi. | mineur | corrigé | E2E étape 4 |
| 21 | Échap ne fermait pas l'éditeur de scène (seulement la petite croix). | mineur | corrigé (sauf si une fenêtre ouverte depuis l'éditeur est au-dessus) | E2E étape 4 |
| 22 | Dérushage : sur les vrais extraits, deux « Garder » de suite laissaient le panier **vide, sans aucun signe**. Trois causes : pendant « Préparation de l'épisode… » (remux du MKV), **tous** les raccourcis du Dérushage étaient coupés ; tant que la vidéo n'était pas chargée, un changement de scène était annulé par le lecteur (le 2e clic retirait la 1re) ; et en changeant d'épisode, l'ancienne liste de scènes restait active quelques instants (un clic rapide visait une scène de l'épisode précédent). Trouvé par la suite E2E jouée sur les vrais extraits. | majeur | corrigé | E2E étape 4 (« 1 gardées » puis « Scène 2/… » puis « 2 gardées ») |
| 20 | Un épisode interrompu (appli fermée en pleine indexation) apparaît comme indexé (« 1 épisode · 82 scènes ») puis est **entièrement refait** à la reprise. | gênant | **non corrigé** (touche au pipeline et au schéma) | — |

## 4. Ce qui reste confus (non corrigé)

- **Modèles d'analyse** : pas de téléchargement guidé (taille, progression,
  annulation) ; il se fait au début de la première indexation, barre figée.
  Chaque lancement interroge aussi Hugging Face (requêtes HEAD) même quand les
  modèles sont là : à vérifier hors ligne.
- **Épisode interrompu** : affiché comme indexé, puis refait entièrement à la
  reprise. La reprise reste manuelle, mais la Bibliothèque dit maintenant au
  relancement que des dossiers attendent encore (E2E étape 8).
- **Échecs d'indexation** : la liste rouge disparaît au redémarrage de l'appli.
- **Fermer l'appli** pendant une indexation ou une finition ne demande pas de
  confirmation ; le job de finition est perdu.
- **Tutoiement / vouvoiement** mélangés selon les écrans (accueil « tes
  épisodes », tutoriel « Ajoutez votre bibliothèque »).
- Jargon : le tutoriel parle de « wd-tagger + SigLIP 2 », les paramètres de
  « backbone », « patches », « CRF » — opaque pour un monteur.
- Le bandeau « GPU indisponible » reste affiché sur un profil où le CPU n'a
  pas été choisi explicitement (ancienne installation) : correct, mais il ne
  dit pas comment l'enlever (Paramètres > Indexation > Périphérique = CPU).
- Sur CPU, les modèles (tagger puis SigLIP) sont **rechargés à chaque
  épisode** (le tagger est déchargé entre les phases, utile seulement pour la
  mémoire des cartes graphiques) : 1 à 2 min perdues par épisode sur ce VPS.
  Pas modifié : cela touche le mode « streaming » d'Apple Silicon.
- Personnages : avec peu d'exemples, presque toute la série est retenue (une
  phrase d'aide a été ajoutée : rejeter les mauvais plans et ajouter des
  exemples) ; la qualité de reconnaissance elle-même n'a pas été travaillée.

## 5. Les 5 améliorations d'ergonomie les plus utiles pour un monteur AMV

1. **Un vrai panier de montage** : réordonner les plans par glisser-déposer,
   et que l'export timeline (Premiere, Resolve, Vegas…) suive cet ordre.
   Aujourd'hui le panier est trié du plus récent au plus ancien et la timeline
   par épisode puis temps. *Demande une colonne « position » (migration de
   schéma : réservée à la session C).*
2. **Ajouter au panier depuis la Recherche** (sélection multiple, touche
   « garder » sur une carte) : aujourd'hui on cherche dans un onglet et on
   garde dans un autre, épisode par épisode.
3. **Téléchargement guidé des modèles au premier lancement** (taille, barre,
   annulation, reprise), et une **progression globale de l'indexation**
   (tous épisodes, temps total restant) avec reprise automatique et sans
   refaire l'épisode interrompu.
4. **Recherche bilingue plus tolérante** : traduire/compléter la requête
   française (ou chercher FR + EN et fusionner), car « gros plan visage »
   trouve 6 fois moins que « close-up face » ; et des suggestions tirées de
   la bibliothèque (personnages créés, tags fréquents) plutôt que des
   exemples génériques.
5. **Confirmations et raccourcis de monteur** : confirmation avant de fermer
   pendant un travail long (indexation, finition), et un bouton « Envoyer vers
   Premiere/Resolve » qui ouvre directement la timeline exportée dans le
   logiciel choisi.

## 6. Ce que Léandro doit tester lui-même

- **Le glisser-déposer d'un dossier d'épisodes** depuis l'Explorateur Windows
  sur la Bibliothèque (corrigé, mais impossible à jouer sous Xvfb).
- Sur le PC Windows + NVIDIA : premier lancement avec CUDA (le vrai `uv sync`,
  sauté ici), puis une indexation : les étapes et le temps restant doivent
  défiler ; le bandeau GPU ne doit pas apparaître.
- L'**aperçu au survol** sur des épisodes HEVC/10 bits (lecture directe de la
  source par Chromium : selon le codec, on doit voir la vidéo ou, à défaut, la
  miniature — jamais un cadre noir).
- Une **finition qui va au bout** (upscale, ralenti, profondeur) sur GPU :
  rien n'a pu se terminer en CPU cette nuit.
- Ouvrir dans **Premiere / Resolve / Vegas** les timelines exportées (la suite
  vérifie qu'elles sont écrites, pas qu'elles s'ouvrent).
- `npm run e2e` sur Windows n'a pas été essayé (la suite vise Linux/Xvfb ; sur
  Windows elle utilise l'écran normal).

## 7. Livrables et fichiers

- Suite E2E : `app/e2e/parcours.e2e.mjs`, `helpers.mjs`,
  `playwright.config.mjs`, `run.mjs` (`npm run e2e`, Xvfb automatique si pas
  d'écran ; `AMV_E2E_MEDIA`, `AMV_E2E_CAPTURES`). Derniers passages :
  **7/7 en 1 min 30** sur les clips synthétiques, et **6/6 (+1 sauté) en
  11 min 45 sur les deux vrais extraits** (`AMV_E2E_MEDIA="…/Séries test"`,
  indexation réelle comprise ; le test qui supprime un fichier source est
  sauté sur de vrais médias). Captures : `apres/e2e/` et
  `apres/e2e-extraits-reels/`. Scripts
  d'exploration réelle : `app/e2e/explore/`.
- Job CI optionnel : `.github/workflows/e2e-linux.yml` (manuel, met en cache
  les poids des modèles). **Fichier nouveau, les workflows existants ne sont
  pas touchés** ; jamais exécuté sur GitHub (pas de push).
- Tests : `tests/test_ui_journey.py` (9 tests backend), vitest
  `dropPaths`, `indexProgress`, `searchDefaults`, `derushExport`.
- Ancien travail DeepSeek : scripts à chemins en dur et profil dans
  `/tmp/opencode` remplacés ; le reformatage de `package.json` (tableaux de
  la config electron-builder) annulé ; `@playwright/test` épinglé en 1.49.1
  (déjà installé, rien retéléchargé). Le petit proxy de bridage qu'elle avait
  écrit (`/tmp/opencode/throttle.py`) n'est pas utilisé ni committé.

### Fichiers partagés modifiés (à signaler aux autres sessions)

- `app/electron/preload.ts` (**session A**) : une ligne, `pathForFile`
  (`webUtils.getPathForFile`) pour le glisser-déposer.
- `app/src/components/TimelineExportDialog.tsx` (fonction de A) : libellé
  « Fermer » après export, une ligne.
- `app/src/App.tsx` : barre d'état d'indexation déléguée à
  `IndexingStatus.tsx`, raisons du bandeau GPU traduites, écran
  « injoignable », props de `LibraryTab`.
- `app/src/i18n/index.tsx` : import de `strings.journey.ts` ajouté en fin de
  liste (la session C ajoutera le sien au même endroit : conflit trivial).
- `app/src/api/client.ts` / `types.ts` : ajouts (`removeFromQueue`,
  `modelsStatus`, `pathForFile`, champs d'événement `file_index`…).
- Backend : `backend/indexing/pipeline.py` (événements de progression),
  `backend/api/{status,setup,settings,derush}.py`, `backend/models/device.py`.
  Aucune migration de schéma.

## 8. Vérifications finales (27 octobre, 3 h 15)

- `uv run --no-sync pytest -m "not slow"` : **358 réussis**, 1 sauté (matériel
  réel). Un passage précédent sous forte charge a vu
  `test_finish.py::test_registry_download_checksum_and_delete` échouer une
  fois ; il passe isolé, dans son fichier (2/2) et dans la suite complète
  relancée : test sensible à la charge, sans lien avec ce chantier (le
  registre de finition n'est pas modifié). Note : `uv run` sans `--no-sync`
  resynchroniserait l'environnement de dev (il retirerait onnxruntime et
  changerait torch) ; c'est pour cela que `--no-sync` a été utilisé partout.
- `npx tsc --noEmit`, `npx vitest run` (42 tests), `npm run build` : OK.
- `npm run e2e` : 7/7 (synthétique) et 6/6 + 1 sauté (vrais extraits).

