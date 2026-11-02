# Test de l'installeur macOS sur le vrai MacBook (M1 Pro)

Branche `jarvis/mac-install-fixes`, lundi 2 novembre 2026, 15 h 51 → 17 h 30.
Machine : le MacBook Pro M1 Pro 16 Go de Léandro, macOS 26.6.2, sur batterie
(100 % → 80 %). Installeur de départ : `.dmg` du run
[41463521345](https://github.com/leandro-dbb/AMV-Tools/actions/runs/41463521345)
(0.5.0-beta.2, branche `jarvis/windows-test`), puis le `.dmg` corrigé de
cette branche (run 41476210144, puis 41479621066 pour la fuite du
dossier temporaire, voir § 6).

## En bref

- **Le parcours complet fonctionne sur Mac**, de l'ouverture du `.dmg` au
  détourage exporté en ProRes 4444 : installation en **78 s** (dont 21 s
  pour le moteur Apple Silicon), indexation d'un épisode H.264 en 1 min 43
  et d'un épisode HEVC en 1 min 39 (après correction), recherche FR/EN,
  personnage, exports, synchro musique, finition, roto.
- **Premier lancement** : macOS 26 bloque l'appli (« Élément « AMV Tools »
  non ouvert »), avec **« Placer dans la corbeille » comme bouton par
  défaut** — c'est ce qui est arrivé au premier essai : l'appli a fini à la
  corbeille. Le mode d'emploi exact est au § 1 ; le README le disait mal
  (« clic droit → Ouvrir », qui ne marche plus depuis macOS 15).
- **14 problèmes trouvés, 12 corrigés** (12 correctifs + la documentation, tests de non-régression),
  dont un **plantage du moteur** à la relance (recherche tapée dans les
  premières secondes → « Failed to fetch » partout, appli inutilisable),
  une **double lecture des épisodes HEVC** (+2 min 18 sur 3 min 54), la
  **boîte de finition à moitié hors de la fenêtre**, le **Neural Engine
  abandonné au premier upscale**, et 14 Mo laissés dans le dossier
  temporaire à chaque fermeture.
- Non vérifié : le clic sur « Ouvrir quand même » (bouton système, il faut
  l'autorisation Accessibilité + le mot de passe, impossible à distance) ;
  l'import réel dans Final Cut Pro (pas installé sur ce Mac).

## 0. Comment le test a été mené

| Élément | Ce qui a été fait |
|---|---|
| Point de départ | Désinstallation complète de l'essai précédent : `~/Applications/AMV Tools.app`, `~/Library/Application Support/amv-tools` (2,1 Go), `~/Library/Preferences/com.amvtools.app.plist`. `~/.cache/huggingface` et `~/Library/Application Support/AMVTools` avaient déjà été mis de côté par l'essai arrêté (dans `~/amv-tools-test/install-test/backup-before/`) : le Mac était donc vierge de tout modèle. |
| Téléchargement | `.dmg` copié par `scp`, puis marqué en quarantaine **comme un téléchargement Safari** (`com.apple.quarantine`) pour que Gatekeeper réagisse comme chez un testeur. |
| Double-clic / glisser | `open` sur le `.dmg` (même chemin que le double-clic), copie par le **Finder** (AppleScript `duplicate`) vers `~/Applications` : la quarantaine suit, comme avec un glisser. |
| Lancement | `open ~/Applications/AMV Tools.app` dans la session graphique de Léandro (pas de lancement SSH). Un port de débogage (`--remote-debugging-port`) a été ajouté pour piloter la page avec Playwright (`connectOverCDP`) ; vrais clics dans l'interface. |
| Captures | `screencapture` échoue en SSH (permission « Enregistrement de l'écran ») : lancé **dans le Terminal déjà ouvert**, qui a la permission (fenêtre ciblée par son numéro pour ne pas capturer le reste de l'écran). Captures de l'appli : capture interne de la page. Dossier hors dépôt. |
| Limites | Aucune appli n'a l'autorisation Accessibilité et `sudo` demande un mot de passe : impossible de cliquer dans une fenêtre système (dialogue Gatekeeper, Réglages, sélecteur de dossier). Les deux boîtes natives de l'appli (« Ajouter un dossier », « Enregistrer sous ») ont été remplacées par l'appel d'API que l'appli fait juste après. |

## 1. Premier lancement et mode d'emploi pour les testeurs — vérifié en réel

1. Double-clic sur le `.dmg` : il s'ouvre en 1 s sur la fenêtre « AMV Tools
   0.5.0-beta.2-arm64 » (icône de l'appli + raccourci Applications)
   [`01-dmg-ouvert.png`].
2. Glisser l'appli dans Applications (ici `~/Applications`) : 294 Mo.
3. Premier double-clic sur l'appli : **refusé**. Message exact (capture
   `02-gatekeeper-dialogue.png`) :

   > **Élément « AMV Tools » non ouvert**
   > Apple n'a pas pu confirmer que « AMV Tools » ne contenait pas de
   > logiciel malveillant susceptible d'endommager votre Mac ou de porter
   > atteinte à votre vie privée.
   > [ **Placer dans la corbeille** ] (bouton bleu, par défaut) — [ Terminé ]

   Ni « endommagé » ni « développeur non identifié » : la signature ad hoc
   est acceptée, c'est l'absence de notarisation qui bloque (journal
   `syspolicyd` : « Prompt shown », puis « Terminating process due to
   Gatekeeper rejection »). **Au premier essai, l'appli est partie à la
   corbeille 5 s après l'affichage** (« moveItemToTrash » dans le journal) :
   le bouton par défaut a été activé — un testeur qui tape Entrée perd
   l'appli.
4. Après « Terminé », **Réglages Système › Confidentialité et sécurité**,
   section Sécurité, affiche : « « AMV Tools » a été bloqué pour protéger
   votre Mac. » avec le bouton **Ouvrir quand même**
   [`03-reglages-ouvrir-quand-meme.png`]. Ce bouton n'a pas pu être cliqué
   à distance (voir § 0).
5. Ce qui a été utilisé et qui marche :
   `xattr -dr com.apple.quarantine ~/Applications/AMV\ Tools.app`, puis
   double-clic : l'appli s'ouvre sans autre question.

**Mode d'emploi à donner aux testeurs (macOS 15 et plus)** :

1. Ouvrir le `.dmg`, glisser **AMV Tools** dans **Applications**.
2. Double-cliquer sur AMV Tools. macOS dit « Élément « AMV Tools » non
   ouvert ». Cliquer **Terminé** — surtout pas « Placer dans la corbeille »
   (le bouton bleu), et ne pas taper Entrée.
3. Ouvrir **Réglages Système › Confidentialité et sécurité**, descendre
   jusqu'à « Sécurité », cliquer **Ouvrir quand même** en face de « « AMV
   Tools » a été bloqué… », valider avec le mot de passe ou Touch ID, puis
   **Ouvrir** dans la fenêtre suivante. À faire une seule fois (et à chaque
   nouvelle version).
4. Variante Terminal (vérifiée) : avant le premier lancement,
   `xattr -dr com.apple.quarantine "/Applications/AMV Tools.app"`.
5. Le clic droit → Ouvrir ne suffit plus depuis macOS 15 (non retesté ici).

Le README (section macOS) et `docs/RELEASING.md` disent maintenant cela.

## 2. Premier lancement : temps, téléchargements, disque — vérifié en réel

| Étape | Temps | Détail |
|---|---|---|
| Lancement → écran « Bienvenue » | 40 s | uv télécharge Python 3.13.18 (24 Mo) et les paquets de base (torch 77 Mo, opencv 46 Mo, llvmlite 39 Mo…) [`04`, `05`] |
| « Installer et continuer » (Apple Silicon recommandé, profil Précis) → appli | 21 s | ajout coremltools, onnxruntime… [`06`] |
| **Total jusqu'à l'appli utilisable** | **78 s** | puis un tutoriel de 4 étapes (« Passer ») |
| Trafic réseau pendant l'installation | ≈ 518 Mo | compteur de l'interface Wi-Fi, autres applis comprises : ordre de grandeur |
| Cache uv | purgé | « uv cache prune (engine install): cache 1.68 -> 0.00 GB, disk freed 0.65 GB » ; il reste 1,6 Mo |

Disque après installation : appli 294 Mo + `amv-tools/.venv` 1,1 Go +
Python 71 Mo. Après le test complet (tous les modèles téléchargés) :
données de l'appli 2,0 Go (`.venv` 1,2 Go, modèles de finition et
programmes Core ML 374 Mo, exports 246 Mo) + **modèles Hugging Face 2,4 Go
dans `~/.cache/huggingface`** (SigLIP 2 1,4 Go, BiRefNet 424 Mo, wd-tagger
361 Mo, SAM 2.1 tiny 149 Mo, TransNetV2 29 Mo), soit **≈ 4,7 Go**.
À savoir pour désinstaller : ces modèles ne sont pas dans le dossier de
l'appli (README complété).

Défaut corrigé : l'écran disait « Détecté sur ce PC : aucun GPU
utilisable » tout en recommandant Apple Silicon ; il dit maintenant
« Détecté sur cet ordinateur : Apple M1 Pro » (vérifié sur le build corrigé :
`/api/setup/hardware` → `Apple M1 Pro`).

## 3. Diagnostic, réglage automatique, réseau de coupes — vérifié en réel

- Paramètres › Diagnostic [`07-diagnostic.png`] : PyTorch 2.11.0, Python
  3.13.18, ffmpeg 7.1, accélération `mps`, périphérique actif « Apple
  Silicon GPU (MPS) ».
- **Réglage automatique**, premier passage (clip synthétique) : **114 s**.
  8 décodeurs VideoToolbox, tagger sur **Core ML** (0,045 s/image contre
  0,070 en MPS), lot de 8. Le réseau de coupes est ignoré : TransNetV2
  n'est pas téléchargé par défaut.
- TransNetV2 (29 Mo) : l'option « Coupes précises » renvoyait à
  « Paramètres › Modèles », où il n'existe pas ; le seul bouton était dans
  Diagnostic et s'appelait « Réessayer le téléchargement ». Corrigé (lien
  et libellé « Télécharger ») ; le réglage automatique mesure maintenant le
  réseau dès son téléchargement au lieu d'attendre le lancement suivant.
- Réglage relancé avec TransNetV2 : 87 s ; **réseau de coupes le plus
  rapide sur Core ML / Neural Engine** (0,66 ms/image contre 1,0 en MPS et
  4,4 sur CPU). Surcoût mesuré +13,5 s par épisode (puis +10,7 s sur un vrai
  épisode) : l'option reste manuelle (seuil 10 %). L'interface affichait
  « environ +1 min » pour 13,5 s : corrigé (« +14 s »).
- En indexation réelle, coupes précises activées sur l'extrait de 4 min
  de DBS : worker Core ML prêt en 5,5 s, **87 coupes au lieu de 71**, pas
  plus lent (24 s contre 27 s).
- Au lancement suivant (bibliothèque non vide) le réglage se refait sur un
  vrai épisode : 61 s, en arrière-plan, sans gêner la recherche.

## 4. Indexation — vérifiée en réel, un bug corrigé

| Épisode | Avant correction | Après |
|---|---|---|
| DBS Super Beerus 01 (H.264, 1,4 Go, 23 min) | **1 min 43** (décodage 95 s), 397 coupes | — |
| Black Clover S02E04 (HEVC 10 bits, 608 Mo, 24 min) | **3 min 54** dont 2 min 18 de seconde lecture | **1 min 39** (vérifié sur le build corrigé) |

L'interface reste fluide pendant l'indexation (délai d'affichage mesuré
toutes les 2 s : 10 ms en général, 32 ms au pire) ; progression, nom du
fichier et « ~N s restantes » s'affichent [`09-indexation-debut.png`].

Bug : la vidéo de Black Clover finit 6 s avant l'audio, et une coupe est
trouvée sur ses dernières images ; la dernière « scène » n'avait donc
aucune image décodée, n'était pas transmise au découpage des longues
scènes fait pendant la lecture, et ce seul manque faisait **relire tout
l'épisode** (« Découpage des longues scènes (lecture de 890 images) » puis
« Extraction de 560 images clés »). Corrigé : la dernière image décodée la
représente. Le nombre total de scènes passe de 1 096 à 1 091 (Black Clover
est maintenant découpé comme DBS, pendant la lecture).

## 5. Recherche, personnages, exports, musique, finition, roto — vérifiés en réel

| Fonction | Résultat |
|---|---|
| Recherche EN « explosion in the sky » | 78 résultats pertinents en 0,9 s [`10`] |
| Recherche FR « Goku qui crie de colère » | 164 résultats en 5,2 s (première requête : chargement du modèle) |
| Personnage « Beerus » | 3 exemples choisis dans la recherche intégrée → 113 scènes, toutes Beerus en tête [`12`] |
| Export de clip (1,1 s) | `dbs-beerus-01_scene_444.mp4`, 1080p H.264 + audio, encodé VideoToolbox ; nom sans doublon ; un second export réécrit le même fichier [`14`] |
| Export timeline, 78 clips (4 min 04, 23,976 i/s) | Premiere (`.xml`), Resolve et Final Cut (`.fcpxml` 1.9), After Effects (`.jsx`), Vegas (`.txt`), OTIO : tous écrits, XML bien formés, chemins `file://` corrects. Import réel non fait (Final Cut absent, Resolve non pilotable). **Final Cut Pro n'ouvre pas les `.mkv`** et rien ne le disait : avertissement ajouté. |
| Synchro musique (« Voltaic », Kevin MacLeod, CC-BY) | analyse 9 s : 120 BPM, 9 passages, 1 drop ; montage proposé en 1 s (137 plans) ; aperçu vidéo 22 s ; « 17 plans contiennent une coupe » → Régénérer → 7 [`18`–`20`] |
| Upscale ×2 (1,1 s) | bouton « Télécharger » (2,5 Mo, 2 s), estimation 21 s, réel 23 s [`22`] |
| Ralenti ×2 | bouton « Télécharger » (RIFE 4.26, 24,6 Mo, 5 s), 5,3 s sur Core ML [`23`] |
| Profondeur | bouton « Télécharger » (Video Depth Anything Small, 116 Mo, 13 s), 27 images en 4,3 s, PNG 16 bits [`24`] |
| Roto (« Détourer ce plan », 44 images) | passe IA 2 min 06 dont téléchargement BiRefNet + SAM 2.1 tiny (~80 s) et suivi 35 s ; 9 images « à vérifier » ; 2 corrections (gomme, pinceau) en ~0,1 s chacune ; **export ProRes 4444 avec alpha** (yuva444p12le, 44 images) + PNG + matte en 20 s [`26`–`34`] |

Bugs vus et corrigés à cette étape :

- **Boîte de finition coupée** : son titre et ses premières options
  étaient au-dessus de la fenêtre (haut à −133 px). Elle était rendue dans
  l'en-tête flouté du mini-éditeur, que Chromium traite comme le cadre de
  ses éléments `fixed`. Elle est maintenant rendue au niveau de la page
  (vérifié : de 265 à 608 px dans une fenêtre de 872).
- **Premier upscale sans Neural Engine** : « Neural Engine upscale stopped
  (OSError(66, 'Directory not empty')) », 26 images sur 27 sur le GPU. Les
  deux processus Neural Engine préparaient le même programme Core ML sous
  le même nom temporaire. Vérifié sur le build corrigé, cache vidé : plus
  d'erreur.
- Le menu Finition annonçait « RIFE 4.25 » et « Depth Anything V2 » alors
  que le palier Mac 16 Go utilise RIFE 4.26 et Video Depth Anything Small.

## 6. Relance, recherche immédiate, fermeture — vérifiées en réel

- **Fermeture** (bouton rouge de la fenêtre) : 0,9 s, **aucun processus
  restant** (Electron, uv, Python, workers Core ML).
- **Relance + recherche tapée tout de suite** (version d'origine) : le
  moteur Python **plante** (SIGSEGV dans un noyau Metal de PyTorch, rapport
  `python3.13-2026-11-02-164835.ips`) ; l'appli reste ouverte avec « Failed
  to fetch » partout [`35-relance-recherche.png`]. Cause : le préchauffage
  de la recherche et la recherche faisaient tourner SigLIP sur MPS en même
  temps — PyTorch MPS ne le supporte pas. Une recherche pendant
  l'indexation était exposée de la même façon. Corrigé (passes SigLIP à
  tour de rôle sur MPS). **Build corrigé : interface prête en 3,9 s,
  131 résultats 11,5 s après le lancement, pas de plantage** ; le réglage
  automatique démarre ensuite en arrière-plan et la recherche ne l'attend
  pas [`40`].
- Si le moteur meurt quand même, l'appli le **relance seule** (2 fois
  max., puis message d'erreur) : vérifié en tuant le moteur par SIGSEGV —
  nouveau moteur prêt en ~3 s, recherche aussitôt servie [`43`].
- **Dossier temporaire** : chaque session qui chargeait le réseau de
  coupes y laissait sa copie compilée Core ML (14 Mo) — 91 restes
  (265 Mo) des sessions de développement précédentes sur ce Mac. Le moteur
  tuait ses workers sans leur laisser le temps de nettoyer. Premier
  correctif insuffisant (vérifié : la copie restait), second correctif :
  le worker libère son modèle et sort aussitôt. Vérifié sur le dernier build : pendant l'exécution la copie apparaît (14 → 15 dans le dossier temporaire) et **disparaît à la fermeture** (15 → 14), fermeture en 2,0 s.
- Autres restes de la session, sans gravité : trois verrous vides
  `uv-setuptools-*.lock` (uv, à l'installation).

## 7. Bugs trouvés

| # | Bug | Gravité | Correction | Test |
|---|---|---|---|---|
| 1 | Premier lancement : bouton par défaut « Placer dans la corbeille », mode d'emploi du README faux pour macOS 15+ | haute (testeurs) | README + RELEASING (`454aaad`) ; la vraie solution est de notariser l'appli (compte développeur Apple) | — |
| 2 | Plantage du moteur (MPS) si l'on cherche juste après le lancement ou pendant l'indexation | **critique** | `55fca39` | pytest |
| 3 | Moteur mort = appli bloquée sur « Failed to fetch » | haute | relance auto `2cc6994` | vérifié sur le Mac (SIGSEGV simulé) |
| 4 | Épisode HEVC dont la vidéo finit avant l'audio relu en entier (+2 min 18) | haute (perf) | `2bb234b` | pytest + vérifié (3 min 54 → 1 min 39) |
| 5 | Boîte de finition à moitié hors de la fenêtre | haute (UI) | `c519935` | E2E + vérifié |
| 6 | Premier upscale : Neural Engine abandonné | moyenne (perf ×2) | `708ad72` | pytest + vérifié |
| 7 | 14 Mo laissés dans le dossier temporaire à chaque session | moyenne | `82c411e` puis `fa90900` | pytest + vérifié |
| 8 | Final Cut Pro + sources MKV : clips hors ligne sans avertissement | moyenne | `4f3ce6d` | vitest |
| 9 | « Coupes précises » renvoie vers un écran sans TransNetV2 ; bouton « Réessayer » pour un premier téléchargement ; coût mesuré seulement au lancement suivant | moyenne | `e809f4a` | pytest + vérifié |
| 10 | « Détecté sur ce PC : aucun GPU utilisable » sur un M1 Pro | basse | `58149ad` | pytest + vérifié |
| 11 | Coût des coupes précises arrondi à « +1 min » pour 13 s | basse | `ecb5d7a` | vitest |
| 12 | Menu Finition : noms de modèles faux pour le palier | basse | `7d0c27b` | — |
| 13 | Un dossier `*.partial.mlpackage` de l'ancienne course (bug 6) reste dans le cache de l'upscale | basse | non corrigé (sans effet, quelques Mo) | — |
| 14 | Le mini-éditeur ouvert depuis Personnages nomme le plan `scene0432` et l'upscale `scene444 / #443` (identifiant et index mélangés) | basse | non corrigé | — |

## 8. Ce qui n'a pas pu être vérifié

- Le clic sur « Ouvrir quand même » et la fenêtre de confirmation qui suit
  (bouton système, Accessibilité + mot de passe).
- Le sélecteur de dossier et la boîte « Enregistrer sous » (natifs) : les
  chemins ont été passés par l'API que ces boîtes appellent.
- L'ouverture des timelines dans Final Cut Pro (absent du Mac), Premiere,
  After Effects, Vegas ; DaVinci Resolve est installé mais n'est pas
  pilotable sans Accessibilité.
- La suite E2E (`npm run e2e`) avec le nouveau contrôle de la boîte de
  finition n'a pas été rejouée (le correctif a été vérifié sur le Mac).

## 9. Vérifications

- VPS : `uv run pytest -m "not slow"` → **603 passés, 19 ignorés, 0 échec** (sur le dernier commit de code) ; `cd app && npx tsc
  --noEmit && npx vitest run && npm run build` → **OK** (88 tests vitest).
- Mac : build corrigé installé par-dessus l'ancien (mise à jour), relance,
  recherche immédiate, plantage simulé, upscale à froid, réindexation HEVC,
  coupes précises, fermeture.
- Fin de test, sur le Mac : appli, `~/Library/Application Support/amv-tools`,
  préférences `com.amvtools.app`, modèles Hugging Face téléchargés pendant
  le test (`~/.cache/huggingface`, 2,4 Go), rapport de crash et 72 restes
  AMV du dossier temporaire (copies Core ML des sessions de développement,
  dossiers `amv-autotune-*`, `amv_searchplus_*`…) supprimés ; médias
  (`~/amv-tools-test/media`) intacts ; batterie à 80 %.
- Restent sur le Mac, à décider par Léandro : la sauvegarde faite par
  l'essai arrêté, `~/amv-tools-test/install-test/backup-before/` (5,8 Go :
  anciennes données `amv-tools` et `AMVTools`, et l'ancien
  `~/.cache/huggingface` — uniquement SigLIP 2 et wd-tagger — à remettre en
  place ou à supprimer) ; l'appli mise à la corbeille par Gatekeeper au
  premier essai (dans la Corbeille, inaccessible en SSH) ; 32 dossiers
  `tmp*.mlpackage` / `.mlmodelc` de coremltools (33 Mo) dans le dossier temporaire
  (sessions de développement antérieures, non supprimés faute de pouvoir
  les attribuer avec certitude).
