# v06 — correction des bugs connus (nuit du 29 au 30 octobre)

Branche `jarvis/v06`. Fusion des chantiers B (parcours UI / E2E) et C
(synchro musique), puis correction des bugs connus avec, pour chacun, la
cause racine, la correction et une preuve chiffrée sur des données réelles.

**Phase 3 (installation sur le PC Linux de Léandro) : annulée** sur
consigne de Léandro à 00 h 00 — aucune connexion à sa machine, rien
installé ni téléchargé dessus.

## En bref

| # | Bug | État | Preuve |
|---|---|---|---|
| — | Fusion B + C | ✅ conflits résolus, 1 régression de fusion corrigée (OTIO musique) | suite complète verte |
| 1 | Détection de coupes | ✅ nettement amélioré, **objectif 98 % non atteint** (voir § 1) | rappel 70 % → 81 % (DBS), 38 % → 86 % (Black Clover) à l'image exacte ; précision 72 % → 92 % / 67 % → 90 % ; identique CPU / NVIDIA ; Mac par lecture de code |
| 2 | HEVC fige le serveur | ✅ cause racine trouvée (interblocage PyAV), corrigé + 2ᵉ déclencheur (torchvision) | HEVC 8 bits CPU : indexé en 160 s via le vrai serveur, API répondante (576 requêtes, 0 échec) ; HEVC 10 bits sur RTX 3090 : indexé, 1 789 requêtes, 0 échec |
| 3 | Reprise d'indexation | ✅ « incomplet » + reprise là où ça s'est arrêté | tests : 0 coupe recalculée, seules les scènes manquantes retraitées |
| 4 | Glisser-déposer | ✅ Musique corrigée (lisait encore `File.path`), dépôt hors zone sécurisé | E2E `drop.e2e.mjs` 3/3 |
| 5 | Musique : drops et premiers temps | ✅ amélioré, objectif « 4/4 + 2 morceaux » atteint sur les morceaux d'origine, **pas sur tous les drops annotés** | drops 2/14 → 8/14, 0 faux ; drops tombant sur un « 1 » : 5/14 → 8/14 avec Beat This! (optionnel) |
| 6 | Test d'autotune instable | ✅ horloge virtuelle, plus aucun chronométrage réel | 3/3 passages |
| 7 | Autres bugs bêtes | ✅ 20 corrigés (revue indépendante + trouvés en route), dont 1 bloquant (dépendances après mise à jour) | voir § 7 |

## 0. Fusion

`jarvis/v06-ui-e2e` puis `jarvis/v06-music-sync` fusionnées dans
`jarvis/v06`. Conflits :

- `CHANGELOG.md` : les trois sections (A, B, C) gardées.
- `OnboardingScreen.tsx` : A choisit le backend selon le GPU détecté
  (`setupHardware`), B pré-sélectionne le profil « Rapide » quand on choisit
  le CPU. Les deux sont gardés dans une fonction commune : la suggestion du
  GPU détecté règle aussi le profil, un clic de l'utilisateur garde la main.
- `i18n/index.tsx` : les deux dictionnaires (parcours + musique).

**Régression de fusion trouvée par les tests** :
`test_otio_mixed_rates_keep_cuts_on_beats` échouait (coupes à 74 ms des
temps). A avait corrigé l'OTIO dans l'exporteur commun (durées sur la
cadence de la séquence) ; C avait corrigé le même problème dans l'export
musique en réécrivant ces durées en images source. Les deux ensemble
faussaient le résultat : la retouche de C est retirée.

## 1. Détection de coupes

**Vérité terrain.** Les deux extraits ont été annotés image par image
(chaque image regardée ; chaque coupe vérifiée sur l'image d'avant, la
coupe et l'image d'après ; planches contact relues) :
`tests/data/cuts_gt/*.json`. Il y a **67 coupes franches dans l'extrait DBS
(pas 43) et 42 dans Black Clover (pas 27)**, plus quelques fondus et des
cas ambigus notés comme tels (rafales de coups où le cadrage saute toutes
les 2-4 images, zooms par paliers) qui ne comptent ni comme ratés ni comme
fausses coupes. J'ai revérifié moi-même un échantillon de planches.

**Cause.** Tous les chemins de décodage comparaient l'histogramme de gris
d'une image sur 4 à celui de l'image 4 plus tôt, contre un seuil fixe.
Deux plans d'anime de même palette ont le même histogramme (coupe ratée),
et un flash, une explosion ou une aura changent tout l'histogramme dans un
même plan (fausse coupe). Mesuré avec l'ancien code, chemin CPU, à l'image
exacte : DBS rappel 70 %, précision 72 % ; Black Clover rappel **38 %**,
précision 67 %. (Le « 21 plans pour 43 coupes » du rapport C venait d'un
autre réglage ; sur cet extrait l'ancien code donne 68 plans.)

**Correction.** Un détecteur unique, `backend/indexing/shotdetect.py`,
utilisé par **tous** les chemins, qui voit **chaque image** (160×90 gris,
ce que chaque chemin décodait déjà) :

- différence mesurée sur la zone d'image seulement (les bandes noires des
  épisodes en 2,39:1 diluaient tout changement : Black Clover) ;
- après le meilleur petit décalage (±3 px) : un panoramique ou une
  secousse ne compte pas ;
- comparée à l'activité locale (les 4 images de chaque côté) : l'action
  rapide change beaucoup à chaque dessin ;
- rejet des flashs / images d'impact : si l'ancienne image revient dans
  les 6 images suivantes, ou si la nouvelle était déjà là juste avant, ce
  n'est pas une coupe ;
- décision 7 images après coup (le test du flash regarde devant) ; le
  curseur « seuil de détection » des Paramètres règle la sensibilité.

Les seuils ont été choisis par recherche sur grille (le pire des deux
extraits), puis vérifiés à l'envers (choisis sur un extrait, mesurés sur
l'autre). J'ai aussi essayé une régression logistique et du gradient
boosting sur les mêmes mesures (98,5 % / 100 % sur les extraits
d'apprentissage, mais 57 à 70 % sur l'autre extrait : apprentissage par
cœur, écarté), et **TransNetV2** (réseau MIT de référence) : 88-95 % de
rappel, 76-93 % de précision ici, et 14-21 images/s sur CPU (40 min par
épisode) : écarté.

**Résultat (à l'image exacte, mêmes extraits) :**

| Chemin | DBS rappel / précision | Black Clover rappel / précision |
|---|---|---|
| Ancien (CPU) | 70 % / 72 % | 38 % / 67 % |
| CPU ffmpeg (Linux/Windows sans GPU) | **80,6 % / 91,5 %** | **85,7 % / 90,0 %** |
| CPU PyAV (secours) | 80,6 % / 91,5 % | 85,7 % / 90,0 % |
| CPU passe unique (`single_pass`, aussi Mac) | 80,6 % / 91,5 % | 85,7 % / 90,0 % |
| RTX 3090 ffmpeg CUDA / `onepass` / NVDEC `nvc_pass` | 80,6 % / 93,1 % | 85,7 % / 90,0 % |
| RTX 3090, Black Clover réencodé HEVC 10 bits | — | 85,7 % / 90,0 % |

(Précision : les coupes « à une image près » comptent comme fausses ;
seuls 1-2 cas sont à une image.)

**Objectif 98 % / 98 % : non atteint.** Les ratés restants sont surtout
des cas limites que l'annotateur lui-même a marqués comme discutables
(coupe à travers un flash blanc, paliers de zoom, rafales de coups,
plan qui commence sur une image noire) ; les fausses coupes sont des
flashs d'aura longs ou des mouvements de main sur 2 images. Aller plus loin
demande la couleur (aujourd'hui seul le gris 160×90 sort des décodeurs) ou
un modèle appris sur beaucoup plus d'épisodes annotés.

**Mac (lecture de code).** Le Mac passe par `single_pass` (même code que le
CPU Linux, décodeur logiciel ou VideoToolbox) : mêmes décisions. Le
décodage parallèle par morceaux (`chunked`, Apple Silicon) recolle les
coupes aux frontières de morceaux avec l'ancienne logique ; il ne porte pas
encore le contexte du nouveau détecteur. **Il est désactivé par défaut**
(`AMV_CHUNKED=1` pour le réactiver) : le Mac indexe en une passe
séquentielle, plus lente (non mesurée ici, pas de Mac), mais avec les
bonnes coupes.

**Coût.** Le détecteur traite 1 200-1 700 images/s sur un cœur. Sur la
3090, passe NVDEC par défaut (`nvc_pass`) : 6,2 → 6,5 s (DBS) et 3,0 →
3,5 s (Black Clover) ; passe ffmpeg (`onepass`) : 6,7 → 8,5 s et 3,6 →
5,6 s. Sur CPU, le décodage ffmpeg en sous-processus est aussi plus rapide
que l'ancien décodage PyAV.

**Tests** : `tests/test_shotdetect.py` — les 3 chemins CPU sur les vrais
extraits contre la vérité terrain (s'exécute si les extraits sont présents,
`AMV_CUTS_MEDIA`), + cas synthétiques (coupe sur la bonne image, flash
d'une image, image d'impact, panoramique, bandes noires, flux = lot).

## 2. HEVC : le serveur se figeait

**Cause racine.** `backend/indexing/frames.py` appelait
`av.logging.set_level(av.logging.PANIC)` « pour faire taire ffmpeg ». Cela
installe un *callback* Python pour les logs de ffmpeg. Le décodeur HEVC
est multi-thread : à la fermeture du décodeur, ses threads écrivent des
logs et attendent le GIL… tenu par le thread d'indexation qui attend ces
mêmes threads. Interblocage, GIL tenu : **tout le processus** est figé,
API comprise. Reproduit en isolé : même script, avec l'appel → bloqué à la
fermeture ; sans → OK. Le réglage par défaut de PyAV (aucun callback) est
déjà silencieux. **Deuxième déclencheur** trouvé par le test de
non-régression : `torchvision.io` fait `av.logging.set_level(ERROR)` à
l'import, c'est-à-dire dès que les modèles se chargent ; le bug serait
revenu en plein milieu de la première indexation.

**Correction.**
- Appel supprimé ; `av.open` est enveloppé (`backend/av_quiet.py`) pour
  remettre le réglage sûr avant chaque ouverture, quoi qu'aient fait les
  bibliothèques importées.
- Sans décodage matériel (CPU Linux/Windows, AMD/Intel), la détection des
  coupes et l'extraction des images passent par **ffmpeg en
  sous-processus** (multi-thread, hors du GIL, interruptible) au lieu de
  PyAV dans le processus ; PyAV reste en secours (`AMV_DECODE=pyav`).
  L'indexation tournait déjà dans un thread séparé de la boucle asyncio ;
  le travail lourd est maintenant dans ffmpeg (autre processus).
- **NVIDIA 10 bits** (trouvé sur la 3090) : PyNvVideoCodec annonce 8 bits
  pour un HEVC 10 bits en MKV (le cas habituel des sorties anime) ; la
  passe NVDEC rapide plantait et retombait sur ffmpeg. La profondeur vient
  maintenant du format de pixel réel.

**Preuves.**
- CPU (VPS), vrai serveur, épisode Black Clover HEVC 8 bits : indexé en
  160 s, `/api/status` + `/api/videos` interrogés toutes les 250 ms pendant
  tout le run : 576 réponses, 0 échec, pire latence 2,0 s (chargement des
  modèles), sinon 5-120 ms.
- RTX 3090, vrai serveur, même épisode réencodé HEVC Main 10 : indexé
  (41 plans), 1 789 réponses, 0 échec, pire 1,45 s. NVDEC 10 bits : mêmes
  coupes que le 8 bits.
- Tests `tests/test_fix_v06.py` : HEVC 8 et 10 bits (libx265) indexés sur
  CPU par ffmpeg et par PyAV, sans blocage ; le callback de log reste
  coupé même après le réglage de torchvision.
- Mac : VideoToolbox décode HEVC 8/10 bits ; le chemin `single_pass` est
  le même code (lecture de code, pas de Mac ici).

## 3. Reprise d'une indexation interrompue

**Cause.** L'épisode était inscrit avec ses scènes dès le début (statut
`indexing`) ; la Bibliothèque comptait toutes les vidéos et toutes les
scènes quel que soit le statut (« 1 épisode · 82 scènes »), et le run
suivant effaçait tout et recommençait depuis la détection des coupes.

**Correction.**
- Un épisode non terminé (appli fermée, Stop, échec) n'est **plus compté
  comme indexé** : la Bibliothèque affiche « N épisode(s) incomplet(s) »,
  une pastille « incomplet » sur l'épisode et un bouton « Reprendre
  l'indexation ». Il n'apparaît pas dans la recherche (déjà le cas).
- Le run suivant **reprend là où il s'est arrêté** : pas de nouvelle
  détection de coupes ni de nouvelles lignes de scènes ; seules les images
  médianes des scènes encore sans tags ou sans embedding sont décodées.
  L'avancement des tags est enregistré dans la même transaction que les
  tags (aucune scène taguée deux fois) ; les embeddings, ligne par ligne.
- Les épisodes non terminés sont repris même si leur dossier n'est plus
  dans la file.
- Une erreur du tagger (ex. onnxruntime absent) était **avalée** : un
  avertissement dans le log et l'épisode marqué indexé sans un seul tag
  (vu en vrai ce soir). Elle fait maintenant échouer l'épisode
  visiblement (liste rouge), qui reste reprenable.

**Preuves** (`tests/test_fix_v06.py`, coupure simulée pendant les tags
puis pendant les embeddings) : pendant la coupure, statut ≠ `completed`,
0 vidéo et 0 scène comptées, 1 incomplète ; à la reprise, aucun appel à la
détection de coupes, mêmes identifiants de scènes, tagger et SigLIP
appelés uniquement sur les scènes restantes, aucun tag en double.

## 4. Glisser-déposer

- **Musique** : la zone lisait encore `File.path` (supprimé depuis
  Electron 32) → message « Impossible de lire le chemin » à chaque dépôt.
  Corrigé (pont `webUtils.getPathForFile`, comme la Bibliothèque).
- **Recherche par image** et **Personnages** : envoient le contenu du
  fichier, pas son chemin : non concernés, fonctionnent.
- **Dépôt hors zone** : un fichier lâché ailleurs faisait naviguer la
  fenêtre vers ce fichier (la vidéo remplaçait toute l'appli). Les dépôts
  hors zone sont ignorés, et le processus principal refuse toute
  navigation hors de l'appli (et l'ouverture de nouvelles fenêtres).
- **E2E** `app/e2e/drop.e2e.mjs` (3/3) : vrais objets `File` adossés à des
  fichiers sur disque (via un `<input type=file>` rempli par Playwright,
  exactement ce que produit un dépôt natif), déposés sur la zone de la
  Bibliothèque (le chemin arrive dans la file d'indexation) et de la
  Musique (le morceau est analysé). Limite : le geste de l'OS lui-même
  (glisser depuis Dolphin / l'Explorateur) ne peut pas être joué sous Xvfb,
  ni le dépôt d'un *dossier* — à vérifier à la main une fois.
- Trouvé en lançant l'E2E : sans `uv` embarqué (checkout de dev), l'appli
  lançait le backend avec le `python3` du système (« No module named
  uvicorn ») ; elle prend maintenant le Python du venv du projet.

## 5. Musique : drops et premiers temps

**Vérité terrain.** Les drops des 6 morceaux réels (Kevin MacLeod, CC-BY)
ont été annotés à partir des spectrogrammes et des courbes basses / volume
/ attaques (je ne peux pas écouter), avec un niveau de confiance et la
liste des moments qui *ressemblent* à des drops mais n'en sont pas :
`tests/data/music_drops_gt.json`. Résultat : **14 drops** (7 sûrs, 7
moyens), aucun dans *Monkeys Spinning Monkeys* (ses trois pauses où le même
groove reprend ne sont pas des drops). Les annotations précédentes ont été
revues : Double Drift 106,65 s confirmé, Take the Lead 54,9 s confirmé,
109,7 s gardé (confiance moyenne), **177,7 s déplacé à 178,32 s** (177,7
est le dernier coup du roulement ; le drop est sur le crash). Trois drops
de Double Drift avaient été oubliés (10,67, 32,03, 181,36 s).

**Cause des drops manqués.** Un drop n'était détecté qu'au début d'une
section ; un drop au milieu d'une section (« même boucle, beaucoup plus
fort ») ou à une mesure de la frontière était raté.

**Correction.** Détection au niveau du temps en plus : la batterie et la
basse reviennent après une coupure ou une montée (2 mesures comparées de
chaque côté), à condition que la musique après soit plus forte que le
passage *avant* la coupure ou revienne sur une frappe (kick / crash) — une
pause où le même groove reprend au même niveau n'est pas un drop. Le drop
est placé sur le temps où la basse revient ; une détection suivie de près
par un drop plus net est une montée. Version d'analyse incrémentée (les
analyses en cache sont refaites).

| | Avant | Après |
|---|---|---|
| Drops annotés trouvés (±0,25 s) | 2 / 14 | **8 / 14** (5/7 sûrs, 3/7 moyens) |
| Drops sur les morceaux d'origine (Take the Lead + Double Drift 106,7) | 2 / 4 | 3 / 4 (178,32 raté : le drop est un retour du crash, la basse ne s'arrête pas) |
| + 2 autres morceaux | — | Rocket Power 2/2, Double Drift +2 (32,0 et 181,4) |
| Détections hors annotations | 1 (Voltaic 160 s) | **0** |
| Morceau sans drop (Monkeys) | 0 | 0 |
| Banc synthétique (drops EDM, temps, mesures) | 2/2 à 6 ms | identique |

Restent ratés : Take the Lead 178,32 ; Double Drift 10,67 et Voltaic 16,04
(entrées après l'intro, passage encore faible) ; Voltaic 63,82 ; Electro
Cabello 90,8 et 155,9 (annotés « moyens », même volume avant/après).
L'objectif « ≥ 4/4 sur les morceaux annotés + 2 de plus, sans faux drop »
est atteint pour Rocket Power et les drops ajoutés de Double Drift et
**sans aucun faux drop**, mais pas pour le 4ᵉ drop de Take the Lead ni
pour tous les drops annotés ensuite.

**Premiers temps de mesure.** Indicateur réel faute de pouvoir écouter :
un drop tombe sur le « 1 ». Avec l'heuristique d'accents, 5 des 14 drops
annotés tombaient sur un premier temps (Take the Lead : les trois à 2 temps
près). **Beat This!** (CPJKU, MIT) testé sur debian-nas (AVX) : 10/14 avec
ses propres temps, petit modèle (8,5 Mo) comme grand (81 Mo). Intégré
comme **modèle optionnel** (Paramètres > Modèles, téléchargé à la demande,
8,5 Mo) : il choisit la phase de notre grille de temps pour les marqueurs
« 1 » → **8/14**. Sans lui, la phase suit désormais les drops quand il y en
a. Garde-fous : un accent net type métronome est gardé (Beat This! ratait
toutes les mesures du métronome synthétique : 0 % → nous 100 %), et ses
mesures ne sont prises que si elles tombent sur nos temps. Les sections
gardent leur propre grille (recoupées sur ses mesures, elles perdaient des
drops : 3 faux drops mesurés).

Le code du modèle est **embarqué** (`backend/music/beatthis`, Beat This! et
rotary-embedding-torch, MIT, licences jointes) avec le front-end log-mel
réécrit sur `torch.stft` pour éviter torchaudio (dont les roues doivent
correspondre à chaque build de torch : CPU, CUDA 12.6, 13, Mac). Vérifié
contre le vrai paquet sur debian-nas : écart du mel 5·10⁻⁵, temps et
premiers temps identiques sur un vrai morceau. Coût : 4-7 s par morceau sur
un CPU récent, quasi nul sur GPU, **15 à 30 s sur le CPU sans AVX du VPS**
(au-dessus de l'objectif de 15 s ; le test de rapidité ne passe pas avec le
modèle installé sur cette machine). L'onglet Musique dit d'où viennent les
« 1 » et propose le téléchargement.

Tests : `tests/test_music_drops_real.py` (vrais morceaux, si présents),
`tests/test_music_beatthis.py` (banc de filtres mel = librosa slaney,
absence du modèle sans erreur, choix de phase).

## 6. Test d'autotune instable

`test_time_parallel_finds_the_engine_width` mesurait de vrais `sleep` à
l'horloge murale : faux sous charge. Il tourne maintenant sur un moteur
simulé à deux unités avec une horloge virtuelle : mesures exactes
(`{1: 0.04, 2: 0.02, 4: 0.02}`), déterministes. Les attentes de threads
d'arrière-plan gardent un délai d'attente large (ce sont des attentes,
pas des mesures).

## 7. Autres bugs du même genre

Revue du code fusionné par un agent indépendant (sans mes conclusions),
chaque point vérifié dans le code avant correction, plus ce que j'ai
trouvé en route :

| Bug | Gravité | Correction |
|---|---|---|
| Une erreur du tagger (onnxruntime absent…) était avalée : épisode « indexé » sans un tag | bloquant | échec visible, épisode reprenable (§ 3) |
| « Lancer / Reprendre l'indexation » ne faisait rien, sans message, pendant une ré-analyse, une fusion de bibliothèques ou la passe d'aperçus de fin d'indexation | majeur | refus expliqué ; la route de la file dit si un run tourne, les boutons suivent |
| Un plantage de l'indexation hors de la boucle par fichier ne s'affichait que quelques millisecondes | majeur | listé dans les échecs |
| Un dossier de la file introuvable (disque débranché) ou sans vidéo disparaissait en silence | majeur | signalé (« introuvable », « aucune vidéo ») |
| Un épisode incomplet dont le fichier a disparu : bandeau impossible à faire partir, reprise sans effet | mineur | signalé « introuvable » à la reprise |
| Vidéo restée « en cours d'indexation » après un plantage | mineur | marquée interrompue au démarrage ; pas de pastille « incomplet » sur l'épisode en cours |
| « Similaires » en erreur 422 avec « Résultats max : illimité » (10 000 > 1 000) | majeur | plafonné à 1 000 |
| Lien de licence des modèles (AnimeJaNai, VDA Large) sans effet | mineur | les liens `_blank` s'ouvrent dans le navigateur (régression de mon blocage des fenêtres, attrapée par la revue) |
| Paramètres : version affichée « v0.2.0-alpha », accueil « v0.4 alpha » | mineur | version lue dans `package.json` |
| Export JSON des réglages : deux boîtes « enregistrer sous », la première ignorée ; import invalide sans message | mineur | un seul téléchargement ; erreur affichée |
| Étape « Downloading model » en anglais pour un utilisateur français | mineur | traduite |
| Dépôt hors zone : la vidéo remplaçait l'appli | majeur | § 4 |
| Lanceur de dev : `python3` système au lieu du venv | mineur | § 4 |
| **Mise à jour de l'appli : les nouvelles dépendances n'étaient jamais installées** (venv réutilisé avec `--no-sync`) → onglet Musique en erreur « No module named librosa » pour tous ceux qui mettent à jour | **bloquant** | si `uv.lock` a changé depuis la dernière synchro, une synchro au démarrage — avec l'extra du backend choisi |
| La réparation automatique du venv faisait un `uv run` sans extra : elle remplaçait un torch CUDA par la version CPU | majeur | même correctif (extra gardé) ; message clair si un module manque |
| Musique : un montage sauvegardé était rejoué sur une autre bibliothèque (mêmes numéros de scènes, autres images dans la timeline exportée) | majeur | le montage est lié à sa bibliothèque |
| Musique : les personnages choisis ignoraient les épisodes indexés depuis la dernière recherche | majeur | correspondances rafraîchies avant la proposition |
| Musique : « Proposer un montage » jetait les plans verrouillés | mineur | verrous gardés |
| Musique : double clic sur « Construire l'aperçu » = deux rendus ; une erreur de suivi perdait un rendu en cours | mineur | bouton bloqué pendant le rendu, suivi tolérant |
| Musique : épisodes sans série affichés « __none__ » | mineur | « Sans groupe » |

Non corrigé (mineur) : sous Linux, une carte AMD n'est pas détectée à
l'accueil (« CPU » proposé) ; en musique, « Personnages » favorise les
plans du personnage sans les imposer, et le curseur de rythme n'agit pas
sur « Régénérer ce passage » (seulement sur une nouvelle proposition).

## Machines et écarts aux consignes

- **RTX 3090 (debian-nas)** : aucun timer failsafe présent ; GPU pris
  (`amv-gpu-take`) pour les mesures de coupes, HEVC 10 bits et vitesse, puis
  rendu (`amv-gpu-release`) ; `hyperqwen-single-1` vérifié *healthy*
  ensuite. Aucun timer créé.
- **Écart** : le run d'indexation complète via le serveur sur la 3090 a
  téléchargé SigLIP 2 so400m (4,5 Go, le modèle du palier GPU, absent du
  cache) **sans bridage** (~12 Mo/s pendant 6 min) — j'aurais dû lancer ce
  test hors ligne ou sous `trickle`. Le modèle est resté dans le cache
  Hugging Face du NAS (utile à AMV Tools sur ce GPU). Les autres
  téléchargements (Beat This!, torchaudio pour la comparaison) ont été bridés
  à 5 Mo/s.
- Fichiers de travail sur le NAS (`/srv/data/amv-tools/v06fix/` : extraits,
  morceaux, paquets de comparaison Beat This!) supprimés à la fin ; restent
  les checkpoints Beat This! dans `~/.cache/torch/hub/checkpoints/` (90 Mo)
  et le dossier `repo/` synchronisé sur la branche.
- **PC de Léandro : aucune connexion** (phase 3 annulée).

## Tests

- `uv run pytest -m "not slow"` : **447 passés**, 9 ignorés (tests sur
  vrai matériel / vrais médias opt-in, et la comparaison morceaux-séquentiel
  du décodage par morceaux désactivé). Les tests sur vrais extraits et vrais
  morceaux (`test_shotdetect`, `test_music_drops_real`) tournent dès que les
  fichiers sont présents (c'est le cas sur le VPS).
- `cd app && npx tsc --noEmit && npx vitest run && npm run build` : OK
  (66 tests vitest).
- E2E (`npm run e2e`, appli construite, vrai serveur, vrais modèles CPU,
  Xvfb à part) : **10/10** sur clips synthétiques (dépôt 3/3 + parcours
  7/7) ; **parcours sur les vrais extraits** (DBS H.264 + Black Clover
  HEVC, `AMV_E2E_MEDIA`) : 6/6 (le cas « fichier source supprimé » est
  volontairement sauté avec de vrais médias) — indexation des deux épisodes
  par l'interface en 9 min sur le VPS, Bibliothèque « 2 épisodes · 101
  scènes » (60 + 41 : les plans du nouveau détecteur), recherche, aperçus,
  panier, exports clips + timelines.

## Ce qui reste / à vérifier par Léandro

1. **Coupes** : 81-86 % de rappel et ~90-92 % de précision à l'image exacte
   sur nos deux extraits — loin des 98 % demandés. Prochaine marche : faire
   sortir un peu de couleur des décodeurs (aujourd'hui gris 160×90) et
   annoter 3-4 épisodes de plus pour régler sur des données qu'on n'a pas
   vues.
2. **Mac** : décodage parallèle par morceaux désactivé (nouveau détecteur
   non porté) → indexation Mac plus lente qu'en v0.5 (non mesurée, pas de
   Mac) ; HEVC Mac vérifié seulement par lecture de code.
3. **Glisser-déposer** : à essayer une fois à la main depuis Dolphin sur la
   Bibliothèque (un dossier) et la Musique (un morceau).
4. **Musique** : écouter 2-3 morceaux à soi — les drops (8/14 sur nos
   annotations faites à l'œil) et les « 1 » (télécharger Beat This! dans
   Paramètres > Modèles pour les fiabiliser).
5. **Mise à jour** : au premier lancement après mise à jour, l'appli
   installe les nouvelles dépendances (une fois, quelques secondes à
   quelques minutes) — vérifier que le GPU reste actif ensuite.
6. Installation sur le PC Linux : à faire quand Léandro le souhaite.
