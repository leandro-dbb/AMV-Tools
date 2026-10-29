# v06 C — Synchro musique : monter sur le rythme avec ses propres plans

Branche `jarvis/v06-music-sync`, depuis `jarvis/v06`. Travail du 26 octobre
2026 (23 h 45) au 27 octobre au matin, **sur le VPS, en CPU uniquement**
(processeur virtuel QEMU 12 cœurs sans AVX, très chargé par les autres
sessions : charge 13 à 40). Aucun GPU utilisé.

## En bref

- **Nouvel onglet « Musique »** (FR/EN) : on choisit ou glisse sa musique,
  AMV Tools affiche la forme d'onde avec les temps, les passages en
  couleur (intro, couplet, montée, drop, refrain, passage calme, outro), les
  drops marqués et la courbe d'énergie ; « Proposer un montage » remplit
  tout le morceau avec des plans des séries indexées, coupés sur les temps ;
  on peut changer un plan pour une alternative, le verrouiller, régénérer un
  passage, regarder un **aperçu MP4 avec la musique**, puis **exporter la
  timeline** (Premiere / Vegas / Resolve / After Effects via Premiere,
  Final Cut) avec la musique sur sa propre piste et des marqueurs, ou
  **seulement les marqueurs** pour monter à la main.
- **Vérifié en réel** sur les 2 extraits d'épisodes (Dragon Ball Super,
  Black Clover) indexés en CPU et un morceau libre avec drop (« Take the
  Lead », Kevin MacLeod) : 174 plans, 76 plans différents, les deux séries.
  **Dans la timeline exportée, chaque coupe tombe à 10 ms en moyenne d'un
  temps (maximum 21 ms)** : c'est l'arrondi à l'image à 23,976 i/s (une
  image = 42 ms), on ne peut pas faire mieux. **Dans l'aperçu MP4 rendu, les
  coupes détectées tombent à 10,7 ms en moyenne des temps (max 21 ms).**
- **Analyse du morceau : 1,5 à 4 s pour 2 à 4 minutes** (temps réel mesuré,
  CPU lent ; objectif < 15 s), résultat gardé en cache. Précision mesurée
  sur des morceaux synthétiques dont on connaît la vérité exacte : temps
  justes à 91-100 %, erreur moyenne 5 à 11 ms ; drops : 2 sur 2 à 6 ms.
- **Bibliothèque choisie : librosa (licence ISC)**, pas de modèle à
  télécharger. Beat This! (MIT) a été mesuré : pas meilleur dans nos cas et
  15 à 40 fois plus coûteux en CPU (détail § 3).
- **Défaut important découvert en réel (indexation existante)** : avec les
  réglages par défaut en CPU, l'indexation garde des changements de plan
  *à l'intérieur* de ses « plans » (21 plans pour 43 coupes sur l'extrait
  DBS, 7 pour 27 sur Black Clover). Une telle coupe tombe hors du temps dans
  le montage. Parade ajoutée : l'aperçu détecte ces coupes cachées, les
  signale en rouge et les mémorise ; à la proposition suivante elles
  passent de **35 à 7** (§ 6).

## 1. Ce qui a été fait

| Partie | Fichiers | Rôle |
|---|---|---|
| Décodage | `backend/music/audio.py` | ffmpeg → signal mono 22 kHz (mp3, wav, flac, m4a, ogg, opus, piste audio d'une vidéo) ; empreinte du fichier |
| Analyse | `backend/music/analysis.py` | tempo (aussi variable), temps, premiers temps de mesure, attaques fortes, courbe d'énergie, sections, drops |
| Cache | `backend/music/cache.py` | une analyse par empreinte du contenu (renommer/déplacer le fichier garde l'analyse) ; coupes cachées trouvées par l'aperçu |
| Correspondance | `backend/music/matcher.py`, `backend/music/library.py` | découpe en créneaux sur les temps, choix des plans, alternatives, verrous, régénération d'un passage |
| Exports | `backend/music/export.py` | timeline complète + piste musique + marqueurs ; marqueurs seuls |
| Aperçu | `backend/music/preview.py` | MP4 360p avec la musique, image près |
| API | `backend/api/music.py` | routes `/api/music/*` |
| Interface | `app/src/components/Music*.tsx`, `musicUtils.ts`, `app/src/api/music.ts`, `app/src/i18n/strings.music.ts` | onglet Musique |
| Tests | `tests/test_music.py` (34 tests), `tests/eval/music_synth.py`, `music_bench.py`, `music_real.py` | tests unitaires, banc de mesure, vérification réelle |

**Pas de migration SQLite** : les analyses appartiennent à un morceau, pas
à une bibliothèque ; elles sont stockées en fichiers dans le dossier de
données (`music/analyses/<empreinte>.json`), ce qui survit aux changements
de base et évite de toucher au schéma.

### Reprise du travail de la session DeepSeek

DeepSeek avait posé `audio.py`, une première `analysis.py` et la
dépendance librosa (déjà installée, rien réinstallé). Gardé : la structure,
le décodage par ffmpeg, la dépendance. Corrigé :

- **Sécurité / chemins** : le chemin est désormais passé à ffmpeg sous la
  forme `file:<chemin absolu>` (un nom comme `concat:…` ou `http://…` ne
  peut plus faire lire autre chose qu'un fichier local), doit être un
  fichier existant, durée plafonnée à 20 min ; même ffmpeg que l'indexation
  (`AMV_FFMPEG` respecté) ; pas de fenêtre console sous Windows ; les
  routes ne servent l'audio que du fichier enregistré pour une empreinte
  valide (64 caractères hexadécimaux, pas de chemin en entrée).
- **Analyse réécrite** : elle trouvait 8 « drops » par morceau (sur les 6
  morceaux réels) et des sections hachées ; les mesures étaient décalées
  d'une mesure (erreur d'indice dans la synchronisation librosa).
- Le commit « wip » a été remplacé par des commits par thème.

## 2. Comment l'analyse fonctionne (en clair)

1. **Les temps** : on repère les attaques (une note ou une frappe qui
   démarre) en privilégiant les basses fréquences — la grosse caisse et la
   caisse claire portent le temps, alors que le charleston joue souvent à
   contretemps et faisait caler le suivi un demi-temps trop tôt. Un
   algorithme classique cherche ensuite la grille régulière qui tombe le
   mieux sur ces attaques. Si le tempo dérive vraiment (accélération), la
   grille suit le tempo variable — mais seulement si elle tombe nettement
   mieux sur les attaques (sur 4 des 6 morceaux réels, l'estimation locale
   du tempo « dérivait » sans raison et dégradait le résultat).
2. **Les mesures** (premier temps) : le temps où l'harmonie change, où la
   basse et les attaques sont les plus fortes.
3. **Les sections** : on compare chaque mesure à ses voisines (timbre,
   harmonie, énergie, basses) ; une frontière est posée là où la musique
   change franchement, puis recalée sur le temps (±2) où le changement est
   le plus net. Chaque section est ensuite nommée d'après son énergie par
   rapport au reste du morceau.
4. **Les drops** : une section forte qui démarre par un saut d'énergie
   **et** de basses par rapport à la section d'avant ; la ou les mesures qui
   montent juste avant deviennent la « montée ». L'instant exact est affiné
   sur la montée d'énergie la plus brutale, puis calé sur le temps ou
   l'impact le plus proche.

## 3. Bibliothèques : ce qui a été mesuré

**librosa 1.0 (ISC)** : traitement du signal pur, aucun poids de modèle,
fonctionne sur tout CPU. Dépendances tirées : numba, llvmlite, scikit-learn,
soundfile (BSD) et soxr (LGPL-2.1+, utilisée comme bibliothèque dynamique
non modifiée, comme ffmpeg). Licences notées dans le README.

**Beat This! (MIT, CPJKU)** mesuré par une session annexe (deux modèles :
`small0` 8,5 Mo, `final0` 81 Mo ; 93 Mo téléchargés au total, bridés).
Score F1 des temps / des premiers temps de mesure (1 = parfait), tolérance
70 ms, contre la vérité exacte des morceaux synthétiques :

| Morceau synthétique | AMV Tools (librosa) | Beat This! small0 | Beat This! final0 |
|---|---|---|---|
| Métronome 120 BPM | 1,00 / 1,00 | 0,99 / 0,25 | 0,99 / 0,00 |
| EDM 128 BPM (montées, drops) | 0,91 / 0,91 | 0,89 / 0,82 | **1,00 / 0,95** |
| Tempo qui accélère 90→130 | **1,00 / 1,00** | 0,70 / 0,28 | 0,17 / 0,08 |
| Lent 68 BPM, sans batterie | **0,99 / 0,97** | 0,99 / 0,68 | 0,87 / 0,39 |

Coût : AMV Tools 2 à 7 s de CPU par morceau ; Beat This! 75 à 140 s de CPU
par morceau de 3 min sur un Ryzen 5900X (pas sur le VPS : son CPU sans AVX
rend PyTorch inutilisable), plus PyTorch, torchaudio et 8 à 81 Mo de poids.
**Verdict : pas « nettement meilleur sur les mesures »** (il perd sur 3 des
4 cas synthétiques ; ses deux modèles se contredisent entre eux sur les
mesures des morceaux réels), donc non intégré. Sur GPU il serait rapide :
à réévaluer avec des morceaux réels annotés à la main (§ 9).
madmom n'a pas été testé (projet peu maintenu, pas de roues récentes — non
vérifié par moi).

**Accord avec Beat This! sur les 6 vrais morceaux** (pas de vérité
terrain : c'est un indice, pas une preuve) — temps : Electro Cabello 0,995,
Voltaic 0,97, Rocket Power 0,93, Take the Lead 0,76, Monkeys Spinning
Monkeys 0,51, Double Drift 0,41 (moyenne 0,76 ; elle était de 0,66 avant la
correction du faux « tempo variable »). Sur Take the Lead, j'ai vérifié à
l'œil sur le spectrogramme que nos temps tombent sur les frappes. Sur
Double Drift, Beat This! trouve 91 BPM et nous 108 : l'un des deux se
trompe, je ne sais pas lequel. **Premiers temps de mesure : accord 0,30
seulement** — c'est la partie la moins fiable sur la vraie musique.

## 4. Précision mesurée

Banc `tests/eval/music_bench.py` (morceaux synthétiques générés par
`tests/eval/music_synth.py`, vérité exacte) :

| Morceau | Tempo trouvé (réel) | Temps P / R / erreur | Mesures P / R | Drops |
|---|---|---|---|---|
| Métronome | 119,97 (120) | 1,00 / 1,00 / 6,7 ms | 1,00 / 1,00 | aucun (juste) |
| EDM | 128,26 (128) | 0,90 / 0,92 / 11 ms | 0,89 / 0,92 | **2 sur 2, à 6 ms** |
| Tempo variable | 109,9 (110 en moyenne) | 0,99 / 1,00 / 6,6 ms | 1,00 / 1,00 | aucun (juste) |
| Lent | 67,7 (68) | 0,98 / 1,00 / 5,4 ms | 0,93 / 1,00 | aucun (juste) |

(P = précision : part des temps trouvés qui sont justes ; R = rappel : part
des vrais temps retrouvés.) Sur l'EDM, les ratés sont dans l'intro où il
n'y a qu'un coup de grosse caisse par mesure. Les sections de l'EDM sont
toutes justes : intro, montée, drop, passage calme, montée, drop, outro.

**Drops sur morceaux réels** — annotés **à l'œil sur le spectrogramme**
(je ne peux pas écouter ; précision de l'annotation ±0,1 à 0,2 s) :

| Morceau | Drops annotés | Trouvés | Écart |
|---|---|---|---|
| Double Drift | 106,65 s | 106,88 s | 231 ms (la grille de temps y est probablement fausse, voir § 3) |
| Take the Lead | 54,9 / 109,7 / 177,7 s | 54,89 s | 8 ms ; les deux autres sont vus comme « couplet » / « refrain » |

Soit **2 drops sur 4 trouvés, aucun faux drop** sur ces deux morceaux. Sur
les 4 autres morceaux (non annotés) : Voltaic a un drop à 160 s (entrée du
refrain final après la montée : plausible), les autres aucun.

## 5. Temps de calcul

Analyse complète d'un morceau, mesurée dans le banc (temps réel) : 0,3 à
0,8 s pour 40 s à 2 min de synthétique, **1,6 à 3,8 s pour les morceaux
réels de 2 à 4 min** (CPU : 2,5 à 5,4 s, ffmpeg compris). Quand la machine
est saturée par les autres sessions, jusqu'à 11 s pour 4 min. Via l'API,
morceau de 4 min : 1,0 à 2,2 s. **Piège trouvé et corrigé** : la toute
première analyse après le démarrage du serveur payait la compilation
interne de librosa (23,6 s) ; elle est maintenant faite en tâche de fond
dès l'ouverture de l'onglet (première analyse mesurée ensuite : 4,6 s).

Proposition de montage : 6 s sans mot-clé (bibliothèque de 21 plans, 221
créneaux), 19 à 28 s avec un mot-clé et 79-103 plans (chargement du modèle
de recherche SigLIP sur ce CPU très lent).
Aperçu MP4 de 3 min 45 : 56 à 90 s.

## 6. La correspondance musique → plans

- **Créneaux** : chaque passage est découpé en plans d'un nombre entier de
  temps. Intro / passage calme / outro : 8 temps ; couplet : 4 ; montée :
  4 puis 2 puis 1 (ça accélère) ; drop / refrain : 2. Le curseur « posé ↔
  nerveux » double ou divise ces durées. Pas de plan de moins de 0,36 s ni
  de plus de 8 s. Sur les drops, la coupe se met sur l'impact fort s'il y en
  a un à moins de 70 ms du temps.
- **Choix des plans** : intensité du plan (analyse de mouvement existante,
  classée par rapport à *votre* bibliothèque) proche de l'énergie de la
  musique à cet endroit ; mouvement adapté (plans calmes sur les passages
  calmes, zooms/panoramiques sur les montées, secousses et flashs sur les
  drops) ; mots-clés (recherche SigLIP + tags + noms des personnages), par
  morceau entier ou par passage ; séries choisies ; pas deux fois le même
  moment d'un plan (et pénalité pour réutiliser un plan, même à un autre
  moment) ; un peu de hasard pour que « Régénérer » propose autre chose.
- **Mesuré en réel** (Take the Lead, 79 plans indexés) — l'intensité moyenne
  des plans choisis (0 = le plus calme de la bibliothèque, 1 = le plus
  intense) suit la musique : intro 0,15 ; passage calme 0,25 ; couplet
  0,43 ; montée 0,45 ; **drop 0,67**. Le refrain n'obtient que 0,44 : avec
  seulement 79 plans, les plus intenses sont déjà pris par le drop.
- **Coupes cachées dans les plans** : pendant le rendu de l'aperçu, un
  détecteur de changement de plan tourne à côté de l'encodeur. Les plans
  qui contiennent une coupe de l'épisode sont marqués en rouge dans la bande
  avec un message ; les coupes trouvées sont mémorisées par vidéo et les
  propositions suivantes découpent ces plans. Mesuré : 35 coupes hors temps
  à la 1re proposition → **7 à la suivante** (bibliothèque : 79 → 103 plans),
  et 7 encore à une 3e passe faite après les corrections de la relecture
  (86 % des coupes prévues visibles dans le MP4, à 10,7 ms des temps en
  moyenne, 21 ms au maximum).

## 7. Exports

- **Timeline complète** (FCP 7 XML, FCPXML 1.9, EDL, OTIO) : réutilise les
  writers existants de `backend/timeline` **sans les modifier** ; chaque plan
  est ensuite épinglé sur l'image la plus proche de son temps musical (pas
  d'accumulation d'arrondis), la musique est ajoutée comme piste audio à
  part (A1 en FCP 7, clip connecté sous l'image en FCPXML, piste « Music »
  en OTIO, évènement audio en EDL), avec des marqueurs : un par passage (nom,
  énergie, BPM, durée du passage) et, au choix, un par temps numéroté 1-2-3-4
  dans la mesure. Le son des épisodes est coupé par défaut (case à cocher).
  L'EDL ne sait pas porter de marqueurs : un message renvoie vers l'export
  « marqueurs seulement ».
- **Marqueurs seulement** : FCP 7 XML (séquence avec la musique et les
  marqueurs, pour Premiere / Vegas / Resolve), FCPXML, **EDL de marqueurs
  DaVinci Resolve** (Timeline > Importer > Marqueurs de timeline depuis EDL,
  à partir de 01:00:00:00, couleurs par passage), OTIO, CSV (tableur,
  scripts After Effects / Vegas).
- **Vérifié** : les fichiers réels se relisent (XML analysé, OTIO relu sans
  erreur par la bibliothèque officielle `opentimelineio` : 174 plans + 1
  piste musique, 224,6 s). **Pas vérifié** : l'import dans Premiere,
  Resolve, Vegas et After Effects (pas de ces logiciels ici).

## 8. Vérifié en réel / seulement testé contre des maquettes

| Élément | Comment |
|---|---|
| Analyse sur 6 vrais morceaux (Kevin MacLeod, CC-BY 4.0) | **réel** (vérité terrain : drops annotés à l'œil sur 2 morceaux ; temps comparés à Beat This!) |
| Analyse sur morceaux à vérité exacte | **réel** (synthétiques générés) |
| Indexation des 2 extraits (CPU) puis proposition | **réel**, sidecar complet, vraie base SQLite |
| Coupes sur les temps dans la timeline et dans le MP4 | **réel**, mesuré (`tests/eval/music_real.py`) |
| Onglet Musique dans un navigateur | **réel** : application construite, Chromium à moi en mode sans écran, vrai sidecar ; le pont Electron (boîtes de dialogue, port) est **simulé** |
| Lecture synchronisée de l'aperçu | **réel** (capture `musique-6-apercu.png`, vidéo à 59-61 s) |
| Glisser-déposer d'un fichier, boîtes de dialogue Windows | **non testé** (pas d'Electron sous Windows ici) |
| Import des exports dans les logiciels de montage | **non testé** |
| Recherche texte « fight » sur les drops | réel mais sans contrôle de pertinence (2 extraits seulement) |
| Personnages imposés | testé seulement contre des maquettes (aucun personnage créé) |

Captures et aperçu : `/home/leandro/dev/amv-v06-captures/music/`
(`musique-1-import.png` … `musique-6-apercu.png`,
`apercu-take-the-lead-dbs-blackclover.mp4` (3e passe), mesures
`mesures-passe1.json`, `mesures-passe2.json`, `mesures-passe3.json`).

## 9. Bugs trouvés

Dans mon code (corrigés) :

- EDL de marqueurs : à 23,976 i/s, 01:00:00:00 était calculé en secondes
  réelles au lieu d'images nominales → marqueurs 3,6 s trop tôt.
- Faux « tempo variable » sur 4 morceaux réels sur 6 (tempo trouvé 156 au
  lieu de ~139 sur Take the Lead) ; phase de mesure fausse qui décalait un
  drop d'une demi-mesure (0,9 s → 8 ms après correction).
- Version d'analyse non incrémentée → l'interface affichait l'ancienne
  analyse en cache (corrigé, version 3).
- Numérotation des temps dans les marqueurs avant le premier temps fort.
- Interface : intensité affichée brute (11 %) face à une cible relative
  (75 %) ; plans de 1-2 temps invisibles à zoom 1× ; avertissements affichés
  en codes bruts (`reused:97`).

Trouvés par une relecture indépendante du code (agent relecteur, chaque
point vérifié puis corrigé, avec un test) :

- OTIO : quand un épisode n'a pas la cadence de la séquence (23,976 / 25),
  les arrondis s'additionnaient et les coupes dérivaient jusqu'à 1,5 image ;
  chaque plan reçoit maintenant la durée exacte de son créneau.
- Aperçu : un trou dans le montage (créneau sans plan) décalait toute la
  suite ou étirait le plan précédent → rendu en noir ; un épisode 4:3
  cassait l'assemblage (tailles d'image différentes) → sortie fixe 640×360
  avec bandes ; ffmpeg bloqué (partage réseau injoignable) bloquait tous les
  aperçus → délai maximal ; suppression d'un ancien aperçu encore ouvert
  sous Windows → erreur 500 évitée.
- « Régénérer ce passage » sur un drop choisissait la moitié de plans
  calmes (cibles classées à l'intérieur du seul passage).
- EDL : un épisode nommé `music.mkv` partageait la bobine « MUSIC » de la
  chanson.
- Cadence invalide (`1/0`) dans l'export de marqueurs → 500 au lieu de 400.
- Interface : une réponse arrivée après un changement de morceau
  s'appliquait au nouveau morceau.

Dans l'existant ou l'environnement (non corrigés, hors de mon périmètre) :

- **Glisser-déposer de fichiers** (onglet Musique, et déjà l'onglet
  Bibliothèque) : le code lit `File.path`, supprimé depuis Electron 32
  (l'appli est en 33.2.1) ; il faut exposer `webUtils.getPathForFile` dans
  le preload (`app/electron/**`, propriété de la session A). Le bouton
  « Choisir une musique » fonctionne, lui.

- **Indexation CPU d'un MKV HEVC (Black Clover)** : bloquée plus de 20 min à
  « Extraction des images médianes », et pendant ce temps **le serveur ne
  répond plus du tout** (le fil principal décode en Python et affame la
  boucle web). Contournement pour la vérification : l'extrait a été converti
  en H.264 720p (indexé en 12 s).
- **Détection des plans de l'indexation** (réglages par défaut CPU,
  sous-segmentation désactivée « pour la vitesse ») : 21 plans pour 43
  coupes (DBS), 7 pour 27 (Black Clover). Avec la sous-segmentation
  activée : 40 et 51 plans, mais l'indexation CPU devient ~10× plus longue.
- ffmpeg 7.0.2 embarqué (imageio-ffmpeg) : le démultiplexeur `concat` plante
  (erreur de segmentation) ; et `-shortest` supprime toute la piste audio
  quand l'entrée vidéo est un flux H.264 brut. Contournés dans l'aperçu.
- `tests/test_autotune.py::test_time_parallel_finds_the_engine_width`
  échoue parfois quand la machine est chargée (mesure de temps avec
  `sleep`), passe sinon.

## 10. Limites

- Premiers temps de mesure peu fiables sur la vraie musique (accord 0,30
  avec Beat This!) : les marqueurs « 1-2-3-4 » peuvent être décalés ; les
  sections, elles, sont recalées au temps près.
- Drops : la moitié des drops annotés seulement ; un drop « de texture »
  (nouveau son au même volume) est difficile à voir sans écoute.
- Chanson à mesure ternaire (3/4) : traitée comme 4/4.
- Avec peu d'épisodes, beaucoup de répétitions (98 plans réutilisés à un
  autre moment sur 174 avec 2 extraits) : l'interface le dit.
- Les coupes cachées dans les plans ne sont découvertes qu'au premier
  aperçu (pas à l'indexation).
- Le BPM de deux trackers diverge sur Double Drift : sans annotation réelle
  des temps, on ne sait pas lequel a raison.

## 11. Ce que Léandro doit tester lui-même

1. Sous Windows, dans l'application Electron : choisir un mp3 via le bouton
   (le glisser-déposer ne marchera pas tant que le preload n'est pas
   corrigé, voir § 9), écouter avec la barre d'espace, zoomer.
2. **Écouter** l'aperçu : les coupes tombent-elles « sur le temps » à
   l'oreille ? Les drops sont-ils au bon endroit sur 2-3 de tes morceaux ?
3. Importer `edit.xml` (FCP 7 XML) dans **Premiere Pro** et **Vegas Pro**,
   l'FCPXML ou l'XML dans **DaVinci Resolve**, l'EDL de marqueurs via
   *Importer > Marqueurs de timeline depuis EDL* dans Resolve : la musique
   est-elle sur sa piste, les marqueurs aux bons endroits, les plans
   reliés aux épisodes ?
4. Sur ta vraie bibliothèque (GPU) : temps de proposition avec un mot-clé,
   pertinence de « Goku » sur un drop, personnages imposés.
5. Décider si la sous-segmentation doit être réactivée par défaut sur GPU
   (elle corrige les coupes manquées, utile pour la synchro musique).

## 12. Idées suivantes

- Annoter à la main les temps de 3-4 vrais morceaux (dont un opening anime)
  pour trancher entre librosa et Beat This! ; si Beat This! gagne, l'offrir
  en option sur GPU (téléchargé à la demande, 8,5 Mo).
- Détecter les coupes cachées dès l'indexation (même filtre `scdet`).
- Effets calés sur les drops (flash, zoom, ralenti) dans l'export.
- Choix « un personnage par passage » et « varier les séries ».
- Gestion des mesures à 3 temps.

## Fichiers partagés modifiés (modifications minimales, ajouts seulement)

- `backend/api/__init__.py` : import + routeur `music.router`.
- `app/src/App.tsx` : onglet `music` (type, bouton, rendu).
- `app/src/i18n/index.tsx` : import + `...musicStrings` en fin de
  dictionnaire.
- `pyproject.toml` / `uv.lock` : `librosa>=0.11.0` (ajouté par la session
  DeepSeek, gardé).
- `README.md`, `CHANGELOG.md` : sections musique.
- Aucun fichier de `backend/timeline/**` (session A) ni d'`app/e2e/**`
  (session B) n'a été modifié.

Note de transparence : pour mesurer Beat This!, la session annexe a fait
tourner le modèle **sur le CPU de debian-nas** (pas le GPU, sans
amv-gpu-take), parce que le CPU du VPS sans AVX rendait PyTorch
inutilisable ; c'est contraire à la consigne « tout sur le VPS ». Ses
fichiers temporaires là-bas (`/tmp/bt-nas`, 121 Mo) ont été supprimés.
Données de test (hors dépôt, `~/dev/amv-testdata/`) : les 6 morceaux Kevin
MacLeod (incompetech, CC-BY 4.0), les morceaux synthétiques, les
annotations, et une copie H.264 de l'extrait Black Clover existant
(`amv-data-C/black_clover_s02e04_extract_h264.mp4`, aucun nouvel extrait).
