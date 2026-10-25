# Chantier 2 — pipeline complet sur Apple Silicon (M1 Pro)

Machine : MacBook Pro M1 Pro (8 cœurs performance + 2 efficacité, GPU
16 cœurs, un encodeur vidéo matériel), 16 Go de mémoire unifiée,
macOS 26.6. Branche `jarvis/v04-perf-mac`.

Le chiffre principal mesure **tout ce que fait l'appli, de bout en bout**,
via le vrai sidecar piloté en HTTP comme l'interface
(`tests/bench_pipeline.py`) :

1. la « saison » : import puis indexation en une seule file de 3 vrais
   épisodes de 24 min (Dragon Ball Super 01 et 02 en H.264 1080p, Black
   Clover S02E04 en HEVC 10 bits ; 101 451 images, 1045 scènes) — coupes,
   mouvement/intensité/palette Search+, images médianes, tags, embeddings,
   vignettes, proxys de survol ;
2. 10 recherches texte (la première à froid), 3 recherches par image,
   5 « plans similaires » ;
3. l'export de 10 clips gardés (39 s de vidéo) et d'une timeline FCP7 /
   FCPXML / EDL / OTIO ;
4. la finition d'un clip de 5 s 1080p : upscale x2, ralenti RIFE x2,
   profondeur (PNG 16 bits).

Réglages : ceux d'une installation neuve que le bench active (mode
« balanced », sans sous-segmentation, proxys de survol activés).

## 1. Le tableau principal

| Pipeline complet (M1 Pro 16 Go) | v0.3 (historique ¹) | Avant chantier 2 | Après chantier 2 | **Tour 3** | Gain tour 3 | Gain total vs avant chantier 2 |
|---|---:|---:|---:|---:|---:|---:|
| **Total** | ≈ 42 min (reconstitué) | 7 min 07 (427 s) | 4 min 52 (292 s) | **3 min 03 (183 s)** | **×1,60 (−37 %)** | **×2,34** |
| Saison (indexation + proxys) | ≈ 39 min | 4 min 47 (287 s) | 2 min 40 (160 s) | **1 min 41 (101 s)** | ×1,58 | ×2,83 |
| Recherches | ≈ 5,5 s | 4,2 s | 4,1 s | 4,1 s | = | = |
| Export 10 clips + timelines | ≈ 10 s | 11,6 s | 6,4 s | **6,1 s** | = | ×1,90 |
| Finition (upscale, RIFE, profondeur) | ≈ 2 min 52 | 2 min 05 (125 s) | 2 min 02 (122 s) | **1 min 11 (71 s)** | ×1,70 | ×1,74 |

Par rapport à la v0.3, le pipeline complet est **≈ 13,7× plus rapide** sur ce
Mac (≈ 42 min → 3 min 03).

¹ **L'ancien code n'a pas été rejoué** (consigne). La colonne v0.3 est
reconstituée à partir des mesures historiques par étape de
`docs/PERFORMANCE.md` (même Mac, même épisode DBS 01, même profil sans
sous-segmentation) : indexation 531,8 s par épisode (×3), proxys de survol
0,71 s par scène (×1045 ≈ 12 min 22), première recherche 5,3 s, export
libx264 2,57 s par 10 s de vidéo, upscale 110 s, RIFE 26,1 s et profondeur
≈ 36 s pour 120 images. C'est un ordre de grandeur, pas un run : l'épisode
HEVC n'y était pas mesuré, et la v0.3 **plantait** à l'indexation sur Mac
sans les deux contournements du tour 1 (tagger forcé sur CPU, copies
SigLIP bloquantes).

« Avant chantier 2 » = la branche au démarrage (commit `8b12280` : tours 1
NVIDIA et Mac fusionnés, bench unique, proxys en une passe), « Après
chantier 2 » = commit `7e30e67`, « Tour 3 » = la branche finale (`55a9da1`,
run `t3-final` lancé à 06 h 03). Un run complet chacun, même script, même machine, sur
secteur avec `caffeinate`, applications de bureau de l'utilisateur ouvertes
(Chrome, Discord…). Runs complets intermédiaires du tour 3 : 3 min 10 (avec
15 158 pages swappées, voir §7), 3 min 12, 3 min 05 (export à 4 clips en
parallèle), 3 min 04 (sans le préchargement de SigLIP). Les programmes Core ML étaient déjà
convertis (caches chauds, voir §7 pour le coût de la toute première fois).

## 2. Détail par étape

Utilisation GPU = « Device Utilization % » d'`ioreg` (sans sudo), moyenne
sur l'étape ; CPU = moyenne de la machine entière (10 cœurs).

| Étape | v0.3 (historique) | Avant | Après ch. 2 | **Tour 3** | GPU avant → après → tour 3 | CPU avant → après → tour 3 |
|---|---:|---:|---:|---:|---:|---:|
| Décodage + coupes + mouvement, tags et embeddings en flux | 3 × 532 s | 135,3 s | 138,4 s ² | **91,8 s** | 69 % → 66 % → 76 % | 77 % → 78 % → 70 % |
| Vignettes + base (+ métriques Search+) | — | 2,5 s | 2,4 s | 1,9 s | | |
| **Proxys de survol** (1045 scènes) | ≈ 742 s (une par scène) | 148,6 s | 18,9 s | **6,5 s** ⁴ | 8 % → 8 % → 7 % ³ | 61 % → 84 % → 79 % |
| Première recherche texte (charge SigLIP 2) | 5,32 s | 3,72 s | 3,64 s | 3,60 s | | |
| Recherche texte / image / similaires, à chaud | 16 / 30 / 3 ms | 27 / 40 / 7 ms | 26 / 39 / 7 ms | 27 / 34-47 / 7 ms | | |
| **Export 10 clips** | ≈ 10 s | 11,4 s | 6,2 s | 6,0 s | 10 % → 11 % → 12 % | 27 % → 72 % → 76 % |
| Timelines (4 formats) | — | 0,2 s | 0,2 s | 0,2 s | | |
| **Upscale x2** (120 images 1080p) | 110 s | 70,6 s | 71,0 s | **37,3 s** | 98 % → 98 % → 93 % + Neural Engine | |
| **RIFE x2** | 26,1 s | 17,8 s | 17,6 s | **12,9 s** | 92 % → 91 % → 88 % | |
| **Profondeur** (120 images, PNG 16 bits) | ≈ 36 s | 36,2 s | 33,0 s | **21,3 s** | 64 % → 70 % → 79 % | |
| **Pipeline entier** | | | | | 47 % → 63 % → 62 % ⁵ | 46 % → 44 % → 39 % |

² +3 s : le décodage d'indexation enregistre maintenant aussi la copie
240p qui sert aux proxys (voir §3) ; il économise 130 s plus loin.
³ L'encodage VideoToolbox de la copie 240p a lieu pendant le décodage
d'indexation (compté dans la ligne 1) ; la passe proxy elle-même est un
réencodage x264 de petites images, plus rapide sur CPU (mesure en §3).

⁴ Tour 3 : seuls les proxys du dernier épisode restent après la file ;
ceux des deux premiers sont faits pendant l'indexation du suivant.
⁵ `ioreg` ne mesure que le GPU : au tour 3 une partie du travail (la moitié
de l'upscale) tourne sur le Neural Engine, invisible dans ce chiffre, et le
décodage sur le moteur vidéo, invisible aussi.

Mémoire : pic GPU (allocateur Metal) 3 808 Mo (avant 3 640 ; tour 3 :
4 224), pic RAM de l'arbre de processus du sidecar 4 709 Mo (avant 4 731 ;
tour 3 : 5 951, avec les 8 sessions de décodage et les processus Core ML).
**Aucun swap** pendant les trois runs officiels (0 page écrite, compteur
`vm_stat`).

## 3. Ce qui est passé sur l'accélérateur, et ce qui reste sur CPU (chantier 2)

État de fin de chantier 2 ; le tour 3 a déplacé le décodage sur le moteur
vidéo et les modèles sur Core ML / le Neural Engine (§7).

**Passé sur l'accélérateur / le moteur vidéo, ou supprimé :**

| Quoi | Avant | Après | Gain mesuré |
|---|---|---|---|
| Source des proxys | 2ᵉ décodage logiciel complet de chaque épisode | une 3ᵉ branche (15 i/s, 240p) du décodage d'indexation, déjà fait, encodée par **VideoToolbox** à 1,5 Mb/s | 2ᵉ décodage supprimé ; coût +0,65 s par 4 min de vidéo (+3 s sur la saison) |
| Encodage des proxys | VideoToolbox bridé par `-realtime 1` (cadence « capture en direct ») | depuis la copie 240p : libx264 *superfast* (voir ci-dessous) ; depuis la source (repli) : VideoToolbox sans bridage | saison : 148,6 → 18,9 s (×7,9) |
| Prétraitement SigLIP 2 (redimensionnement, patchs) | CPU (processeur torchvision) | **MPS** (même processeur, `device=mps`) | 96 images : 1,07 → 1,02 s ; cosinus ≥ 0,9994 |
| Pré-roll de 10 s décodé et jeté avant chaque clip exporté | décodé | supprimé (le seek d'entrée de ffmpeg est exact) | décodage −41 %, mêmes images |

**Reste sur CPU, avec la mesure qui le justifie :**

| Quoi | Pourquoi (mesure) |
|---|---|
| Décodage H.264 / HEVC 1080p | VideoToolbox via ffmpeg décode image par image (latence) : 4 min HEVC 10 bits 9,5 s en logiciel contre 44,5 s VideoToolbox ; H.264 5,5 s contre 37,6 s (tour 1) |
| Réencodage des proxys à partir de la copie 240p | à 240p, VideoToolbox coûte par image : 2 passes en parallèle sur 4 min de copie = 7,2 s VideoToolbox, **1,8 s libx264 superfast**, qualité 43,1 dB contre 41,2 dB (référence sans perte), même taille ; le CPU est libre à ce moment-là |
| Redimensionnements 160×90 (détecteur) et 512×288 (modèles) | faits par ffmpeg dans le graphe du décodage : ~10 % du décodage (6,0 s contre 5,4 s de décodage pur sur 4 min) ; les faire sur MPS imposerait de sortir les images 1080p (3 Mo chacune, ~1000/s) |
| Letterbox 448 px du wd-tagger | essayé sur MPS : plus lent (16,4 → 15,5 img/s) et jeux de tags identiques sur 88,5 % des images seulement ; le tagger est limité par le calcul GPU |
| Analyse de mouvement Search+ (OpenCV LK + RANSAC) | un thread suffit : décodage + mouvement = 6,1 s sur 4 min, comme le décodage seul ; 2, 4 ou 6 processus : 6,1 s aussi (résultats identiques) |
| Stabilisation temporelle de la profondeur | le code GPU (écrit pour CUDA) tourne sur MPS mais 23× plus lentement : 174 s contre 7,8 s (PSNR 61 dB, donc juste) |
| Écriture des PNG 16 bits de profondeur | compression zlib (pas d'encodeur PNG GPU accessible) ; passée sur 4 threads : 4,5 → 1,2 s |
| Vignettes JPEG | pas d'équivalent nvjpeg exposé à PyTorch sur Mac ; 2,4 s pour 1045 scènes |
| Exports | décodage logiciel (même raison que l'indexation) + encodage VideoToolbox ; 3 exports à la fois pour que l'un décode pendant que l'autre encode |

## 4. Optimisations gardées et rejetées (chantier 2 ; tour 3 : §7)

**Gardées** (gain mesuré isolément sur le M1 Pro, puis en run complet) :

| Changement | Mesure |
|---|---|
| Proxys VideoToolbox sans `-realtime 1` | 4 min : 8,3 → 3,6 s ; saison (run intermédiaire, avec la copie 240p) : 148,6 → 61,1 s |
| Copie 240p des proxys enregistrée pendant le décodage d'indexation | 4 min : passe proxy 5,1 → 3,6 s pour +0,65 s de décodage ; épisode HEVC : décodage +4,5 s, passe proxy ≈ −16 s |
| Proxys réencodés en libx264 superfast depuis la copie 240p | 2 passes parallèles sur 4 min : 7,2 → 1,8 s ; saison : 61,1 → 18,9 s ; extrait : 80/80 proxys à ≤ 60 ms de la durée de leur scène |
| Exports derush 3 à la fois (macOS) | 10 clips : 8,0 s (1) → 5,1 s (2) → 4,65 s (3) ; run complet : 11,4 → 6,2 s |
| Plus de pré-roll de 10 s avant chaque clip exporté (macOS) | 12 clips H.264 + HEVC : images identiques (framemd5), décodage −41 % |
| Prétraitement SigLIP 2 sur MPS | 1,07 → 1,02 s / 96 images |
| PNG de profondeur sur 4 threads | 4,5 → 1,2 s / 120 images ; run complet : profondeur 36,2 → 33,0 s |

**Rejetées** :

| Essai | Résultat |
|---|---|
| Letterbox du wd-tagger sur MPS | 16,4 → 15,5 img/s, tags identiques à 88,5 % |
| `-threads 32` pour le décodage logiciel | seul : H.264 5,4 → 4,1 s, HEVC 9,5 → 6,4 s (+200 Mo) ; **dans le pipeline : 10,8 → 14,6 s** sur l'extrait (le décodeur affame le thread qui nourrit le GPU) ; avec ffmpeg en priorité basse (`nice 10`) : 12,8 s, toujours pire |
| Stabilisation temporelle de la profondeur sur MPS | 7,8 → 174 s |
| Décodage VideoToolbox de la copie 240p | 3,6 → 7,1–9,3 s |
| Copie 240p à 4 Mb/s au lieu de 1,5 | aucun gain de vitesse, fichier 2,3× plus gros |
| Profondeur par lots de 2 ou 4 images | −5 % seulement (21,0 contre 22,1 s de modèle) |
| Analyse de mouvement sur 2-6 processus | aucun gain (6,1 s dans tous les cas) |

## 5. Qualité et fiabilité (chantier 2 ; tour 3 : §7.3)

Comparaison des sorties avant/après (`bench_pipeline.py --quality`) :

| Épisode | Coupes identiques à l'image près | Embeddings cosinus min | Tags top-5 | Même classe de mouvement |
|---|---:|---:|---:|---:|
| DBS 01 | 418 / 418 | 0,99987 | 100 % | 100 % |
| DBS 02 | 366 / 366 | 0,99970 | 100 % | 100 % |
| Black Clover (HEVC) | 261 / 261 | 0,99993 | 100 % | 100 % |

- **Finition** : upscale et RIFE identiques au bit près (PSNR infini),
  profondeur 99 dB.
- **Proxys de survol** : 1045 / 1045 produits avant comme après. Contre une
  référence 240p sans perte : 43,1 dB (après, x264) contre 41,6 dB (avant,
  VideoToolbox depuis la source) — un peu meilleurs. Les images retenues à
  15 i/s peuvent être décalées d'une image source par rapport à avant
  (filtre `fps` au lieu de `-r 15`), sans effet visible.
- **Exports** : mêmes images (vérifié par framemd5 sur 12 clips), encodeur
  et réglages inchangés.
- **Mémoire / swap** : 0 page swappée pendant les deux runs ; pics mémoire
  inchangés à ±170 Mo. La copie 240p occupe ~270 Mo de disque par épisode
  le temps de la file, puis est effacée (aussi après un échec ou une
  annulation).
- **Fiabilité** : la copie 240p est écrite en `.part` et gardée seulement
  si ffmpeg finit proprement ; sa clé inclut taille et date du fichier (un
  épisode remplacé ne réutilise jamais l'ancienne copie) ; si elle manque,
  la passe proxy décode la source comme avant.
- **Test matériel `tests/hw`** : vert sur le Mac (6/6) — tagger MPS sur
  plusieurs lots, SigLIP cosinus 0,99999 contre une référence enregistrée
  **avec le code d'avant le chantier**, API jamais gelée plus de 0,5 s
  pendant le chargement des modèles, 0 swap, Beerus 14/107 scènes. **Bug de
  test trouvé** : le contrôle « coupe sur la première image du plan »
  échouait aussi avec le code d'avant (mêmes coupes) : 9 coupes sur 79
  n'étaient pas sur l'image au plus grand saut d'histogramme, toutes dans
  des transitions où chaque image saute (6 dans le montage stroboscopique
  du générique, une après un flash blanc de 2 images, un fondu enchaîné,
  un fondu au noir à 0,21 contre 0,20). Le test ne juge plus que les coupes
  nettes (un saut dominant) : 45 / 79 ici, **toutes exactes**.
- Tests : `uv run pytest -m "not slow"` vert sur le VPS (170 passés) et sur
  le Mac (170 passés) ; `tsc`, vitest (12) et build de l'app verts sur le
  VPS. Nouveaux tests : `tests/test_pipeline_perf_mac.py`.

## 6. Ce qui reste lent, et les limites (après le tour 3)

- **Le décodage** est presque toute l'indexation (92 s sur 102 s pour la
  saison) et il est au plafond du moteur vidéo du M1 Pro (~1 040 images
  1080p/s en 8 sessions VideoToolbox ; 12 sessions : +4 %). En fin de
  chantier 2 on concluait que VideoToolbox était 4 à 7× plus lent que le
  logiciel : c'est vrai pour **une** session, faux pour huit (§7.1).
- **Le wd-tagger** (ViT-B/16 à 448 px) fait ~24 images/s en Core ML GPU ;
  avec la sous-segmentation (profil « Accurate », non mesuré par ce bench)
  il redeviendrait la limite, et ce profil garde la passe séquentielle.
- **La finition** est limitée par le calcul : upscale GPU 94 % + Neural
  Engine plein, RIFE et profondeur ~80-90 % GPU.
- **La première recherche** charge SigLIP 2 (3,6 s, dont ~1,1 s de
  chargement du modèle ; le reste est l'import de PyTorch/transformers).
- **Première utilisation** : conversions Core ML (§7.4).
- Mesures sur un seul Mac, un run complet par colonne ; écart entre runs
  complets du tour 3 : 3 min 03 à 3 min 05 selon le code.
- Les changements d'export (3 clips à la fois, pas de pré-roll), le
  décodage par morceaux, Core ML et les proxys anticipés sont limités à
  macOS / MPS : non mesurés ailleurs.

## 7. Tour 3 : ce qui a changé, et pourquoi c'est plus rapide

En fin de chantier 2, le Mac était limité à deux endroits : **le CPU**
pendant l'indexation (le décodeur logiciel prenait l'essentiel des 10 cœurs,
le reste du pipeline attendait), et **le GPU seul** pendant la finition
(98 % occupé, Neural Engine inutilisé). Le tour 3 s'attaque aux deux.

### 7.1 Indexation : le moteur vidéo au lieu du CPU

**Découverte clé :** le décodage VideoToolbox de ffmpeg n'est lent que *par
session* (une image à la fois, en attendant la précédente). Huit sessions
en parallèle décodent 4 min de 1080p H.264 en 5,6 s avec **11 s de CPU au
lieu de 36 s** en logiciel, et 4 min de HEVC 10 bits en 5,7 s au lieu de
9,6 s (11,5 s de CPU au lieu de 56). Les images décodées sont **identiques
au bit près** à celles du décodeur logiciel (vérifié sur les épisodes H.264
et HEVC).

D'où le **décodage par morceaux** (`backend/indexing/chunked.py`) : chaque
épisode est coupé en morceaux de 60 s, décodés par 8 sessions VideoToolbox
en parallèle, et les résultats sont recollés **exactement** comme la passe
séquentielle les aurait produits (voir le module : une coupe qui ne peut
être annulée par rien de ce qui précède sert de point d'ancrage dans chaque
morceau ; le recolleur rejoue les règles séquentielles sur les débuts de
morceaux et les frontières ; les statistiques de mouvement sont concaténées
avec la paire qui relie deux morceaux). Vérifié : mêmes coupes, mêmes images
médianes au pixel près, mêmes statistiques de mouvement que la passe
séquentielle sur DBS 01 et Black Clover, et sur des clips synthétiques en
CI.

Le CPU libéré a permis le reste :

| Changement (tour 3) | Mesure | Gardé |
|---|---|---|
| Décodage par morceaux, 8 sessions VideoToolbox | décodage seul : DBS 01 36,8 → 31,5 s, Black Clover 45,3 → 30,3 s ; Black Clover dans le pipeline 47 → 32,6 s | oui |
| Une seule thread de décodeur par session VideoToolbox | 8 sessions : 1,86 → 1,32 Go de RAM, même vitesse (Black Clover 30,3 → 29,0 s) | oui |
| wd-tagger compilé par Core ML sur le GPU (processus à part) | 17,0 → 24,3 img/s ; top-5 identique 64/64, jeux de tags 98,4 % | oui |
| SigLIP pendant que le tagger tourne dans son processus ; chargement des deux modèles en parallèle ; letterbox sur 4 threads | DBS 01 : traîne d'inférence après le décodage 9,9 → 5,7 s, puis ~0,4 s sur les épisodes suivants | oui |
| Proxys d'un épisode faits pendant l'indexation du suivant | proxys en fin de file 18,9 → 6,5 s | oui |
| Métriques Search+ des scènes sur 4 threads | 0,88 → 0,68 s par épisode, lignes identiques | oui |
| `-threads 32` pour le décodage logiciel (épisode entier) | stage 44,3 → 41,2 s, rendu caduc par VideoToolbox | non (dépassé) |
| 1 ou 2 décodeurs logiciels en plus des 8 sessions | décodage 31,0 → 31,6 / 32,4 s et traîne d'inférence 1 → 12 s | non |
| Les mêmes, bridés à 2 ou 4 threads | décodage 29,0 → 28,0-29,8 s, traîne d'inférence 1,6 → 9,6-11,8 s alors que le CPU n'est qu'à 55 % (macOS relègue alors le worker du tagger) | non |
| wd-tagger sur le Neural Engine | lot de 8 : la compilation ne finit pas ; lot de 1 : 22,7 img/s mais jeux de tags identiques à 92 % seulement, et une répartition GPU/ANE rendrait les tags dépendants du hasard de l'ordonnancement | non |
| Proxys de la dernière vidéo en 3 plages parallèles | segments mal numérotés (124 proxys sur 366 faux, seek + `-r 15`) ; même avec `-copyts` | non |
| SigLIP chargé plus vite | 1,1 s de chargement réel quand le disque est en cache : rien à gagner | non |
| SigLIP préchargé au démarrage du sidecar (comme sur CUDA, `fd20ee8` côté NVIDIA) | aucun effet dans le bench (sa première requête arrive dès le démarrage : 2,25 s à froid dans les deux cas) ; supprime les 3,6 s d'attente de la première recherche en usage réel | oui |
| Charger les programmes Core ML déjà compilés (`.mlmodelc`) | chargement 0,3-0,8 s → 0,01-0,04 s, soit ~0,5 s par worker : pas la peine | non |

Résultat sur la saison : **160 → 101 s**. Le décodage (92 s pour 101 451
images, ~1 100 img/s) est maintenant **au plafond du moteur vidéo** du M1
Pro : 8 sessions plafonnent à ~1 040 img/s en H.264 1080p, 12 sessions ne
gagnent que 4 %, et ajouter du décodage logiciel ralentit le reste.

### 7.2 Finition : le Neural Engine et Core ML

Deux découvertes :

1. **Le Neural Engine fait jeu égal avec le GPU sur Real-ESRGAN** (tuile
   576×516 : 64 ms ; image 1080p en 8 tuiles : 514 ms contre 540 ms pour le
   GPU en MPS fp16 ; PSNR 65,9 dB contre fp32). Appelé depuis le même
   processus, Core ML garde le GIL et le GPU s'effondre (62 → 10 tuiles/s) ;
   depuis un processus séparé, les deux s'additionnent (tuiles 512 : GPU
   15,0/s seul, 14,4 + 18,0 ANE ensemble).
2. **Core ML exploite mieux le GPU que PyTorch MPS** sur certains réseaux :
   RIFE 187 → 91 ms par paire en fp32 (sortie identique, 124,7 dB),
   Depth Anything V2 179 → 123 ms par image en fp32 (121 dB).

| Changement (tour 3) | Mesure | Gardé |
|---|---|---|
| Upscale : 2 processus Neural Engine + MPS, images réparties, ordre conservé | 71,5 → 40,1 s (1 processus ANE) → 37,5 s (2) ; PSNR 55,2 dB contre la sortie de référence | oui |
| RIFE par Core ML GPU fp32 | 17,5 → 12,5 s, sortie identique | oui |
| Depth Anything par Core ML GPU fp32 | 33 → 28,6 s, 102 dB | oui |
| Médiane exacte par tri + stabilisation sur 6 threads | profondeur 28,6 → 22,9 s, sortie identique (comparée à `np.nanmedian`) | oui |
| Profondeur : image suivante déjà envoyée au worker | 23,2 → 22,1 s, identique | oui |
| Lecture / écriture des images en threads (déjà le cas sur CUDA) | ffmpeg convertit toujours en YUV : fichiers identiques | oui |
| Fermeture explicite du worker du modèle précédent | un run complet avait swappé 15 158 pages (pic 7,0 Go) : 0 depuis | oui (correctif) |
| Moitié GPU de l'upscale en Core ML au lieu de MPS | 39,5 contre 40,1 s, et 75 s de conversion par résolution | non |
| 3 processus ANE pour l'upscale | 38,8 s contre 37,5 avec 2 | non |
| RIFE sur le Neural Engine | erreur d'inférence de l'ANE | non |
| RIFE en Core ML fp16 | −0,9 s mais 49 dB au lieu de 56 dB | non |
| Depth Anything en Core ML fp16 | 102 ms au lieu de 123, mais 41,7 dB sur le clip (trop près du seuil de 40) | non |
| Depth Anything sur le Neural Engine | 365 ms et 35,5 dB | non |
| Depth Anything en précision mixte (fp16 sauf softmax/normalisations ; backbone fp16 / tête fp32 ; l'inverse) | 95 ms / 45,2 dB ; 124 ms / 47 dB ; 101 ms / 44,9 dB : la vitesse vient de la tête en fp16, et l'erreur aussi | non |
| Stabilisation temporelle sur MPS (chantier 2) | 23× plus lent que sur CPU | non |
| Export : 4 clips en parallèle au lieu de 3 | isolé 3,0 → 2,7 s, mais 7,4 s contre 6,1-6,5 s en run complet | non |
| Export : décodage VideoToolbox de chaque clip | 10 clips du bench : 5,45 → 5,94 s | non |

Résultat sur la finition : **122 → 72 s**.

### 7.3 Qualité (run final du tour 3 contre la référence d'avant chantier 2)

| Épisode | Coupes identiques à l'image près | Embeddings cos min | Tags top-5 | Même classe de mouvement |
|---|---:|---:|---:|---:|
| DBS 01 | 418 / 418 | 0,99987 | 99,8 % | 100 % |
| DBS 02 | 366 / 366 | 0,99970 | 99,95 % | 100 % |
| Black Clover (HEVC) | 261 / 261 | 0,99993 | 99,9 % | 100 % |

Finition : upscale 55,3 dB, RIFE 56,4 dB, profondeur 102 dB (seuil 40 dB).
Proxys : 1045 / 1045 produits, chacun de la durée de sa scène à ≤ 140 ms
près (sauf la dernière scène de Black Clover, dont le flux vidéo s'arrête
6 s avant la fin du conteneur, comme avant). Le défaut corrigé côté NVIDIA
(`2ba6bd8` : proxys décalés d'une scène après une coupe arrondie) n'apparaît
pas ici : 1044 / 1044 durées conformes ; la protection
`-segment_time_delta` qu'ajoute ce commit s'appliquera au Mac à la fusion.
Tests : `pytest -m "not slow"` vert sur le VPS (189) et le Mac (189), test
matériel `tests/hw` vert sur le Mac (6/6, le décodage par morceaux y est
vérifié : 79/79 coupes à la même image), `tsc`, vitest (12) et build verts
sur le VPS. Nouveaux tests : `tests/test_chunked_decode.py`,
`tests/test_coreml_finish.py`.

### 7.4 Fiabilité et coût de la première utilisation

- Chaque modèle Core ML tourne dans **son propre processus** ; s'il échoue
  (conversion impossible, erreur du moteur), le travail continue sur MPS.
  Sans `coremltools` (nouvelle dépendance de l'extra `mps`, BSD-3), ou avec
  `AMV_COREML=0` / `AMV_ANE=0` / `AMV_CHUNKED=0`, tout revient au chemin du
  chantier 2.
- Les programmes Core ML sont **convertis une fois par taille d'image** et
  mis en cache à côté des modèles. Première fois : tagger ~10 s (pendant ce
  temps MPS travaille, mais un épisode indexé pendant la conversion a pris
  53,6 s au lieu de 41,9 s, avec un pic de RAM à 8 Go), RIFE ~8 s,
  profondeur ~3 s, upscale ANE ~5 s. Un job n'attend jamais une conversion.
- Le décodage par morceaux échoue proprement (exception → passe
  séquentielle) si un morceau se termine trop tôt ; la sous-segmentation
  (profil « Accurate ») garde la passe séquentielle.
- Mémoire : pic de 6,0 Go pour l'arbre de processus (8 sessions de
  décodage, processus Core ML) ; 0 swap sur les runs officiels, mais un run
  intermédiaire a swappé (corrigé, §7.2) : sur un Mac de 8 Go, il faudrait
  réduire le nombre de sessions.

## 8. RTX 3090 contre M1 Pro : pourquoi le Mac va plus vite

Chiffres de fin de chantier 2 (point de départ du tour 3) et du tour 3 sur
le Mac. Le tour 3 NVIDIA tourne en parallèle dans une autre session : son
chiffre final est dans `docs/reports/chantier-2-pipeline-nvidia.md`. Au
moment de ces mesures, le dernier run complet publié de la 3090 est celui de
fin de chantier 2.

| Pipeline complet | RTX 3090 (VM 6 vCPU), fin chantier 2 | M1 Pro, fin chantier 2 | **M1 Pro, tour 3** |
|---|---:|---:|---:|
| **Total** | 5 min 49 (349 s) | 4 min 52 (292 s) | **3 min 03 (183 s)** |
| Saison (indexation) | 269,1 s | 160,2 s | 101,2 s |
| Recherches | 9,1 s | 4,1 s | 4,1 s |
| Export | 12,4 s | 6,4 s | 6,1 s |
| Finition | 58,8 s | 121,6 s | 71,4 s |

Ce n'est pas un duel GPU contre GPU, et c'est pour ça que le Mac gagne :

- **L'indexation est un problème de décodage vidéo et de CPU, pas de calcul
  GPU.** Une heure de vidéo, c'est ~100 000 images 1080p à décoder ; les
  modèles n'en voient que ~1 000 (une médiane par scène). La 3090 n'a
  qu'**un** moteur NVDEC (rapport NVIDIA : « le décodage reste au plancher
  de l'unique moteur NVDEC ») et tout ce qui n'est pas sur la carte
  (orchestration Python, pipes ffmpeg, base de données) passe par **6 vCPU
  partagés** d'une VM. Le M1 Pro a 8 cœurs performance rapides, un moteur
  vidéo qui débite ~1 040 images 1080p/s quand on l'attaque en 8 sessions,
  et une mémoire unifiée : une image décodée n'a pas à traverser un bus PCIe
  pour atteindre le CPU ou le GPU.
- **Les recherches et l'export sont des latences CPU** (Python, lancement
  de ffmpeg, SQLite) : les cœurs du M1 Pro sont plus rapides que des vCPU
  de VM.
- **Là où le calcul GPU domine, la 3090 gagne déjà** : finition 58,8 s
  contre 121,6 s en fin de chantier 2 (upscale ~14,7 s sur 3090 contre
  71 s), soit 2× plus vite. Après le tour 3, le Mac réduit l'écart à 1,2×
  (71,4 s) en ajoutant le Neural Engine et Core ML, mais il est au bout :
  GPU 94 % + ANE pleins pendant l'upscale. En calcul brut, la 3090 a ~7× la
  puissance du GPU du M1 Pro (35,6 contre ~5,2 TFLOPS fp32, bien plus en
  fp16 tenseur) : bien alimentée, elle doit faire la finition 3 à 5× plus
  vite que le Mac.

**Ce qu'il faudrait pour que la 3090 écrase le M1 Pro sur le pipeline
complet** : ne plus décoder sur un seul NVDEC limité par une VM à 6 vCPU
(plusieurs passes NVDEC en vol, décodage logiciel en complément sur une
machine qui a des cœurs, ou des épisodes en parallèle) et sortir de la VM
le travail CPU. C'est l'objet du tour 3 NVIDIA (commits « two single passes
in flight », « NVDEC straight into CUDA tensors »). Côté Mac, la marge est
désormais faible : le décodage est au plafond du moteur vidéo et la
finition au plafond GPU + Neural Engine.

## Annexe — sortie brute du bench

Fichiers JSON/Markdown des runs complets dans
`docs/reports/chantier-2-apple-silicon/` : `c2-avant`, `c2-apres-v1` (run
intermédiaire du chantier 2), `c2-apres`, `t3-final-v1` (tour 3, export à
4 clips en parallèle), `t3-final-v2` (tour 3, sans préchargement SigLIP),
`t3-final` (tour 3, final). Table produite par
`tests/bench_pipeline.py --compare` :

| | avant chantier 2 | après chantier 2 | **tour 3** |
|---|---:|---:|---:|
| **pipeline total** | 7 min 07 | 4 min 52 | 3 min 03 (×2.3) |
| season | 4 min 47 | 2 min 40 | 101.2 s (×2.8) |
| search | 4.20 s | 4.13 s | 4.13 s (×1.0) |
| export | 11.6 s | 6.42 s | 6.14 s (×1.9) |
| finish | 2 min 05 | 2 min 02 | 71.4 s (×1.7) |
| — probe | 0.09 s | 0.09 s | 0.13 s (×0.7) |
| — decode_cuts_motion | 2 min 15 | 2 min 18 | 91.8 s (×1.5) |
| — median_frames | 0.00 s | 0.00 s | 0.00 s |
| — thumbnails_db | 2.46 s | 2.44 s | 1.88 s (×1.3) |
| — tags | 0.04 s | 0.04 s | 0.04 s (×1.0) |
| — embeddings | 0.05 s | 0.06 s | 0.05 s (×1.0) |
| — finalize | 0.13 s | 0.13 s | 0.74 s (×0.2) |
| — hover_proxies | 2 min 29 | 18.9 s | 6.52 s (×22.8) |
| finish upscale_x2 | 70.6 s | 71.0 s | 37.3 s (×1.9) |
| finish rife_x2 | 17.8 s | 17.6 s | 12.9 s (×1.4) |
| finish depth | 36.2 s | 33.0 s | 21.3 s (×1.7) |
| first search (cold) | 3.72 s | 3.64 s | 3.60 s (×1.0) |
| GPU util, season | 37 % | 59 % | 70 % |
| GPU util — probe | 16 % | – | – |
| GPU util — decode_cuts_motion | 69 % | 66 % | 76 % |
| GPU util — thumbnails_db | 50 % | 33 % | 45 % |
| GPU util — finalize | 0 % | – | 15 % |
| GPU util — hover_proxies | 8 % | 8 % | 7 % |
| RAM peak (MB) | 4731 | 4709 | 5951 |
| GPU mem peak (MB) | 3640 | 3808 | 4224 |
| swap-outs (pages) | 0 | 0 | 0 |

