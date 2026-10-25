# Chantier 2 — le pipeline complet sur NVIDIA (RTX 3090)

Branche `jarvis/v04-perf-cuda`, machine de test RTX 3090 (VM Proxmox : RTX 3090
24 Go, 6 vCPU d'un Ryzen 9 5900X partagé, 21 Go de RAM vus par la VM, Debian
12, torch 2.11 + cu126, ffmpeg 5.1 Debian avec NVDEC / `scale_cuda` / NVENC).
Ce rapport ne concerne que la machine NVIDIA ; le Mac a sa propre session et
son propre rapport.

## Tour 3 (25 octobre, 02:25 → 10:00 CET) : aller beaucoup plus vite

### Le tableau principal

Pipeline complet (`tests/bench_pipeline.py`, mêmes 3 épisodes, mêmes
recherches, exports et finition que plus bas) :

| | v0.3 (estimé) | avant chantier 2 | après chantier 2 | **tour 3** |
|---|---:|---:|---:|---:|
| **pipeline complet** | ≈ 45 min 30 | 8 min 12 | 5 min 49 | **4 min 32 – 4 min 43** |
| saison (3 épisodes + proxys) | ≈ 39 min 50 | 336,6 s | 269,1 s | **193,7 – 203,9 s** (158,6 s au mieux) |
| recherches (18) | ≈ 45 s | 11,8 s | 9,1 s | **7,6 – 7,7 s** |
| export (10 clips + timelines) | ≈ 59 s | 29,6 s | 12,4 s | **11,5 – 12,9 s** |
| finition (upscale + RIFE + profondeur) | 237,6 s | 114,0 s | 58,8 s | **57,9 – 60,3 s** |

**Attention au bruit de mesure.** La VM NVIDIA partage son hôte Proxmox
(Ryzen 9 5900X) avec d'autres VM, dont celle où tournent les tests et une
autre session de travail : le même code a donné une saison de 197,5 s à
04:00 et de 266,3 s à 05:00 (charge de l'hôte). Les chiffres « avant / après
chantier 2 » ci-dessus sont des runs uniques pris à d'autres moments. Pour
une comparaison honnête, le tour 3 se termine par des runs **alternés** :
code final (`638789f`) et code de fin de chantier 2 (`a42f217`), mêmes
3 épisodes, même hôte, deux paires :

| run alterné | référence (fin du chantier 2) | **tour 3** | écart |
|---|---:|---:|---:|
| pipeline complet, paire 1 | 329,9 s (5 min 30) | **272,0 s (4 min 32)** | −18 % |
| pipeline complet, paire 2 | 382,9 s (6 min 23) | **283,4 s (4 min 43)** | −26 % |
| saison, paires 1 / 2 | 249,3 / 297,4 s | **193,7 / 203,9 s** | −22 / −31 % |
| — décodage + coupes + mouvement | 175,4 / 214,1 s (NVDEC 54 / 46 %) | **163,6 / 169,8 s (NVDEC 65 / 61 %)** | |
| — tags | 21,3 / 23,6 s | **10,7 / 11,0 s** | ×2,1 |
| — embeddings | 23,4 / 25,2 s | **6,2 / 8,5 s** | ×3,3 |
| — proxys (queue de file) | 25,1 / 30,3 s | **7,3 / 7,5 s** | ×3,7 |
| recherches | 7,7 / 8,5 s | 7,6 / 7,7 s | = |
| export | 12,6 / 11,8 s | 12,9 / 11,5 s | = |
| finition | 60,4 / 65,2 s | 57,9 / 60,3 s | ≈ |
| utilisation GPU moyenne de la saison | 20 / 17 % | 22 / 20 % | |
| pic VRAM / RAM du sidecar | 2,2 Go / 3,7 Go | 5,5 Go / 4,4 Go | |

Une première série alternée une heure plus tôt (code `0644c85`, sans le
dernier réglage des proxys) donnait 292,9 / 370,6 s pour la référence contre
259,7 / 277,6 s. Sur les 8 runs, la médiane de la référence (≈ 350 s,
saison ≈ 269 s) retombe exactement sur les chiffres de fin de chantier 2
(349,4 / 269,1 s) : la comparaison tient malgré le bruit.

Meilleure saison isolée mesurée au cours du tour (hôte plus calme) :
**158,6 s** (code final, qualité identique).

### Ce qui limitait vraiment la 3090 (mesuré)

1. **Le décodeur matériel n'est pas le plafond qu'on croyait.** La 3090 n'a
   qu'un moteur NVDEC : ~740 im/s en H.264 1080p (P2, horloge vidéo
   1740 MHz, décodeur à 100 %), ~1450 im/s en HEVC 10 bits ; deux décodages
   simultanés prennent exactement deux fois plus de temps. Plancher pour les
   101 451 images de la saison : ~110 s. Les décodages réels prenaient
   pourtant 55 à 85 s par épisode, NVDEC à 50-65 %.
2. **Le vrai goulot est le CPU de la VM** (6 vCPU partagés). Expérience
   décisive, passe unique sur l'extrait : seule 12,4 s ; avec une charge GPU
   dans un autre processus 13,6 s ; avec 4 processus qui occupent le CPU
   **32,0 s (×2,6)**.
3. **Ce qui consommait ce CPU** : la chaîne ffmpeg (décodage NVDEC puis, pour
   *chaque image*, 3 `scale_cuda`, 2 `overlay_cuda`, un téléchargement et une
   écriture dans un pipe) coûtait 8 à 9 secondes CPU par 4 min de vidéo,
   alors que le décodage seul en coûte 3 : environ 2 ms de CPU par image.
   Dans le sidecar (profil py-spy d'une saison) : lecture du pipe 37 % des
   échantillons actifs, logique de coupe 13 %, suivi de mouvement GPU 14 %
   (dispatch Python), palettes 9 %, tags/embeddings/base 11,5 %, et un fil de
   lecture de stderr qui lisait octet par octet 5 %.

### La réponse : NVDEC directement dans des tenseurs CUDA, au bit près

`backend/indexing/nvc_pass.py` : PyNvVideoCodec (licence MIT, dépendance
optionnelle des extras CUDA) décode dans la mémoire du GPU, et torch fait la
réduction par lots de 32 images. Pour que **les résultats restent
identiques**, la réduction reproduit au bit près le `scale_cuda` d'ffmpeg 5.1
(lu dans ses sources) : bicubique par défaut (A = 0), texels normalisés
k/255 (k/65535 en 10 bits) arrondis en float32, produit puis
multiplications-additions fusionnées (`torch.addcmul` est compilé avec la
même FMA qu'nvcc), troncature finale ; numérotation des images comme
`-fps_mode cfr` ; une copie `cuMemcpyDtoDAsync` par image hors du décodeur ;
réduction, téléchargement et analyse dans un second fil, sur un flux CUDA à
part ; copie 240p des proxys encodée par NVENC depuis le GPU. Les images
alimentent le même analyseur qu'avant (coupes, médianes, mouvement).

| vérification | résultat |
|---|---|
| plan gris 160×90 (coupes, mouvement) vs ffmpeg | identique sur 200/200 images |
| image 512×288 NV12 (médianes) vs ffmpeg | identique sur 40/40 images |
| extrait 4 min H.264 : coupes / échantillons de mouvement / médianes | 79/79, 1449/1449, 79/79 identiques |
| Black Clover entier (HEVC 10 bits) | 261 coupes, 8572 échantillons, 260/260 médianes identiques |
| saison complète vs fin de chantier 2 | coupes 1045/1045, embeddings ≥ 0,99994, tags top-5 98-99 %, mouvement 100 % |

Gain : extrait H.264 10,4 → 4,0-5,6 s CPU (8,8 → 8,2-8,6 s) ; Black Clover
entier **47,9 s / 67,2 s CPU → 23,4 s / 26,9 s CPU**.

### Les autres changements gardés

| changement | mesure |
|---|---|
| deux passes uniques en vol (ordonnanceur : le fichier dont l'index a besoin + le moins cher des 3 suivants) | NVDEC de 60 % à ~100 % ; A/B alterné : saison 266,3 → 236,3 s |
| lecture du pipe dans un tampon circulaire (`readinto`, Python par échantillon seulement) | sorties identiques (md5) ; Black Clover 42,4 → 37,7 s (chemin ffmpeg) |
| suivi de mouvement GPU en blocs de 1024 paires, sans synchronisation par itération | 37 % → ~14 % des échantillons du sidecar ; résultats identiques |
| palettes d'une vidéo en un lot GPU (init et réduction 48×27 inchangées) | palettes identiques 151/151 ; 0,64 → 0,12 s CPU |
| letterbox du tagger sur GPU | 2,2 → 1,2 s pour 316 images ; top-5 identique à 99,1 % |
| taille de lot GPU par défaut = recommandation Settings (24 Go → 64) | tags 22,5 → 12,8 s, embeddings 23,8 → 7,2 s sur la saison |
| proxys découpés dès qu'une vidéo est indexée (priorité basse), 3 tronçons parallèles dès que plus rien ne décode | queue de proxys 25-30 → 7,3-7,5 s (alterné) |
| copie 240p à 15 im/s, QP 20 | CPU d'ffmpeg ~6,3 → ~4,7 s par 4 min ; copie 42 → 21 Mo |
| stderr d'ffmpeg à nouveau tamponné | 5 % des échantillons du sidecar supprimés |
| profondeur : flux DIS calculés par 4 fils à l'avance | lissage 9,56 → 7,47 s, sortie identique au bit près |
| préchargement de SigLIP au démarrage (CUDA) | utile si la 1re recherche arrive > 7 s après le lancement ; aucun effet sur le bench (requête immédiate) |

### Essayé et rejeté

| idée | mesure |
|---|---|
| décoder dans des processus séparés (sans le GIL) | 201,8 s contre 187,5 s en fils (2 contextes CUDA, +1,5 Go de RAM) |
| 3 décodages en vol | 222 s : l'épisode 1 n'arrive qu'à +181 s |
| indexer dans l'ordre de fin des décodages | 184,8 s contre 161,3 s : les passes proxys s'empilent en fin de file |
| ffmpeg 8.1 (pipeline multi-fils) | images **non identiques** à ffmpeg 5.1 (md5) : coupes non garanties |
| PyAV 17 (FFmpeg 8 avec NVDEC) en zéro copie | `is_hw_owned` non supporté par sa build (errno 95) |
| copie proxy tout-intra + découpe sans réencodage | 3,4× plus de disque pour les proxys (51 contre 15 Mo par 4 min) |
| décodage logiciel en renfort de NVDEC | 206 im/s avec les 6 vCPU, qui manquent déjà |
| graphe CUDA pour le suivi de mouvement | estimations identiques mais 0,69 s contre 0,68 s : le coût n'était pas là |
| pipe de 1 Mo, `-avioflags direct`, deux sorties ffmpeg | pas de gain mesurable au-dessus du bruit |
| bords « reflect-101 » dans le suivi GPU | 93,7 % de classes identiques au lieu de 96,2 % |
| décodeur `h264_cuvid` pour l'indexation, `-hwaccel_flags unsafe_output` | pas de gain CPU |
| compression PNG plus rapide (profondeur) | le défaut d'OpenCV est déjà le plus rapide |
| export par saut direct (sans pré-roll) | 1,25-1,5 s contre 1,5-1,6 s par clip : dans le bruit |

### Bug préexistant trouvé et corrigé : les proxys de survol étaient décalés

Depuis la passe proxys groupée du chantier 2 (`8b12280`), `-force_key_frames`
place chaque image clé sur l'image à 15 im/s **la plus proche** de la coupe
(jusqu'à une demi-image avant), alors que le multiplexeur `segment` ne coupe
qu'aux images clés situées **après** l'instant demandé. Il sautait ces
coupes, fusionnait deux scènes, et **chaque proxy suivant était celui de la
scène précédente** (DBS 01 : 418 segments pour 419 scènes, décalage dès la
3e). Correctif : `-segment_time_delta 0.034` (la demi-image recommandée par
la documentation d'ffmpeg) ; les 419 segments correspondent désormais à leur
scène (durées vérifiées). Le chemin Mac (VideoToolbox) utilise la même passe
et bénéficie du correctif.

### Pourquoi la 3090 ne fait pas mieux que le M1 Pro sur l'indexation

Ce que disent les mesures, sans enjoliver :

- **Le Mac (fin du chantier 2) : pipeline 4 min 52, saison 2 min 40.** La
  3090 (tour 3) : **pipeline 4 min 32 – 4 min 43**, donc un peu plus
  rapide au total, mais **saison 3 min 14 – 3 min 24** (2 min 39 au mieux),
  donc encore plus lente que le M1 Pro sur l'indexation.
- **Là où la 3090 écrase le Mac**, c'est sur ce qui est vraiment du calcul
  GPU : tags 11 s et embeddings 6-8 s pour toute la saison, finition 58-60 s
  (upscale, RIFE, profondeur), recherches et exports ; la partie « hors
  saison » prend ~79 s sur la 3090 contre ~132 s sur le Mac.
- **L'indexation, elle, est un problème de décodage vidéo et de CPU, pas
  de calcul GPU.** Il faut décoder 101 451 images 1080p. Le M1 Pro les décode
  en logiciel sur 8 cœurs performance dédiés + 2 efficaces, en mémoire unifiée
  (aucune copie). La 3090 n'a qu'**un** moteur NVDEC (~740-830 im/s en
  H.264 1080p : ~110 s de plancher pour la saison, mesuré) et chaque image
  coûte encore du CPU (≈ 0,7-1 ms avec le nouveau chemin, contre 2 ms avant
  le tour 3) sur une **VM à 6 vCPU partagés** avec d'autres VM : une charge
  CPU voisine multiplie le temps de décodage par 2,6 (mesuré), et la même
  saison a varié de 161 à 266 s selon l'heure.
- **Ce qu'on peut attendre ailleurs** (estimation, pas mesuré) : sur un PC de
  bureau avec cette 3090 et 8 à 16 cœurs dédiés, le CPU n'est plus le goulot
  (~0,7-1 ms par image) et la saison tend vers le plancher NVDEC (~2 min,
  contre 2 min 40 pour le M1 Pro) ; une carte avec plusieurs moteurs NVDEC
  abaisserait encore ce plancher (non vérifié ici). Sur cette VM, la seule
  façon d'aller nettement plus loin serait de lui donner plus de cœurs réels.

### Commits du tour 3

`adf149e` (lecture du pipe en tampon circulaire), `6e4caaf` (suivi de
mouvement en gros blocs), `eac0ff4` (proxys en recouvrement, lots GPU de 64),
`fd20ee8` (préchargement SigLIP), `2ba6bd8` (**correctif des proxys
décalés**), `c601020` (flux DIS en parallèle), `974d2e8` (letterbox GPU du
tagger), `b0ca69d` (deux décodages en vol), `ff1a8b1` (palettes GPU),
`31224b2` et `0644c85` (**NVDEC → tenseurs CUDA**, H.264 puis HEVC 8/10 bits),
`638789f` (taille des passes proxys décidée au démarrage), et ce rapport.

### Qualité et fiabilité du tour 3

- Contre la fin du chantier 2 (`a42f217`) sur les 3 épisodes : coupes
  1045/1045 identiques, embeddings ≥ 0,99994, tags top-5 98,1-99,2 % (letterbox
  GPU et lots de 64), mouvement 100 % identique, finition : upscale et RIFE identiques au bit près (PSNR infini), profondeur identique (99 dB = égalité).
- Proxys : 0 segment manquant, chaque segment de la bonne durée.
- Tests (code final) : VPS 179 passés, `tsc`/`vitest` (12)/build OK ; NAS
  176 passés + les 3 échecs préexistants du chantier 2 (inchangés) ; test
  `hw` : 5 passés (SigLIP 1,00000 contre la référence, gel API max 1,2 s,
  0 swap) et le même échec préexistant d'alignement des coupes (87,2 %).
- Repli : sans PyNvVideoCodec, avec un autre codec ou en cas d'erreur, le
  chemin ffmpeg du chantier 2 est utilisé (`AMV_NVC_PASS=0` le force) ;
  `AMV_PARALLEL_DECODES=1` revient à un décodage à la fois.
- Mémoire : pic VRAM de la saison ~5,3 Go (2 décodages + modèles), RAM du
  sidecar ~4,5-5 Go.

---

*Ce qui suit est le rapport du chantier 2 (tours 1 et 2), conservé tel quel.*

## 1. Le tableau principal

Pipeline complet mesuré par `tests/bench_pipeline.py` : la « saison »
(3 épisodes réels indexés en une file : DBS Beerus 01 et 02 en H.264, Black
Clover S02E04 en HEVC 10 bits, 1h09 de vidéo, 101 451 images, avec
mouvement/intensité/palette Search+, images médianes, tags, embeddings,
vignettes et proxys de survol), puis 10 recherches texte + 3 recherches image
+ 5 « plans similaires » (la première à froid, sidecar redémarré), l'export
de 10 clips gardés + 4 timelines (FCP7, FCPXML, EDL, OTIO), et la finition
d'un clip de 5 s (upscale x2, RIFE x2, profondeur).

| | v0.3 (historique, estimé) | avant chantier 2 | **après** | gain vs avant |
|---|---:|---:|---:|---:|
| **Pipeline complet** | ≈ 45 min 30 | 8 min 12 (492,0 s) | **5 min 49 (349,4 s)** | **×1,41** |
| saison (indexation 3 épisodes + proxys) | ≈ 39 min 50 ¹ | 336,6 s | **269,1 s** | ×1,25 |
| recherches (18, la 1re à froid) | ≈ 45 s ² | 11,8 s | **9,1 s** | ×1,30 |
| export (10 clips + 4 timelines) | ≈ 59 s ³ | 29,6 s | **12,4 s** | ×2,39 |
| finition (upscale x2 + RIFE x2 + profondeur, 5 s) | 237,6 s | 114,0 s | **58,8 s** | ×1,94 |
| utilisation GPU moyenne pendant la saison | – | 6 % | **19 %** | |

¹ épisode 01 v0.3 (indexation 354 s + proxys 443 s) × 3. ² 1re requête 2,8 s
+ requêtes texte à 4,7 s + image/similaires. ³ 16,9 s par clip de 10 s en
libx264 (défaut v0.3), ramené aux 34,7 s de clips.

- **v0.3 (historique)** : pas rejouée (consigne). Chiffres du tour 1
  (`tests/bench_cuda.py`, commit `be96c09`, médiane de 2 runs, JSON de la
  machine dans `<données>/bench/before-*.json`), par étape. La
  saison v0.3 n'a jamais été mesurée sur 3 épisodes : la valeur est
  l'épisode 01 complet (indexation 354 s + proxys 443 s) multiplié par trois,
  donc une **estimation**. L'export v0.3 est extrapolé de son coût mesuré
  (libx264, 16,9 s pour un clip de 10 s) aux 34,7 s de clips du bench.
- **Avant chantier 2** : l'état de la branche au démarrage (`8b12280`, tour 1
  NVIDIA + Mac fusionnés, proxys en une passe, export H.264 matériel), un run
  complet, cache disque chaud.
- **Après** : la branche finale, un run complet, mêmes conditions.

## 2. Détail par étape

| étape | avant : temps | GPU / NVDEC / NVENC / CPU | **après : temps** | GPU / NVDEC / NVENC / CPU |
|---|---:|---|---:|---|
| décodage + coupes + mouvement (3 épisodes) | 177,7 s | 5 / 55 / 0 / 44 % | **189,9 s** ⁴ | 17 / 50 / 7 / 31 % |
| images médianes | 0,8 s | – | 0,7 s | – |
| vignettes + insertion SQLite | 1,9 s | 0 % GPU | 2,3 s | 0 % GPU |
| tags (wd-tagger fp16) | 23,3 s | 18 / – / – / 61 % | 22,5 s | 24 / – / – / 51 % |
| embeddings (SigLIP 2) | 16,3 s | 13 / – / – / 46 % | 23,8 s ⁴ | 25 / – / – / 43 % |
| **proxys de survol** | 115,6 s | 4 / 90 / 8 / 18 % | **28,2 s** | 25 / 57 / 31 / 40 % |
| **saison (total)** | 336,6 s | **6** / 63 / 3 / 36 % | **269,1 s** | **19** / 45 / 9 / 35 % |
| recherche : 1re requête à froid | 9,8 s | | 7,5 s | |
| recherche : texte (chaud, médiane) | 150 ms | | 104 ms | |
| **export 10 clips + timelines** | 29,6 s | 1 / 15 / 11 / 17 % | **12,4 s** | 3 / – / 38 / 35 % |
| upscale x2 (animevideov3, 5 s 1080p → 4K) | 33,9 s | 32 % | **14,7 s** | **69 %** |
| RIFE x2 | 18,1 s | 12 % | **9,2 s** | 25 % |
| profondeur (PNG 16 bits) | 61,9 s | 3 % | **34,9 s** | 2 % ⁵ |

⁴ Le décodage et les embeddings sont dans le bruit de la machine : le run
intermédiaire (même code de saison, hôte plus calme) donnait 149 s et
16,6 s (saison 239 s). Le décodage n'est pas accéléré par ce chantier : il
est au plancher NVDEC (§5) ; le gain de la saison vient des proxys.
⁵ La profondeur reste dominée par le CPU (prétraitement HF, flux DIS, PNG) :
le modèle lui-même ne prend plus que 2,2 s sur 34,9.

« GPU » = utilisation moyenne des SM (nvidia-smi toutes les 250 ms) pendant
l'étape ; NVDEC / NVENC = moteurs vidéo de la carte. Les étapes d'indexation
se recouvrent (le décodage de l'épisode suivant tourne pendant les tags et
embeddings du précédent) : leur somme dépasse un peu le temps de la saison.

## 3. Ce qui est passé du CPU à la carte

| étape | avant | après | gain mesuré |
|---|---|---|---|
| **Proxys de survol** | 2e décodage 1080p complet de chaque épisode (NVDEC à 90 %) | la passe d'indexation encode en plus une copie 240p (branche `scale_cuda` → NVENC du même décodage, horodatages de la source) ; la passe proxys ne fait que la recouper (`h264_cuvid` → NVENC, images clés aux coupes) | 115,6 s → 28,2 s pour la saison (×4,1) |
| **Mouvement Search+** (Lucas-Kanade + RANSAC) | OpenCV dans 2 processus CPU (≈ 1,2 cœur sur 6) | `backend/searchplus/motion_gpu.py` : même estimateur en PyTorch, 256 paires par lot (coins Shi-Tomasi, pyramide 4 niveaux, aller-retour, hypothèses RANSAC tirées par le même générateur seedé) | libère ~1,2 cœur ; le décodage n'en est pas accéléré, il est au plancher NVDEC (voir §5) |
| **Conversion NV12 → RGB des images médianes** | numpy float32 sur le thread de lecture | même calcul sur la carte | embeddings identiques (cosinus 1,0) ; ~4 s de CPU par saison libérés |
| **Conversion RGB → YUV (I420) des sorties de finition** | OpenCV dans le thread d'écriture : **181,6 ms par image 4K** contre 95,6 ms pour l'upscale lui-même : le CPU dictait le rythme | `backend/finish/yuv_cuda.py` : conversion sur la carte, moitié moins d'octets sur le PCIe | upscale x2 : 33,9 → 14,7 s (GPU 32 → 69 %) ; RIFE x2 : 18,1 → 9,2 s |
| **Profondeur** (Depth Anything V2 Small) | une image par passe du modèle, PNG 16 bits écrits un par un (14 s pour 120 images) | lots de 8 en fp16 (prétraitement du processeur HF en 4 threads), PNG compressés par 4 threads | 61,9 → 34,9 s |
| **Export des clips** | 10 exports NVDEC → NVENC l'un après l'autre (NVDEC 15 %, NVENC 9 %) | 3 exports simultanés sur NVENC | 29,6 → 12,4 s |

## 4. Ce qui reste sur CPU, et pourquoi (chiffres)

| travail | coût mesuré | pourquoi il reste sur CPU |
|---|---|---|
| Lecture du pipe ffmpeg + boucle de coupes (histogramme 32 cases sur 160×90, tampon des médianes) | Python seul : 680 im/s sur l'extrait, ffmpeg seul : 815 im/s | les coupes doivent rester identiques à l'image près : l'histogramme est exact (entiers) ; passé de `np.histogram` à `bincount` (mêmes comptes, bit à bit) : 1,2 s → 0,3 s de CPU pour 4 min |
| Prétraitement du processeur Depth Anything (redimensionnement PIL) | ~15 ms par image, en 4 threads | la version GPU donnait 39,4 dB sur la profondeur finale, sous la barre des 40 dB (rejetée) |
| Lissage temporel de la profondeur (flux optique DIS d'OpenCV) | 6,8 s pour 120 images | pas de flux optique GPU sans nouvelle dépendance (pas de NVOF dans torch) ; les warps et la médiane sont déjà sur GPU (tour 1) |
| Vignettes JPEG 320×180 | 0,4 ms l'une (≈ 0,4 s par saison) | nvjpeg testé : pas plus rapide une fois les transferts comptés (rejeté) |
| Décodage des clips de finition (ffmpeg logiciel → RGB) | ~1 s pour 5 s de 1080p | gain potentiel minime ; un lecteur GPU demanderait de garder les images en tenseurs jusqu'au modèle |
| Recherche texte « chaude » | ~107 ms par requête (SQL des tags, enrichissement) | c'est du SQL : le calcul de similarité est une multiplication de 1045 vecteurs |
| Écritures SQLite, palette (k-means sur 48×27) | négligeable | orchestration |

## 5. Pourquoi la saison ne descend pas plus bas

Le décodage est au plancher matériel : la 3090 n'a **qu'un moteur NVDEC**
(deux décodages simultanés prennent exactement deux fois plus de temps),
qui décode le H.264 1080p à ~750 im/s et le HEVC 10 bits à ~1270 im/s. Pour
les 101 451 images de la saison, c'est ~117 s de décodage incompressible.
ffmpeg avec tout le graphe de filtres (3 `scale_cuda`, 2 `overlay_cuda`,
téléchargement) tient ce rythme (815 im/s mesurés) ; ce qui reste au-dessus
vient de la lecture Python image par image et du recouvrement avec les tags
et embeddings de l'épisode précédent dans le même processus.

## 6. Optimisations gardées / rejetées

**Gardées** (toutes derrière le backend `cuda`, Apple Silicon / CPU / ROCm
gardent leurs chemins) :

| optimisation | mesure de décision |
|---|---|
| Copie 240p des proxys pendant l'indexation, recoupée ensuite | proxys de la saison 115,6 s → 28,2 s (extrait 4 min : 8,7 s → 2,0 s) |
| `h264_cuvid` pour relire la copie 240p | 2,5 s → 1,6 s sur l'extrait (décodage limité par la latence par image) |
| Mouvement Search+ en PyTorch par lots | extrait : saison 30,2 s → 20,5 s avec les proxys ; libère les 2 processus CPU ; 96,2 % de classes identiques |
| Conversion RGB → I420 sur GPU (upscale, RIFE) | 4K : 181,6 ms/image sur CPU contre +0,4 ms sur GPU ; upscale 16,8 → 13,7 s isolé, 27,8 → 14,7 s dans le pipeline complet |
| Profondeur en lots de 8 + PNG en parallèle | modèle 4,85 s → 2,20 s pour 120 images (79 dB vs une image à la fois) ; job 61,9 s → 34,9 s |
| 3 exports NVENC simultanés | extrait, 10 clips : 15,2 s → 8,8 s |
| NV12 → RGB des médianes sur GPU | embeddings identiques (cosinus 1,0) |
| Histogramme des coupes par `bincount` | 1,2 s → 0,3 s de CPU pour 4 min, résultats bit à bit identiques (test) |

**Rejetées** (mesurées, puis retirées) :

| idée | mesure |
|---|---|
| Prétraitement GPU de Depth Anything (bicubique antialiasé + normalisation) | 39,4 dB sur la profondeur finale (même avec l'arrondi uint8 de PIL), sous les 40 dB exigés |
| Vignettes en nvjpeg | 79 vignettes : 33 ms GPU contre 30 ms CPU (0,4 ms l'une) : aucun gain |
| Upscale image entière (sans tuiles) sur 24 Go | 104,0 ms/image contre 95,6 ms en tuiles de 512 |
| Pipe ffmpeg de 1 Mo au lieu de 64 Ko | 8,4 à 11,1 s selon les runs dans les deux cas : pas d'effet mesurable au-dessus du bruit |
| Bords « reflect-101 » (comme OpenCV) dans le suivi GPU | 93,7 % de classes de mouvement identiques au lieu de 96,2 % |
| Copie 240p déjà à 15 i/s (filtre `fps`) | autre phase d'échantillonnage que `-r 15` ; gardée à la cadence source pour que la passe proxys choisisse ses images elle-même |
| Décoder deux épisodes en même temps | la 3090 n'a qu'un moteur NVDEC : 2 × 4 min en parallèle = 15,1 s contre 7,7 s pour un seul |

## 7. Qualité

Comparaison du run final avec le run « avant » sur les 3 épisodes
(`bench_pipeline.py --quality`), extrait 4 min pour le reste :

| garde-fou | seuil | résultat |
|---|---|---|
| coupes | identiques à l'image près | **1045 / 1045 identiques** (419 + 366 + 260) au run final ; puis correctif volontaire (ci-dessous) |
| embeddings SigLIP 2 | cosinus > 0,995 | **1,0000** partout (0 scène sous 0,995) ; test `hw` : 1,00000 contre la référence enregistrée avec le code de départ |
| tags top-5 | ≥ 97 % identiques | **100 %** |
| classe de mouvement | ≥ 95 % (sinon justifier) | **95,3 %** sur la saison (DBS 01 : 95,2 %, DBS 02 : 95,9 %, Black Clover : **94,6 %**) |
| finition | PSNR > 40 dB | upscale **55,9 dB**, RIFE **51,4 dB**, profondeur **76,8 dB** |
| proxys | visuellement identiques | 12 scènes : même nombre d'images 12/12 ; meilleure image à ±1 : **41,1 dB** médiane (min 32,3 dB, deux encodages à 500 kb/s), même indice pour 89 % des images ; copie 240p = source à 47 dB |
| vignettes | identiques | inchangées (même code CPU) |

**Mouvement, justification** : le suivi GPU reprend l'algorithme d'OpenCV
mais pas son arithmétique en virgule fixe (interpolation 14 bits) ni son
ordre exact de sélection des coins ; les écarts touchent surtout la classe
« tremblement » (`shake`, décidée par un rapport de cohérence entre paires,
sensible à une seule paire mal suivie) : shake ↔ zoom/statique/tilt. Black
Clover reste à 0,4 point du seuil (14 scènes sur 260). Variante testée pour
s'approcher d'OpenCV (bords reflect-101) : pire (93,7 % sur l'extrait).

**Correctif de position des coupes (après le run final)** : le test `hw` a
révélé que la passe unique NVIDIA (celle que l'indexation utilise) n'avait
jamais reçu le recalage « première image du plan » ajouté aux autres
détecteurs (2c0a16b) : **seulement 10,3 % des coupes** de l'extrait
tombaient sur la première image du nouveau plan, les autres 1 à 3 images
trop tard (la scène précédente se terminait par des images du plan
suivant). Corrigé (`a42f217`) : **87,2 %**, exactement les positions du
détecteur générique (78/78). Mêmes coupes détectées (79 sur l'extrait), mais
déplacées : la comparaison « identique à l'image près » avec l'avant ne
s'applique plus à ce commit, par construction. Les 10 coupes restantes hors
critère sont 7 séquences stroboscopiques (deux sauts d'histogramme
comparables à ±3 images) et 3 plans de 2 images fusionnés par la règle
d'écart minimal ; atteindre les 95 % du test demanderait de changer les
règles de détection (hors périmètre). Coût : une liste par image et 4
histogrammes par coupe ; extrait 4 min après correctif : saison 18,0 s.

## 8. Fiabilité

- **Mémoire** : pic VRAM de la saison 2208 Mo (avant : 2023 Mo) — la
  copie 240p et le suivi GPU ajoutent ~300 Mo ; pic RAM du sidecar
  3652 Mo (avant : 3655 Mo). **Aucun swap** dans aucun run (0 page
  sortie), aucun OOM.
- **Écart entre runs** : la VM partage l'hôte. Le run complet de la branche
  quasi finale (sans la conversion I420 GPU ni `h264_cuvid`) : 5 min 29 ;
  un run de la branche finale lancé pendant que `pytest` occupait l'autre VM
  du même hôte : 7 min 15 (décodages 1,7× plus lents, alors que le code de
  la saison n'avait pas changé) — écarté et refait au calme. Le décodage
  isolé de l'extrait varie de 8,4 à 11,1 s d'un run à l'autre.
- **Robustesse** : la copie 240p n'est gardée que si ffmpeg finit sans
  erreur (fichier `.part` renommé), sinon la passe proxys repart de la
  source comme avant ; `h264_cuvid` absent → décodage hwaccel ; échec du
  suivi GPU → estimateur CPU ; `AMV_MOTION_CPU=1` force l'ancien chemin. Les
  copies sont effacées à la fin de chaque file.
- **Bugs trouvés** : (1) la session précédente avait laissé tourner un run
  de l'ancien code (`jarvis/v04`), arrêté au démarrage ; (2) premier essai
  de la copie 240p avec le filtre `fps=15` et les horodatages régénérés par
  l'encodeur : les proxys montraient des images décalées (PSNR 11-23 dB
  contre les anciens) — corrigé (cadence source, `-enc_time_base 1/1000`),
  vérifié image par image (47 dB contre la source) ; (3) un double de test
  de l'upscaler ne connaissait pas l'argument `i420` (corrigé, l'argument
  n'est passé que si la conversion GPU est active).

## 9. Limites

- **Plancher NVDEC** : ~117 s de décodage incompressible pour la saison
  (un seul moteur NVDEC sur la 3090). Une carte avec deux NVDEC (A-series,
  RTX 40xx haut de gamme) pourrait décoder l'épisode suivant en parallèle.
- **6 vCPU partagés** : la VM du NAS partage l'hôte Proxmox (Ryzen 9 5900X)
  avec d'autres VM, dont celle où tournent les tests. Deux runs identiques
  du décodage de l'extrait ont varié de 8,4 à 11,1 s ; un run complet lancé
  pendant un `pytest` sur l'autre VM a donné 7 min 15 au lieu de
  ~5 min 30 (écarté, voir §8). Les chiffres sont des runs uniques, pas des
  médianes.
- **Ce qui reste CPU** : la lecture image par image du pipe dans Python
  (680 im/s), le flux optique DIS de la profondeur (6,8 s / 120 images), le
  prétraitement Depth Anything, le décodage des clips de finition, le SQL.
- **Copie 240p des proxys** : ~320 Mo par épisode écrits dans le dossier de
  données pendant l'indexation, supprimés après la passe proxys. Sur un
  disque dur partagé avec les vidéos, ces écritures concurrencent la lecture
  des épisodes s'ils ne sont pas en cache.
- **Phase des proxys** : les proxys sont recoupés dans la copie 240p
  (identique à la source image pour image, 47 dB) au lieu de la source ; le
  choix des images à 15 i/s peut tomber une image source à côté (41 ms) de
  l'ancien : même contenu, même minutage de scène, pas les mêmes octets.
- **Mouvement** : 95,4 % de scènes de même classe sur la saison, 94,6 % sur
  Black Clover (voir §7) ; les écarts sont des scènes limites.
- **Pas testé** : cartes 8-12 Go avec ce code (le chemin 8 Go du tour 1 n'a
  pas été rejoué), Windows, ROCm.

## 10. Vérifications de fin

| vérification | où | résultat |
|---|---|---|
| `uv run pytest -m "not slow"` | VPS | **171 passés**, 1 ignoré (hw) |
| `uv run pytest -m "not slow"` | PC RTX 3090 | 168 passés, **3 échecs préexistants** : les mêmes 3 tests échouent à l'identique sur le code de départ du chantier (2 tests de finition sur la durée audio avec le ffmpeg 5.1 système, 1 test Apple Silicon qui suppose le décodage logiciel) |
| `tests/hw` (marqueur `hw`) | PC RTX 3090, extrait réel, vrais modèles | **5 passés, 1 échec** : alignement des coupes 87,2 % < 95 % (10,3 % au départ du chantier sur la passe d'indexation, 87,2 % pour le détecteur générique au départ comme à la fin, voir §7) ; tagger PyTorch CUDA fp16 sur plusieurs lots OK, SigLIP 1,00000 vs référence, gel API max 748 ms (< 2 s), 0 swap, Beerus 15/114 scènes |
| `cd app && npx tsc --noEmit && npx vitest run && npm run build` | VPS | **OK** (12 tests vitest) |

Commits du chantier sur cette machine : `f39eea4` (mouvement GPU), `b385d08`
(copie 240p des proxys, NV12→RGB GPU), `c1ac6f2` (profondeur), `49601fe`
(exports parallèles), `53679f3` (histogramme), `3ad705c` (I420 GPU),
`a67afda` (`h264_cuvid`), `9c8e971` (correctif de test), `a42f217`
(position des coupes NVIDIA). Run « avant » : `8b12280` ; run « après » :
`9c8e971` (le correctif `a42f217` ne touche pas aux temps : vérifié en
quick).

## Résumé

Sur la RTX 3090 (VM à 6 vCPU partagés), le pipeline complet est passé de
**8 min 12** (avant le chantier 2) à **5 min 49** (fin du chantier 2) puis à
**4 min 32 – 4 min 43** au tour 3 ; dans les mêmes conditions d'hôte, le tour
3 est 18 à 26 % plus rapide que la fin du chantier 2 (runs alternés). La
v0.3 était estimée à ~45 min 30.

| | v0.3 (estimé) | avant chantier 2 | après chantier 2 | **tour 3** |
|---|---:|---:|---:|---:|
| pipeline complet | ≈ 45 min 30 | 8 min 12 | 5 min 49 | **4 min 32 – 4 min 43** |
| saison | ≈ 39 min 50 | 336,6 s | 269,1 s | **193,7 – 203,9 s** (158,6 s au mieux) |
| recherches | ≈ 45 s | 11,8 s | 9,1 s | **7,6 – 7,7 s** |
| export | ≈ 59 s | 29,6 s | 12,4 s | **11,5 – 12,9 s** |
| finition | 237,6 s | 114,0 s | 58,8 s | **57,9 – 60,3 s** |

Gains principaux du tour 3 : décodage NVDEC directement en tenseurs CUDA
avec une réduction au bit près identique à celle d'ffmpeg (CPU de la passe
divisé par ~2, ×2 sur l'épisode HEVC), deux décodages en vol, tags et
embeddings ×2-3 (lots GPU), queue de proxys ÷3,7, et un bug préexistant
corrigé : les proxys de survol étaient décalés d'une scène. Sur cette VM,
c'est le CPU partagé, pas la 3090, qui limite l'indexation ; c'est pourquoi
le M1 Pro (saison 2 min 40) reste devant sur ce seul poste, alors que la
3090 le devance sur tout le reste.
