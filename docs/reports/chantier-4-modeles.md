# Chantier 4 — le meilleur modèle par palier de mémoire (8 / 12 / 16 / 24 Go)

Branche `jarvis/v04-models4` (partie de `jarvis/v04-perf-merged` : pipeline
optimisé des chantiers 2 et 3, NVIDIA + Mac). Mesures du 25 octobre 2026 sur
deux machines :

- **PC RTX 3090** : RTX 3090 24 Go (driver 580), VM Proxmox à 6 vCPU d'un
  Ryzen 9 5900X partagé, ~15-21 Go de RAM, torch 2.11 + cu126 ;
- **MacBook Pro M1 Pro 16 Go** (8 cœurs performance + 2 efficacité, GPU
  16 cœurs), macOS 26, torch 2.11 (MPS), coremltools 9.

Les trois épisodes réels servent partout : *Dragon Ball Super — Beerus* 01
et 02 (H.264 1080p) et *Black Clover* S02E04 (HEVC 10 bits 1080p),
1 045 scènes au total.

## En bref

- **Recherche** : SigLIP 2 **so400m** (NaFlex, 128 patches) remplace
  SigLIP 2 base dès 8 Go de VRAM : **+12 % de précision@10** (0,71 → 0,80)
  et +10 % de nDCG@10 sur 25 requêtes jugées à la main (1 225 jugements),
  +21 % sur les requêtes en français ; coût dans le pipeline complet
  **+8 %** (6 min 25 contre 5 min 57, médianes de 3 runs alternés), surtout
  son chargement. Sur Mac il est 3,4× plus lent que base et a fait swapper
  le M1 Pro 16 Go : réservé aux Mac de 24 Go et plus. Meta PE-Core (T, S,
  B, L) et les SigLIP 2 à résolution fixe sont moins bons ou bien plus
  lents.
- **Tags** : **wd-vit reste** partout. Aucun tagger ouvert ne connaît
  Beerus, Whis, Kaiô ou la nouvelle saison de Black Clover (ni wd v3, ni
  PixAI v0.9) ; les gros taggers trouvent Goku un peu plus souvent mais
  sont 4 à 14× plus lents. Les personnages se cherchent par le nom (so400m)
  ou via les personnages Search+.
- **Upscale** : **2xHFA2kAVCCompact** (CC-BY-4.0) : moitié moins d'écart
  à l'original que animevideov3 (LPIPS 0,044 contre 0,082, +4,4 dB), même
  vitesse. AnimeJaNai est rejeté (licence NC-SA).
- **Interpolation** : **RIFE 4.26** (même graphe et même vitesse que 4.25,
  un peu meilleur) ; 4.26 heavy en Qualité max. GIMM-VFI et AMT rejetés
  (licence), FILM mesuré : 6× plus lent et pas meilleur.
- **Profondeur** : **Video Depth Anything Small** (Apache-2.0), intégré et
  réécrit pour rester sur le GPU : scintillement **÷ 4,8** et plus rapide
  que Depth Anything V2 (26,8 contre 14,8 im/s) ; 518 px à partir de 16 Go,
  364 px à 8/12 Go, en Rapide et sur Mac 16 Go+.
- **Paliers** : Auto / Rapide / 8 / 12 / 16 / 24 Go / Qualité max /
  Personnalisé (Mac : 8 / 16 / 24 / 32 Go et plus), choisis selon la VRAM
  ou la mémoire unifiée. Tous passent la saison complète sans OOM dans une
  carte simulée (lest sur la 3090). Sur le M1 Pro, le palier 16 Go est
  **5 % plus rapide** que l'actuel avec une meilleure profondeur et un
  meilleur upscale.
- **Intégré** : sélecteur FR/EN dans Settings > Models, embeddings
  versionnés par modèle et **réindexation des embeddings seuls** (6 min 07
  pour 3 épisodes sur la 3090, retour arrière en 1 s), contrôle de licence
  avant téléchargement, 2 bogues corrigés (wd-swinv2 sur CUDA, cache Core ML
  du tagger).

## Sommaire

1. Méthode
2. Les candidats : retenus et rejetés
3. Résultats par fonction (RTX 3090)
4. Dans le vrai pipeline (RTX 3090)
5. Recommandation NVIDIA : fonction × palier
6. Apple Silicon (M1 Pro)
7. Ce qui a été intégré
8. Limites et points ouverts
9. Planches visuelles et données brutes

## 1. Méthode

### 1.1 Ce qu'on compare, et à quoi

La référence est **le pipeline optimisé des chantiers 2 et 3** (code de
départ de la branche) avec les modèles actuels : SigLIP 2 base NaFlex
(128 patches) pour la recherche, wd-vit-tagger-v3 pour les tags,
realesr-animevideov3 pour l'upscale, RIFE 4.25 pour l'interpolation, Depth
Anything V2 Small pour la profondeur.

Chaque candidat est mesuré de deux façons :

1. **Isolé**, sur les mêmes images que le pipeline lui donne : les
   1 045 images médianes des scènes des 3 épisodes (512×288, extraites
   par le code de l'application), ou des images 1080p tirées des épisodes
   pour la finition. Débit, chargement à froid, mémoire GPU max, RAM max,
   qualité. Script : `tests/eval/eval_models.py` (recherche, tags) et
   `tests/eval/eval_finish.py` (upscale, interpolation, profondeur).
2. **Dans le vrai pipeline** (`tests/bench_pipeline.py`) : les 3 épisodes
   indexés d'une traite, 18 recherches, 10 exports + 4 timelines,
   puis upscale x2 + RIFE x2 + profondeur sur un extrait de 5 s. C'est ce
   temps-là qui juge le surcoût d'un modèle.

### 1.2 Simuler une carte plus petite

Sur la 3090, une carte de 8, 12 ou 16 Go est simulée par un **lest** : un
processus qui réserve sur le GPU tout sauf N Go (`ballast.py`), si bien que
tout ce qui tourne ensuite (PyTorch, ONNX Runtime, NVDEC/NVENC d'ffmpeg)
ne voit que N Go libres, comme sur la vraie carte. En plus, la variable
`AMV_VRAM_LIMIT_GB=N` (ajoutée dans `backend/models/device.py`) plafonne
l'allocateur PyTorch et fait croire à l'application que la carte a N Go :
tailles de lot, palier « Auto » et tuilage de l'upscale suivent donc ceux
d'une vraie carte de N Go (lot 16 à 8 Go, 32 à 12/16 Go, 64 à 24 Go).
Un modèle n'est **éligible** à un palier que s'il passe sans OOM.

Sur le Mac, 8 Go est simulé en plafonnant Metal :
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5` (la moitié de l'ensemble de travail
recommandé du M1 Pro 16 Go, soit ~5,3 Go, ce qu'un Mac de 8 Go accorde au
GPU), `AMV_VRAM_LIMIT_GB=8`, et en vérifiant que l'empreinte mémoire
totale du processus reste sous ~5 Go (il faut laisser ~3 Go à macOS et à
l'interface). 24 et 32 Go ne peuvent pas être mesurés sur une machine de
16 Go : ils sont **extrapolés** (mémoire mesurée + marge) et marqués comme
tels.

### 1.3 Mesurer la qualité

| fonction | protocole | jeu jugé à la main |
|---|---|---|
| recherche | 25 requêtes (personnages, actions, plans, couleurs, ambiances ; 18 en anglais, 7 en français) ; top 10 de chaque modèle, **mis en commun** puis jugé en regardant les images (0 = hors sujet, 1 = partiel, 2 = pertinent) ; précision@10 (note ≥ 1) et nDCG@10 (gains 0/1/2). « Plans similaires » : 10 scènes de départ, top 5 jugé. | `tests/eval/search_queries.json`, `tests/eval/search_judgments.json` (1 225 jugements) |
| tags | 60 scènes (20 par épisode, espacées régulièrement) annotées à la main : personnages visibles et 21 tags visuels (extérieur, ciel, nuit, espace, 1boy, plusieurs garçons, sourire, bouche ouverte, yeux fermés, nourriture, bâtiment, eau, cheveux noirs/blonds, aura, arme, sans humain…) ; précision, rappel, F1 aux seuils de l'application (0,35 général, 0,85 personnage). | `tests/eval/tagger_truth.json` |
| upscale | 36 images 1080p (12 par épisode, dont 4 très sombres) réduites en 540p (moyenne de zone), remontées x2, comparées à l'original : PSNR et SSIM sur la luminance, LPIPS (AlexNet) ; plus une planche visuelle jugée (traits, aplats, artefacts). | planche `<données>/reports/c4-upscale-planche.png` |
| interpolation | 18 courts plans (6 par épisode : les 3 plus calmes et les 3 plus intenses selon la mesure d'intensité de Search+), 13 images 1080p chacun ; on retire une image sur deux et on la régénère (t = 0,5). Les triplets où l'image retirée est un **doublon** (animation « en deux ») sont comptés à part : là, la bonne réponse est de recopier, pas d'inventer un mouvement. | planche `<données>/reports/c4-interp-planche.png` |
| profondeur | 18 plans calmes/fixes : **scintillement** = variation moyenne de la profondeur d'une image à la suivante, sur les pixels dont la couleur n'a pas bougé, en % de l'étendue de profondeur du plan ; brut, puis après le lissage temporel de l'application. Vitesse sur ~117 images 1080p d'action. | planche `<données>/reports/c4-depth-planche.png` |

Les jugements sont **mis en commun** (« pooling », comme dans les
évaluations TREC) : une image n'est jugée qu'une fois, quel que soit le
nombre de modèles qui la proposent, et un nouveau modèle n'ajoute que ses
images encore jamais vues. `eval_models.py pool` liste ce qu'il reste à
juger, `sheets` fabrique les planches, `score-search` recalcule tout.

### 1.4 Fiabilité

Pour chaque modèle : plantage ou OOM à chaque palier, déterminisme (deux
passes identiques), comportement sur le HEVC 10 bits (Black Clover), sur
les plans très sombres et les flashs (inclus dans les jeux d'images), et
temps de première utilisation (téléchargement bridé à 10 Mo/s + premier
chargement / compilation Core ML).

### 1.5 Estimations pour d'autres cartes

La 3090 calcule plus vite qu'une carte de 8 ou 12 Go. Les débits des autres
cartes sont **estimés** à partir des ratios de puissance fp16 (cœurs
tenseurs) et de bande passante mémoire, en prenant le plus défavorable des
deux pour les gros modèles (limités par le calcul) et la bande passante
pour les petits (limités par la mémoire) :

| carte | VRAM | fp16 tenseur dense (accumulation fp32) | bande passante | facteur appliqué au débit 3090 |
|---|---:|---:|---:|---:|
| RTX 3090 (mesurée) | 24 Go | 71 TFLOPS | 936 Go/s | 1,00 |
| RTX 4060 | 8 Go | 30 TFLOPS | 272 Go/s | 0,30 – 0,42 |
| RTX 3060 | 12 Go | 26 TFLOPS | 360 Go/s | 0,36 – 0,38 |
| RTX 4060 Ti 16 Go | 16 Go | 44 TFLOPS | 288 Go/s | 0,31 – 0,62 |
| RTX 4080 | 16 Go | 97 TFLOPS | 717 Go/s | 0,77 – 1,37 |
| RTX 4090 | 24 Go | 165 TFLOPS | 1 008 Go/s | 1,08 – 2,3 |

(Chiffres des livres blancs Ampere et Ada de NVIDIA ; borne basse = ratio
de bande passante, borne haute = ratio de calcul.) Ces facteurs ne s'appliquent qu'au calcul des modèles : le
décodage vidéo (NVDEC), qui domine le temps d'indexation, a un débit
proche d'une génération à l'autre.

## 2. Les candidats : retenus et rejetés

Licence vérifiée **avant** tout téléchargement, sur la fiche du modèle et
dans le dépôt du code (audit complet : liens, tailles, SHA-256). Règle :
poids **et** code redistribuables avec usage commercial (Apache-2.0, MIT,
BSD, CC-BY-4.0 avec attribution, CC0). Rejet : NC, NC-SA, licences
« recherche », licences propriétaires, et le GPL (copyleft) par prudence.

### 2.1 Mesurés

| fonction | candidats mesurés (licence) |
|---|---|
| recherche | SigLIP 2 base NaFlex 128 et 256 patches (actuel), base 384 px, large 384 px, so400m NaFlex 128 et 256, so400m 384 px (Apache-2.0) ; Meta PE-Core T, S, B et L (Apache-2.0, via open_clip) |
| tags | wd-vit (actuel), wd-swinv2, wd-convnext, wd-vit-large, wd-eva02-large v3 (Apache-2.0) ; PixAI tagger v0.9 en ONNX deepghs (Apache-2.0) |
| upscale | realesr-animevideov3 et RealESRGAN x4plus anime 6B (actuels, BSD-3) ; Real-CUGAN up2x conservative et no-denoise (MIT) ; 2xHFA2kAVCCompact (CC-BY-4.0) ; RealESRGANv2-animevideo-xsx2 (BSD-3) ; 2x_ModernSpanimation V2 (MIT) ; bicubique comme témoin |
| interpolation | RIFE 4.25 (actuel), 4.26, 4.25 lite, 4.26 heavy (MIT) ; simple fondu comme témoin |
| profondeur | Depth Anything V2 Small à 518 px (actuel) et 364 px ; Video-Depth-Anything Small (Apache-2.0) |

### 2.2 Rejetés

| candidat | raison |
|---|---|
| AnimeJaNai V3 / V3.1 (Compact, UltraCompact, SuperUltraCompact) | **licence** CC-BY-NC-SA-4.0 (fiche OpenModelDB et dépôt) |
| AniSD (Sirosky) | **licence** CC-BY-NC-4.0 |
| APISR | **licence** GPL-3.0 + « usage académique seulement » |
| GIMM-VFI | **licence** S-Lab 1.0 (non commerciale) |
| AMT | **licence** CC-BY-NC-4.0 |
| BiM-VFI, SGM-VFI | **licence** absente (aucun droit accordé) |
| Video-Depth-Anything Base / Large, DA3 Large / Giant | **licence** CC-BY-NC-4.0 |
| Depth Pro (Apple), DepthCrafter | **licence** recherche seulement |
| MetaCLIP 2, jina-clip-v2 | **licence** CC-BY-NC-4.0 |
| Apple DFN, MobileCLIP 2 | **licence** Apple « research only » |
| Camie tagger v1/v2 | **licence** GPL-3.0 (copyleft) |
| PixAI tagger v1.0 | **licence** : la fiche dit Apache-2.0 mais le modèle est un dérivé de SAM 3, sous « SAM License » propriétaire de Meta qui s'impose aux dérivés |
| PixAI tagger v0.9 (mesuré) | **qualité / coût** : même vocabulaire de personnages que wd (ni Beerus ni Whis), F1 des tags plus bas, 14× plus lent que wd-vit, ONNX Runtime réserve ~20 Go de VRAM |
| PE-Core T / S / B (mesurés) | **qualité** : nDCG@10 0,52 / 0,55 / 0,59, en dessous du modèle actuel (0,67) |
| PE-Core L (mesuré) | **qualité / coût** : nDCG 0,70 < so400m 0,73, 3× plus lent |
| SigLIP 2 large 384 px, base 384 px, so400m 384 px (mesurés) | **coût** : qualité entre base et so400m NaFlex, mais 3 à 4× plus lents (image déformée en carré, prétraitement CPU) |
| RIFE 4.25 lite (mesuré) | **qualité / coût** : un peu moins bon que 4.25, pas plus rapide (pas de goulot de calcul à 1080p) |
| ModernSpanimation V2, animevideo-xsx2, Real-CUGAN no-denoise, x4plus anime 6B (mesurés) | **qualité** : plus loin de l'original que l'actuel (LPIPS 0,096 à 0,127), artefacts visibles (bavures, halos, traits épaissis) |
| Real-CUGAN conservative (mesuré) | **coût** : traits nets mais 2× plus lent qu'un Compact et plus loin de l'original que 2xHFA2kAVCCompact ; non intégré |
| FILM, EMA-VFI, IFRNet, VFIMamba, DA3 Small/Base, Distill-Any-Depth, MoGe-2 | licence OK mais **non mesurés** faute de temps (voir Limites) |

## 3. Résultats par fonction (RTX 3090)

Les débits « isolés » sont mesurés sur les 1 045 images médianes des scènes
(512×288, lot de 64 comme le pipeline sur une carte de 24 Go), après un
lot de chauffe. « Chargement à froid » = nouveau processus jusqu'au premier
lot traité (poids déjà dans le cache disque de l'OS, comme au deuxième
lancement de l'application). Mémoire GPU = pic de l'allocateur PyTorch.

### 3.1 Recherche sémantique (embeddings)

| modèle | P@10 | nDCG@10 | plans similaires P@5 | débit isolé | chargement | VRAM max | RAM max |
|---|---:|---:|---:|---:|---:|---:|---:|
| **SigLIP 2 base NaFlex 128 (actuel)** | 0,708 | 0,666 | 0,84 | 290 img/s | 18 s | 1,0 Go | 3,6 Go |
| SigLIP 2 base NaFlex 256 | 0,724 | 0,684 | 0,82 | 231 img/s | 17 s | 1,3 Go | 3,6 Go |
| SigLIP 2 base 384 px | 0,748 | 0,716 | 0,90 | 88 img/s | 21 s | 1,6 Go | 3,6 Go |
| SigLIP 2 large 384 px | 0,740 | 0,676 | 0,90 | 57 img/s | 47 s | 2,8 Go | 6,4 Go |
| **SigLIP 2 so400m NaFlex 128** | **0,796** | 0,733 | 0,90 | 212 img/s | 63 s (1re fois) / 10 s | 2,5 Go | 7,8 Go |
| SigLIP 2 so400m NaFlex 256 | 0,768 | **0,737** | 0,90 | 158 img/s | 10 s | 2,9 Go | 7,9 Go |
| SigLIP 2 so400m 384 px | 0,784 | 0,717 | 0,88 | 52 img/s | 42 s | 3,7 Go | 7,9 Go |
| PE-Core T 16 384 | 0,536 | 0,517 | 0,80 | 113 img/s | 10 s | 0,4 Go | 2,3 Go |
| PE-Core S 16 384 | 0,592 | 0,551 | 0,88 | 88 img/s | 9 s | 0,7 Go | 2,3 Go |
| PE-Core B 16 | 0,644 | 0,585 | 0,92 | 141 img/s | 30 s | 1,2 Go | 4,6 Go |
| PE-Core L 14 336 | 0,772 | 0,702 | 0,90 | 68 img/s | 44 s | 2,8 Go | 6,3 Go |

25 requêtes, 1 225 jugements à la main. Détail du nDCG@10 par type de
requête (personnages, actions, plans, couleurs, ambiances) et par langue :

| modèle | personnages | actions | plans | couleurs | ambiances | anglais | français |
|---|---:|---:|---:|---:|---:|---:|---:|
| base NaFlex 128 (actuel) | 0,68 | 0,68 | 0,63 | 0,69 | 0,75 | 0,72 | 0,57 |
| so400m NaFlex 128 | 0,74 | 0,75 | 0,75 | 0,65 | 0,88 | 0,77 | 0,69 |
| so400m NaFlex 256 | 0,74 | 0,74 | 0,83 | 0,63 | 0,87 | 0,75 | 0,78 |

(Ces sous-totaux portent sur 3 à 7 requêtes chacun : ce sont des tendances.)

**Lecture.** so400m NaFlex 128 est le meilleur : **+12 % de précision@10**
(0,708 → 0,796) et **+10 % de nDCG** par rapport au modèle actuel, avec un
gain net sur les requêtes en français (+21 %) et sur les ambiances. Les
personnages par leur nom marchent déjà bien avec les deux (Goku et Beerus
10/10, Whis 10/10 puis 9/10), so400m reconnaît mieux Asta (4 → 7/10) ; le
reste de la différence vient des descriptions. Les
variantes à résolution fixe (384 px) écrasent l'image 16:9 en carré et
passent par un prétraitement CPU : 3 à 4× plus lentes pour un résultat
moins bon que NaFlex. PE-Core, très bon sur les benchmarks photo, est en
dessous sur l'anime (PE-Core L seul s'approche de so400m, 3× plus lent).
Les 256 patches de so400m gagnent un peu en nDCG et sur les plans (gros
plan / plan large), perdent en précision@10 et coûtent 25 % de débit : c'est
l'option « Qualité max ».

**Coût dans le pipeline** (voir § 4) : à lot de 64, l'étape « embeddings »
de la saison passe de ~9,8 s à ~10,2 s ; ce n'est pas le calcul qui limite
cette étape. Le vrai coût est ailleurs : **téléchargement** de 4,5 Go au
lieu de 1,5 Go (≈ 7 min 30 à 10 Mo/s au lieu de 2 min 30), **1re requête
à froid** plus lente (15 s au lieu de 10 s : chargement du modèle) et
**RAM** du sidecar plus haute (+1,7 Go au pic, chargement des poids fp32).

### 3.2 Tagger

| modèle | tags P / R / F1 (seuil 0,35) | meilleur F1 (seuil) | Goku reconnu (12 scènes) | personnages connus du vocabulaire | débit isolé | chargement | VRAM max |
|---|---|---:|---:|---|---:|---:|---:|
| **wd-vit v3 (actuel)** | 0,61 / 0,78 / **0,69** | **0,74** (0,50) | 7 | Goku, Vegeta | **251 img/s** | 4 s | 0,8 Go |
| wd-swinv2 v3 | 0,61 / 0,78 / 0,69 | 0,72 (0,65) | 7 | Goku, Vegeta | 135 img/s | 9 s | 1,7 Go |
| wd-convnext v3 | 0,60 / 0,81 / 0,69 | 0,73 (0,55) | 5 | Goku, Vegeta | 59 img/s | 17 s | 1,3 Go |
| wd-vit-large v3 | 0,57 / 0,77 / 0,66 | 0,71 (0,60) | 9 | Goku, Vegeta | 66 img/s | 9 s | 1,7 Go |
| wd-eva02-large v3 | 0,55 / 0,84 / 0,66 | 0,73 (0,65) | 9 | Goku, Vegeta | 54 img/s | 30 s | 2,0 Go |
| PixAI v0.9 (ONNX) | 0,54 / 0,76 / 0,63 | 0,66 (0,45) | 10 (+1 faux) | Goku, Vegeta | 18 img/s | 16 s | ~20 Go (réservés par ONNX Runtime) |

60 scènes annotées : 36 apparitions de personnages (Goku 12, Beerus 11,
Whis 6, Kaiô 3, Vegeta, Mr Satan, Freezer, Cell) et 21 tags visuels.

**Le point clé : aucun tagger ne connaît Beerus, Whis, Kaiô ni les
personnages de la nouvelle saison de Black Clover.** Leurs vocabulaires
(Danbooru, 10 861 tags pour wd v3, 13 461 pour PixAI) n'ont que Goku et
Vegeta pour ces deux séries (vérifié dans `selected_tags.csv`). Un plus
gros tagger trouve Goku un peu plus souvent (9-10/12 au lieu de 7/12) mais
ne couvre pas un seul personnage de plus. Pour les personnages, la bonne
réponse est la **recherche sémantique** (so400m trouve Beerus et Whis par
leur nom, 10/10) et les **personnages Search+** (prototypes SigLIP créés
par l'utilisateur), pas le tagger. Sur les tags visuels, wd-vit a le
meilleur F1 et est 4 à 14× plus rapide : **il reste le tagger partout**.
wd-eva02-large est proposé en « Qualité max » (meilleur rappel de Goku,
même F1 à son seuil optimal, ~+16 s sur une saison de 3 épisodes).

Fiabilité : wd-swinv2 **plantait** sur CUDA (tampons de position relatifs
restés sur le périphérique « meta » au chargement rapide : tous les lots
échouaient, tags vides) — corrigé (`fix(tagger)`). Le programme Core ML du
tagger était mis en cache sous un nom fixe : changer de tagger sur Mac
réutilisait celui de wd-vit — corrigé aussi.

### 3.3 Upscale

36 images 1080p réelles (dont 4 très sombres) réduites en 540p puis
remontées en x2 ; comparaison avec l'original. LPIPS : plus bas = plus
proche à l'œil.

| modèle | PSNR (dB) | SSIM | LPIPS | PSNR plans sombres | débit 540p→1080p | VRAM | licence |
|---|---:|---:|---:|---:|---:|---:|---|
| bicubique (témoin) | 43,25 | 0,987 | 0,035 | 48,9 | 327 im/s | – | – |
| **realesr-animevideov3 (actuel)** | 38,09 | 0,975 | 0,082 | 44,1 | 47 im/s | 0,6 Go | BSD-3 |
| RealESRGAN x4plus anime 6B (actuel « qualité ») | 35,45 | 0,961 | 0,127 | 38,5 | 4,9 im/s | 3,4 Go | BSD-3 |
| **2xHFA2kAVCCompact** | **42,50** | **0,980** | **0,044** | **45,3** | 48 im/s | 0,6 Go | CC-BY-4.0 |
| Real-CUGAN up2x conservative | 38,99 | 0,977 | 0,087 | 41,1 | 26 im/s | 2,0 Go | MIT |
| Real-CUGAN up2x no-denoise | 36,55 | 0,967 | 0,102 | 38,6 | 26 im/s | 2,0 Go | MIT |
| RealESRGANv2 animevideo xsx2 | 34,80 | 0,962 | 0,110 | 38,4 | 49 im/s | 0,6 Go | BSD-3 |
| 2x ModernSpanimation V2 | 36,93 | 0,955 | 0,096 | 38,1 | 20 im/s | 1,3 Go | MIT |

**Lecture.** Sur ce protocole, le bicubique gagne : les épisodes 1080p
sont déjà doux (souvent produits en 720-900p), et un upscaler « IA »
re-dessine les traits au lieu de retrouver l'original. Parmi les modèles,
**2xHFA2kAVCCompact** est de loin le plus fidèle (LPIPS 0,044 contre 0,082
pour l'actuel), à la même vitesse (même architecture Compact). Planche
jugée (`c4-upscale-planche.png`) :

- animevideov3 : traits plus noirs et nets mais **aplats lissés** (le grain
  et les dégradés de Black Clover disparaissent), teinte légèrement
  réchauffée ;
- 2xHFA2kAVCCompact : le plus proche de l'original, garde la texture,
  traits un peu plus nets que le bicubique, aucun artefact vu ;
- Real-CUGAN conservative : traits très nets, ombrages un peu lissés,
  propre ; 2× plus lent ;
- x4plus anime 6B : traits épaissis, effet « peinture à l'huile », halos ;
  ModernSpanimation : bavures et halos ; xsx2 : très lissé.

Dans le vrai travail (x2 d'un extrait 1080p vers la 4K, pipeline) :
voir § 4 (17,0 s contre 15,2 s pour 5 s de vidéo, mesure unique).

### 3.4 Interpolation

18 plans (6 calmes, 6 d'action, 6 fixes), 13 images 1080p chacun, une
image sur deux retirée puis régénérée.

| modèle | débit 1080p | calme : PSNR / LPIPS | action : PSNR / SSIM / LPIPS | doublons : PSNR |
|---|---:|---|---|---:|
| fondu simple (témoin) | 13 im/s | 44,79 / 0,020 | 21,75 / 0,801 / 0,328 | 40,2 |
| **RIFE 4.25 (actuel)** | 21 im/s | 45,19 / 0,012 | 20,99 / 0,802 / 0,358 | 38,5 |
| **RIFE 4.26** | 20 im/s | 45,37 / 0,012 | 21,02 / 0,802 / 0,357 | 38,6 |
| RIFE 4.25 lite | 16 im/s | 45,15 / 0,012 | 20,91 / 0,800 / 0,367 | 38,4 |
| RIFE 4.26 heavy | 16 im/s | **45,65** / 0,012 | **21,05 / 0,803 / 0,356** | **38,7** |
| FILM (port TorchScript, fp16) | 3,4 im/s | 44,84 / 0,012 | 20,76 / 0,781 / 0,361 | 39,1 |

**Lecture.** Sur les plans d'action de l'anime (grands déplacements,
images « en deux »), aucun modèle ne retrouve l'image manquante (PSNR
~21 dB) et les métriques préfèrent même le simple fondu, qui « floute » au
lieu de se tromper de position. À l'œil (`c4-interp-planche.png`), c'est
l'inverse : le fondu montre une **image fantôme** (deux visages
superposés), RIFE un seul personnage, avec des traits parfois déformés
autour de la bouche et des dents en mouvement rapide. 4.26 est un peu plus
propre que 4.25 (une double ligne de bouche en moins sur la planche),
4.26 heavy le meilleur (menton net où la lite laisse un fantôme), 4.25 lite
le moins bon et pas plus rapide. Doublons : quand l'image retirée
répétait sa voisine, RIFE invente un petit mouvement (38,5 dB) là où il
fallait recopier ; dans l'usage réel (ralenti x2 entre deux images source
identiques) RIFE rend bien l'image identique. Coupes franches : gérées par
l'application avant le modèle (détection de coupe → recopie), identique
pour tous les modèles.

### 3.5 Profondeur

Scintillement = variation moyenne d'une image à la suivante, en % de
l'étendue de profondeur du plan, sur les pixels immobiles de 18 plans
calmes/fixes (216 paires).

| modèle | débit 1080p | scintillement brut | après alignement | après lissage de l'application | VRAM |
|---|---:|---:|---:|---:|---:|
| **Depth Anything V2 Small 518 px (actuel)** | 14,8 im/s | 1,15 % | 0,81 % | 0,51 % | 2,1 Go |
| Depth Anything V2 Small 364 px | 31,6 im/s | 1,45 % | 0,83 % | 0,48 % | 1,1 Go |
| **Video Depth Anything Small 518 px** (notre intégration) | **26,8 im/s** | **0,24 %** | **0,14 %** | **0,11 %** | 11,8 Go |
| **Video Depth Anything Small 364 px** (notre intégration) | **66,5 im/s** | 0,35 % | 0,23 % | 0,16 % | 3,3 Go |
| Video Depth Anything Small 518 px (code d'origine) | 1,9 im/s | 0,24 % | 0,14 % | 0,11 % | 11,6 Go |

**Lecture.** Video Depth Anything traite des fenêtres de 32 images avec une
attention temporelle : la profondeur ne « respire » plus d'une image à
l'autre. Scintillement brut **÷ 4,8** par rapport à l'actuel, et même
contre l'actuel *après* son lissage par flot optique (0,51 %), VDA *sans*
lissage fait mieux (0,24 %). Le code d'origine (`infer_video_depth`)
remonte chaque carte à la taille de la vidéo et aligne les fenêtres sur le
CPU : 1,9 im/s. Notre intégration garde tout sur le GPU à la résolution du
modèle — même résultat au centième près (0,242 % des deux côtés), **14×
plus rapide**, et plus rapide que Depth Anything V2. Le prix : la VRAM (la
fenêtre de 32 images) — 11,8 Go réservés au pic à 518 px, 3,3 Go à 364 px.
Planche (`c4-depth-planche.png`) : VDA garde le même relief d'une image à
l'autre là où DA V2 éclaircit brusquement une mèche de cheveux sur la
3e image ; sur un plan presque noir, tous les modèles inventent un
« vignettage » sans objet (mais stable).

### 3.6 Fiabilité

| point | constat |
|---|---|
| plantages / OOM | aucun pour les modèles retenus, à 8, 12, 16 et 24 Go (§ 4.2, 4.3). Deux défauts **trouvés et corrigés** : wd-swinv2 échouait à chaque lot sur CUDA ; le programme Core ML du tagger était partagé entre modèles. VDA 518 px sur le Mac 16 Go : saturation mémoire (≈ 10 Go de swap), d'où 364 px sur Mac. |
| déterminisme | deux passes identiques (so400m, wd-vit, 3090) : écart maximal **0,0** (sorties identiques au bit près) ; VDA : même scintillement au millième entre le code d'origine et notre intégration ; Mac ↔ CUDA : § 6.1 |
| HEVC 10 bits (Black Clover) | toutes les mesures incluent l'épisode HEVC (images médianes, images 1080p d'upscale, plans d'interpolation et de profondeur) : aucun écart de comportement ; la réindexation relit aussi le HEVC sans erreur |
| plans sombres / flashs | upscale : PSNR des 4 images très sombres 45,3 dB pour HFA2k (48,9 bicubique, 44,1 actuel), pas d'artefact de bande ; profondeur : sur un plan presque noir, tous les modèles produisent un « vignettage » sans objet (stable dans le temps) ; recherche « ambiance sombre » : so400m 0,88 de nDCG sur les ambiances contre 0,75 |
| première utilisation (téléchargement à 10 Mo/s + compilation) | so400m 4,5 Go ≈ **7 min 30** (base : 2 min 30) puis 63 s au tout premier chargement (10 s ensuite) ; wd-eva02 1,26 Go ≈ 2 min ; VDA 116 Mo ≈ 12 s ; HFA2k 5 Mo et RIFE 25 Mo : instantané ; Mac : compilation Core ML de RIFE / DA V2 / tagger à la première taille d'image (10-65 s, mise en cache) |
| RAM | so400m : +1,4 Go de mémoire anonyme au chargement (2,5 contre 1,1 Go, sonde `load_mem.py`) ; le « pic de RAM » du banc (+4 Go) compte aussi les poids fp32 mappés depuis le disque (cache de fichiers, récupérable). Le chargement fp16 direct sur le GPU (`device_map`) demanderait la dépendance `accelerate` : non fait. |

## 4. Dans le vrai pipeline (RTX 3090)

`tests/bench_pipeline.py`, les 3 épisodes : indexation de la saison
(coupes, Search+, tags, embeddings, vignettes, proxys), 18 recherches
(dont la 1re à froid), 10 exports + 4 timelines, puis upscale x2, RIFE x2
et profondeur sur un extrait de 5 s.

### 4.1 Référence contre palier 24 Go, runs alternés

Trois runs alternés de chaque (même hôte, même heure ; médiane, puis
[min – max]). Référence = modèles actuels (SigLIP base, wd-vit,
animevideov3, RIFE 4.25, DA V2) ; palier 24 Go = so400m, wd-vit,
2xHFA2kAVCCompact, RIFE 4.26, Video Depth Anything 518 px (un des trois
runs « palier » utilisait encore DA V2 pour la profondeur).

| | référence | palier 24 Go | écart |
|---|---:|---:|---:|
| **pipeline complet** | **5 min 57** [5 min 34 – 7 min 01] | **6 min 25** [6 min 13 – 6 min 37] | **+28 s (+8 %)** |
| saison (3 épisodes) | 254,7 s [236,6 – 301,0] | 269,2 s [268,8 – 278,3] | +14,5 s |
| — étape tags | 13,2 s | 13,1 s | = |
| — étape embeddings | 9,2 s | 10,9 s | +1,7 s |
| — décodage + coupes + mouvement | 3 min 38 | 3 min 50 | +12 s (voir plus bas) |
| recherches (18) | 11,2 s | 17,4 s | +6,2 s |
| — 1re recherche, à froid | 9,7 s | 15,2 s | +5,5 s (chargement de so400m) |
| export | 15,9 s | 14,4 s | = |
| finition (upscale + RIFE + profondeur) | 77,5 s | 77,1 s | = |
| pic de RAM du sidecar | 4,7 Go | 9,0 Go | **+4,2 Go** |
| pic de VRAM | 5,4 Go | 12,5 Go | +7,1 Go (VDA 518 px) |

**Lecture.** Les nouveaux modèles de finition ne coûtent rien dans le
pipeline (77 s des deux côtés : le job est limité par le décodage, l'encodage
et le lissage CPU, pas par le modèle). Le surcoût vient de so400m :
quelques secondes de calcul, ~5 s de chargement à la 1re recherche, et
surtout son **chargement** (4,5 Go de poids fp32 lus et convertis) qui
tombe pendant le décodage de la saison et le ralentit (+12 s) — et un pic de
RAM de 9 Go. Bilan : **+8 % de temps pour +12 % de précision de recherche,
un upscale deux fois plus fidèle et une profondeur 4,8 fois plus stable.**
Le bruit de l'hôte reste du même ordre que l'écart (la référence varie de
5 min 34 à 7 min 01 sur la matinée).

### 4.2 Chaque palier dans sa carte simulée

Un run par palier, la carte réduite par le lest (le sidecar voit 8, 12 ou
16 Go et choisit ses lots et son palier « Auto » en conséquence), même
hôte, même matinée que le § 4.1 :

| run | carte | pipeline | saison | recherches (1re à froid) | finition : upscale / RIFE / profondeur | OOM / plantage | pic RAM | swap |
|---|---|---:|---:|---:|---|---|---:|---:|
| référence (médiane de 3) | 24 Go | 5 min 57 | 254,7 s | 11,2 s (9,7 s) | 17,9 / 12,7 / 44,1 s | aucun | 4,7 Go | ~0 |
| palier **8 Go** | 8 Go (lest) | 7 min 03 | 318,6 s | 15,8 s (12,4 s) | 21,8 / 12,2 / 38,0 s (VDA 364) | **aucun** | 10,5 Go | 34 pages |
| palier **12 Go** | 12 Go (lest) | 6 min 14 | 278,4 s | 14,2 s (12,0 s) | 17,7 / 13,7 / 35,0 s (VDA 364) | **aucun** | 8,9 Go | 0 |
| palier **16 Go** | 16 Go (lest) | 6 min 23 | 274,3 s | 13,5 s (10,8 s) | 18,4 / 13,4 / 48,3 s (VDA 518) | **aucun** | 8,7 Go | 0 |
| palier **24 Go** (médiane de 3) | 24 Go | 6 min 25 | 269,2 s | 17,4 s (15,2 s) | 18,8 / 13,9 / 44,5 s (VDA 518) | aucun | 9,0 Go | ~0 |
| **Rapide** | 24 Go | **5 min 34** | 246,3 s | 10,9 s (9,4 s) | 17,0 / 13,3 / **31,7 s** (VDA 364) | aucun | 4,7 Go | 0 |
| **Qualité max** | 24 Go | 7 min 04 | 324,6 s (tags 27 s, embeddings 22 s) | 14,8 s (12,4 s) | 17,8 / 15,1 / 37,6 s | aucun | 12,4 Go | 3 407 pages |

Tous les paliers passent la saison entière, les 18 recherches, les exports
et la finition **sans OOM ni plantage**, 1 045 scènes indexées à chaque
fois. À 8 Go la saison est plus lente (lots de 16 au lieu de 64, plus le
bruit de l'hôte) ; c'est le prix de la petite carte, pas des modèles
(so400m passe à 158 img/s au lot de 16, § 4.3). Le mode Rapide (modèles
actuels + VDA 364 px pour la profondeur) est **plus rapide que la
référence** (profondeur 31,7 s au lieu de 44 s) avec une profondeur 3× plus
stable. Qualité max coûte +19 % (tagger eva02 et 256 patches) et 12 Go de
RAM au pic.

### 4.3 Matrice « modèle × palier » (modèles isolés, carte simulée par le lest)

Chaque modèle seul, sur 512 images (8, 12, 16 Go : lest + lot du palier,
16 à 8 Go et 32 à 12/16 Go) ou sur les 1 045 images (24 Go, lot 64).
« OK » = aucun OOM ; débit ; pic de l'allocateur. Les modèles écartés pour
la qualité n'ont été passés qu'au palier le plus contraint (8 Go) : s'ils
passent à 8 Go, ils passent au-dessus.

| modèle | 8 Go | 12 Go | 16 Go | 24 Go |
|---|---|---|---|---|
| SigLIP 2 base 128 (actuel) | OK · 151 img/s · 0,8 Go | OK · 482 · 0,9 | OK · 415 · 0,9 | OK · 290 · 1,0 |
| SigLIP 2 base 256 | OK · 257 · 0,9 | (OK) | (OK) | OK · 231 · 1,3 |
| **SigLIP 2 so400m 128** | **OK · 158 · 2,3** | **OK · 251 · 2,4** | **OK · 250 · 2,4** | **OK · 212 · 2,5** |
| SigLIP 2 so400m 256 | OK · 135 · 2,4 | OK · 173 · 2,5 | OK · 164 · 2,5 | OK · 158 · 2,9 |
| SigLIP 2 base 384 / large 384 / so400m 384 | OK · 162 / 82 / 52 · 1,0 / 1,9 / 2,6 | (OK) | (OK) | OK · 88 / 57 / 52 · 1,6 / 2,8 / 3,7 |
| PE-Core S / B / L | OK · 65 / 129 / 67 · 0,3 / 1,0 / 1,7 | (OK) | (OK) | OK · 88 / 141 / 68 · 0,7 / 1,2 / 2,8 |
| **wd-vit (actuel)** | **OK · 221 · 0,5** | **OK · 243 · 0,5** | **OK · 243 · 0,5** | **OK · 251 · 0,8** |
| wd-swinv2 / convnext / vit-large | OK · 117 / 174 / 58 · 1,0 / 0,8 / 1,4 | (OK) | (OK) | OK · 135 / 59 / 66 · 1,7 / 1,3 / 1,7 |
| wd-eva02-large | OK · 51 · 1,6 | OK · 51 · 1,6 | OK · 52 · 1,6 | OK · 54 · 2,0 |
| PixAI v0.9 (ONNX Runtime) | OK · 19 · (arène ORT : prend la VRAM libre) | (OK) | (OK) | OK · 18 · ~20 Go réservés |
| animevideov3 / 2xHFA2kAVCCompact | (0,6 Go) | (OK) | (OK) | OK · 47 / 48 im/s · 0,6 Go |
| Real-CUGAN / x4plus anime 6B | (2,0 / 3,4 Go) | (OK) | (OK) | OK · 26 / 4,9 im/s |
| RIFE 4.25 / 4.26 / 4.26 heavy | (0,9 / 0,9 / 1,5 Go) | (OK) | (OK) | OK · 21 / 20 / 16 im/s |
| FILM | (7,0 Go réservés : limite) | (OK) | (OK) | OK · 3,4 im/s |
| Depth Anything V2 518 / 364 | (2,1 / 1,1 Go) | (OK) | (OK) | OK · 15 / 32 im/s |
| **Video Depth Anything 364** | **OK dans le pipeline (§ 4.2)** | **OK (§ 4.2)** | (OK) | OK · 66 im/s · 3,3 Go |
| Video Depth Anything 518 | **trop juste** (11,8 Go réservés au pic) | limite | **OK (§ 4.2)** | OK · 27 im/s · 11,8 Go |

Entre parenthèses : déduit du pic mesuré à 24 Go (bien sous la limite),
pas lancé dans la carte simulée.

### 4.4 Réindexation des embeddings seuls

Base indexée avec SigLIP base (copie de la base du run « référence »),
passage à so400m par `POST /api/models/reembed`
(`tests/eval/reembed_bench.py`) :

| mesure | résultat |
|---|---|
| durée (3 épisodes, 1 045 scènes) | **6 min 07** sur la 3090, dominée par la relecture des images médianes dans les vidéos (décodage complet de chaque épisode, ~2 min chacun) |
| résultat | `ok`, 1 045 scènes réembarquées, 0 vidéo manquante, bascule faite, ancien modèle archivé |
| recherche pendant / après | fonctionne avant (modèle de la base) et après (nouveau modèle) |
| retour à l'ancien modèle | **1,0 s** (vecteurs archivés, aucun décodage) |

Coupes, tags, sélections, Search+ intacts (seule la colonne d'embeddings
change). Le décodage pourrait passer par le chemin NVDEC → tenseurs de
l'indexation pour diviser cette durée (non fait).

## 5. Recommandation NVIDIA : fonction × palier

| fonction | Rapide | 8 Go | 12 Go | 16 Go | 24 Go | Qualité max |
|---|---|---|---|---|---|---|
| recherche | SigLIP 2 base, 128 patches | **so400m, 128** | **so400m, 128** | **so400m, 128** | **so400m, 128** | so400m, 256 |
| tags | wd-vit | wd-vit | wd-vit | wd-vit | wd-vit | wd-eva02-large |
| upscale | animevideov3 | **2xHFA2kAVCCompact** | **2xHFA2kAVCCompact** | **2xHFA2kAVCCompact** | **2xHFA2kAVCCompact** | 2xHFA2kAVCCompact |
| interpolation | RIFE 4.26 | RIFE 4.26 | RIFE 4.26 | RIFE 4.26 | RIFE 4.26 | RIFE 4.26 heavy |
| profondeur | **VDA Small 364 px** | VDA Small 364 px | VDA Small 364 px | **VDA Small 518 px** | VDA Small 518 px | VDA Small 518 px |

Justification case par case :

- **Recherche, 8 à 24 Go = so400m 128.** +12 % de précision@10 (0,708 →
  0,796), +10 % de nDCG@10, +21 % sur les requêtes françaises. Au lot du
  palier il passe partout : 2,3 Go de VRAM à 8 Go (lot 16, 158 img/s),
  2,5 Go à 24 Go (lot 64, 212 img/s). Dans le pipeline, l'étape
  « embeddings » passe de 9,2 à 10,9 s ; avec le chargement du modèle
  (pendant le décodage et à la 1re recherche) le palier 24 Go complet
  coûte **+28 s (+8 %)** sur 5 min 57, et +4 Go de RAM au pic, plus 3 Go de
  plus à télécharger. Rapport gain/surcoût : **+12 % de précision pour
  +8 % de temps** (et ce temps achète aussi le meilleur upscale et la
  profondeur stable, gratuits dans le pipeline) → défaut dès 8 Go. Rapide garde base (téléchargement et RAM
  minimums). Qualité max : 256 patches (+0,5 % nDCG, meilleur sur les plans
  et le français, −25 % de débit, invisible dans le pipeline).
- **Tags = wd-vit partout.** Meilleur F1 (0,69 au seuil de l'application,
  0,74 au meilleur seuil), 4 à 14× plus rapide que les autres, et aucun
  tagger ne reconnaît plus de personnages (vocabulaire identique). eva02 en
  Qualité max : Goku reconnu 9 fois sur 12 au lieu de 7, F1 équivalent à
  son seuil optimal, 5× plus lent (≈ +16 s de calcul sur 1 045 scènes).
- **Upscale = 2xHFA2kAVCCompact dès 8 Go.** LPIPS 0,044 contre 0,082
  (moitié moins d'écart à l'original), PSNR +4,4 dB, même architecture
  donc même VRAM (0,6 Go) et même débit isolé ; dans le job x2 vers la 4K
  du pipeline : 18,8 s contre 17,9 s (médianes de 3, dans le bruit). Rapide garde animevideov3 (un peu plus rapide sur le job et
  rendu « net » que certains préfèrent).
- **Interpolation = RIFE 4.26.** Même graphe que 4.25, meilleures sorties
  (calme +0,18 dB, action LPIPS 0,357 vs 0,358, une déformation de bouche
  en moins sur la planche), même vitesse : aucun coût. 4.26 heavy en
  Qualité max : +0,3 dB sur les plans calmes, menton net là où les autres
  laissent un fantôme, ~25 % plus lent.
- **Profondeur = Video Depth Anything Small.** Scintillement brut **÷ 4,8**
  (1,15 % → 0,24 %) et même après le lissage de l'application il reste
  ÷ 4,6 (0,51 % → 0,11 %), tout en étant **plus rapide** que l'actuel avec
  notre intégration (26,8 im/s à 518 px contre 14,8 ; dans le job du
  pipeline : 44,5 s contre 44,1 s à 518 px, 31,7-38 s à 364 px, le reste
  du job étant le décodage et le lissage CPU). Mais la fenêtre de
  32 images coûte de la VRAM : 11,8 Go réservés au pic à 518 px, donc 518 px
  seulement à partir de 16 Go ; 364 px (3,3 Go, 66 im/s, scintillement
  0,35 %) pour 8 et 12 Go et le mode Rapide.

Validation dans le pipeline complet, carte simulée par le lest : voir § 4
(tous les paliers passent la saison complète sans OOM ; 8 Go : 7 min 03,
12 Go : 6 min 14, 16 Go : 6 min 23, 24 Go : 6 min 25, Rapide : 5 min 34,
Qualité max : 7 min 04, référence : 5 min 57).

### Estimations pour d'autres cartes (estimées, non mesurées)

Débits isolés de la 3090 multipliés par les facteurs du § 1.5 (fourchette
basse = bande passante, haute = calcul) :

| modèle | RTX 3090 (mesuré) | RTX 4060 8 Go | RTX 3060 12 Go | RTX 4060 Ti 16 Go | RTX 4080 16 Go | RTX 4090 |
|---|---:|---:|---:|---:|---:|---:|
| so400m 128 (img/s) | 212 | 64 – 89 | 76 – 81 | 66 – 131 | 163 – 290 | 229 – 488 |
| base 128 (img/s) | 290 | 87 – 122 | 104 – 110 | 90 – 180 | 223 – 397 | 313 – 667 |
| wd-vit (img/s) | 251 | 75 – 105 | 90 – 95 | 78 – 156 | 193 – 344 | 271 – 577 |
| 2xHFA2k / animevideov3, 540p→1080p (im/s) | 48 | 14 – 20 | 17 – 18 | 15 – 30 | 37 – 66 | 52 – 110 |
| RIFE 4.26, 1080p (im/s) | 20 | 6 – 8 | 7 – 8 | 6 – 12 | 15 – 27 | 22 – 46 |
| VDA Small 364 px (im/s) | 66 | 20 – 28 | 24 – 25 | 21 – 41 | 51 – 91 | 72 – 153 |

Sur une saison de 3 épisodes, les étapes modèles (tags + embeddings, ~20 s
sur la 3090) deviendraient ~40-60 s sur une 4060 ou une 3060 ; le décodage
(NVDEC, ~2 min 30 à 3 min 50 selon la charge de l'hôte) reste du même
ordre, ce qui ferait **+10 à +20 % sur la saison** (estimé). Le choix des
modèles par palier ne change pas : so400m n'ajoute que ~3 s de calcul
estimé sur ces cartes.

## 6. Apple Silicon (MacBook Pro M1 Pro 16 Go)

Mêmes jeux d'évaluation, mêmes images, mêmes candidats retenus. PyTorch
MPS pour SigLIP, VDA et l'upscale ; Core ML (GPU) pour le tagger, RIFE et
Depth Anything V2, comme dans l'application ; Neural Engine pour une partie
de l'upscale. Copie de travail séparée (`~/amv-tools-test/models-repo`,
`uv sync --extra mps --group dev`), sans toucher au travail du tour Mac.

### 6.1 Équivalence avec la sortie CUDA

| fonction | mesure | résultat |
|---|---|---|
| recherche (base et so400m, 128 et 256) | cosinus image Mac/CUDA, même image | min 0,9977, médiane 0,99996 ; textes ≥ 0,9995 |
| | recouvrement du top 10 des 25 requêtes | 98,0 – 99,2 % |
| tags (vit, swinv2, convnext) | top 5 identique / ensembles au seuil 0,35 | 98,4 – 99,3 % / Jaccard 0,976 – 0,990 |
| upscale animevideov3 | PSNR / SSIM vs original | 38,08 dB / 0,975 (CUDA 38,09 / 0,975) |
| interpolation RIFE 4.25 / 4.26 / heavy (Core ML) | PSNR plans calmes | 45,23 / 45,42 / 45,69 dB (CUDA 45,19 / 45,37 / 45,65) |
| profondeur DA V2 518 (Core ML) | scintillement brut / lissé | 1,10 % / 0,49 % (CUDA 1,15 % / 0,51 %) |
| upscale 2xHFA2kAVCCompact | PSNR / SSIM vs original | 42,51 dB / 0,980 (CUDA 42,50 / 0,980) |
| profondeur Video Depth Anything 364 (MPS) | scintillement brut / aligné / lissé | 0,354 / 0,230 / 0,155 % (CUDA 0,350 / 0,231 / 0,157 %) |

Les sorties sont équivalentes : les choix de qualité faits sur la 3090
valent pour le Mac.

### 6.2 Vitesse et mémoire des candidats (isolés)

| modèle | moteur | débit | chargement | mémoire GPU (Metal) | empreinte du processus |
|---|---|---:|---:|---:|---:|
| SigLIP 2 base 128 (actuel) | MPS | 92,6 img/s (3 runs : 92,5 / 92,9 / 92,6) | 4 s | 1,4 Go | 2,9 Go |
| SigLIP 2 base 256 | MPS | 53 img/s | 3 s | 1,4 Go | 2,6 Go |
| SigLIP 2 so400m 128 | MPS | 27,4 img/s (27,4 / 27,4 / 27,5) | 3-6 s | 2,3 Go | 3,7 Go (pic au chargement 2,9 Go) |
| SigLIP 2 so400m 256 | MPS | 14 img/s | 4 s | 3,3 Go | 4,8 Go |
| wd-vit (actuel) | MPS / Core ML GPU | 16,7 (3 runs identiques) / 22,3 img/s | 3-14 / 18 s | 1,7 / 0,2 Go | 2,6 / 1,2 Go |
| wd-convnext | MPS | 19,5 img/s | 3 s | 1,2 Go | 2,3 Go |
| wd-swinv2 | MPS | 9,0 img/s | 4 s | 1,2 Go | 2,4 Go |
| wd-eva02-large | MPS / Core ML GPU | 2,9 / 5,0 img/s | 9 / 65 s | 2,6 / 1,6 Go | 3,3 Go |
| animevideov3 540p→1080p | MPS | 7,2 im/s | – | 1,1 Go | 1,4 Go |
| 2xHFA2kAVCCompact 540p→1080p | MPS | 7,6 im/s | – | 1,1 Go | 1,4 Go |
| RIFE 4.25 / 4.26 / 4.26 heavy, 1080p | Core ML GPU | 6,8 / 6,7 / 5,6 im/s | – | 2,1 / 2,1 / 3,1 Go | 2,9 / 2,9 / 3,9 Go |
| Depth Anything V2 Small 518 | Core ML GPU | 7,7 im/s | 2 s | 1,4 Go | 3,0 Go |
| Video Depth Anything 364 | MPS (fp32) | **13,6 im/s** | 0,5 s | 6,4 Go | **8,0 Go** |
| Video Depth Anything 518 | MPS (fp32) | **abandonné** : la machine s'est mise à swapper (≈ 10 Go de swap), processus bloqué | | | |

Core ML, unités de calcul : pour le tagger, CPU_AND_GPU = 22,3 img/s
contre 16,7 en MPS (×1,34) ; ALL et CPU_AND_NE **n'ont pas abouti** (la
compilation pour le Neural Engine n'était pas terminée après 15 min, voir
Limites). SigLIP NaFlex n'a pas de chemin Core ML (formes dynamiques) ;
VDA non plus (fenêtres de 32 images), ils restent en MPS.

**so400m coûte bien plus cher sur le Mac que sur la 3090** : 3,4× plus lent
que base (27 contre 92 img/s), parce que l'étape n'est plus masquée par le
décodage.

### 6.3 Pipeline complet sur le Mac

Même banc que sur la 3090 (3 épisodes, 18 recherches, exports, finition).
**Attention** : pendant les premiers runs, le Neural Engine était bloqué
par un `ANECompilerService` resté à 100 % depuis une autre session ;
l'upscale n'a alors utilisé que le GPU (76 s au lieu de 37 s). Les
comparaisons ci-dessous n'utilisent que des runs où le Neural Engine
répondait (vérifié dans le journal : « 68 of the frames on the Neural
Engine »), sauf la ligne « so400m » qui documente le swap.

| configuration | pipeline | saison | recherches (1re à froid) | upscale / RIFE / profondeur | pic RAM | swap |
|---|---:|---:|---:|---|---:|---:|
| référence (modèles actuels), 2 runs\* | **3 min 03** (182,9 / 182,3 s) | 101,0 s | 4,2 s (3,7 s) | 37,2 / 12,7 / 21,2 s | 6,1 Go | 0 |
| **palier Mac 16 Go** (base, wd-vit, HFA2k, RIFE 4.26, VDA 364), 3 runs | **2 min 53** (179,9 / 173,0 / 172,9 s) | 101,1 s | 4,1 s (3,6 s) | 35,4 / 12,8 / **13,6 s** | 7,6 Go | 0 / 3 214 / 0 pages |
| so400m au lieu de base (ancien palier 16 Go), 3 runs, Neural Engine bloqué | 4 min 08 (247,8 / 247,6 / 246,3 s) | **126,2 s** | 7,9 s (6,7 s) | 72,8 / 12,8 / 21,1 s | 9,6 Go | **70 632 / 11 884 / 27 232 pages** |
| référence, Neural Engine bloqué, 3 runs | 3 min 50 | 108,7 s | 4,3 s (3,8 s) | 75,8 / 12,8 / 21,2 s | 5,6 Go | 0 |
| palier Mac 8 Go (simulé : Metal plafonné à ~5,3 Go) | 3 min 39 | 101,4 s | 3,5 s (3,0 s) | 74,3 / 12,5 / 20,9 s (tuilage → pas de Neural Engine) | 5,6 Go | 0 |
| Rapide | 2 min 53 | 100,8 s | 4,3 s (3,8 s) | 37,3 / 12,5 / 13,2 s (DA V2 364) | 5,9 Go | 0 |
| Qualité max (so400m 256 + eva02 à ce moment-là) | 6 min 49 | **331,0 s** | 7,0 s | 35,5 / 15,7 / 13,7 s | 7,7 Go | 0 |

\* un 3e run de référence s'est arrêté sur un délai d'attente HTTP du banc
après la saison (sans lien avec les modèles, aucun modèle nouveau dans
cette configuration).

**Lecture.** Le palier Mac 16 Go retenu est **5 % plus rapide** que la
référence (2 min 53 contre 3 min 03) : Video Depth Anything à 364 px
remplace Depth Anything V2 et divise la profondeur par 1,6 en temps et
par 3 en scintillement, l'upscale HFA2k (x2 natif, Neural Engine) est un peu
plus rapide que animevideov3. so400m, en revanche, ajoute **+25 s à la
saison** (127 contre 101 s) et a fait **swapper** la machine de 16 Go à
chaque run (jusqu'à 70 000 pages) : il est réservé aux Mac de 24 Go et
plus. Qualité max avec eva02 et 256 patches triple la saison : d'où le
choix de garder wd-vit en Qualité max sur Mac.

### 6.4 Paliers Mac simulés et extrapolés

| palier | modèles | mesuré / extrapolé | résultat |
|---|---|---|---|
| Mac 8 Go | base, wd-vit, HFA2k, RIFE 4.26, DA V2 518 | **simulé** (Metal plafonné à 0,5× l'ensemble de travail recommandé ≈ 5,3 Go, `AMV_VRAM_LIMIT_GB=8`) | passe, 3 min 39, empreinte max 5,6 Go, pas de swap ; Metal plafonné : SigLIP base 42 img/s, so400m 28 img/s (passe aussi, empreinte 3,4 Go, mais trop lent pour les puces de base) |
| Mac 16 Go | base, wd-vit, HFA2k, RIFE 4.26, VDA 364 | **mesuré** (machine réelle) | passe, 2 min 53, pic 7,6 Go ; so400m : swap à chaque run → exclu |
| Mac 24 Go | so400m, wd-vit, HFA2k, RIFE 4.26, VDA 364 | **extrapolé** | pic mesuré 9,6 Go avec so400m sur la machine de 16 Go → ~14 Go de marge sur 24 Go ; saison estimée ~2 min 05 sur un M1 Pro (+25 s) |
| Mac 32 Go et plus | idem 24 Go | **extrapolé** | VDA à 518 px non retenu : il a saturé la machine de 16 Go (≈ 10 Go de swap) ; à tester sur 32 Go+ |

### 6.5 Recommandation Mac : fonction × mémoire unifiée

| fonction | Rapide | Mac 8 Go | Mac 16 Go | Mac 24 Go | Mac 32 Go + | Qualité max |
|---|---|---|---|---|---|---|
| recherche | base 128 | base 128 | base 128 | **so400m 128** (extrapolé) | so400m 128 (extrapolé) | so400m 256 |
| tags | wd-vit (Core ML) | wd-vit | wd-vit | wd-vit | wd-vit | wd-vit (eva02 : 2,9-5 img/s) |
| upscale | animevideov3 | **2xHFA2kAVCCompact** | 2xHFA2kAVCCompact | 2xHFA2kAVCCompact | 2xHFA2kAVCCompact | 2xHFA2kAVCCompact |
| interpolation | RIFE 4.26 | RIFE 4.26 | RIFE 4.26 | RIFE 4.26 | RIFE 4.26 | RIFE 4.26 heavy |
| profondeur | DA V2 364 (Core ML) | DA V2 518 (Core ML, 3 Go) | **VDA 364** (8 Go d'empreinte pendant le job) | VDA 364 | VDA 364 | VDA 364 |

Justification : HFA2k et RIFE 4.26 sont plus fidèles sans coût mesurable
(mêmes sorties qu'en CUDA) ; VDA 364 est plus rapide et 3× plus stable que
DA V2 518 mais son empreinte de 8 Go l'écarte des Mac de 8 Go ; so400m
apporte +12 % de précision mais coûte +25 % sur la saison et fait swapper
16 Go → seulement à partir de 24 Go.

### 6.6 Autres puces Apple (estimé)

Estimations à partir du M1 Pro (16 cœurs GPU, 200 Go/s), au prorata du
nombre de cœurs GPU, de la fréquence et de la bande passante ; **non
mesuré** :

| puce | GPU | bande passante | facteur estimé vs M1 Pro | conséquence |
|---|---:|---:|---:|---|
| M1 / M2 / M3 (base) | 7-10 cœurs | 68-100 Go/s | 0,4 – 0,6 | so400m trop lent (~11-16 img/s) : palier 8 Go = base |
| M2 Pro / M3 Pro | 16-19 cœurs | 150-200 Go/s | 1,0 – 1,3 | comme le M1 Pro |
| M4 Pro | 16-20 cœurs | 273 Go/s | 1,4 – 1,8 | so400m confortable |
| M1/M2/M3 Max, M4 Max | 30-40 cœurs | 400-546 Go/s | 1,8 – 3,0 | so400m 256 envisageable |
| M5 (base) | 10 cœurs + accélérateurs neuronaux dans le GPU | 153 Go/s | 0,7 – 1,2 (MPS) | à mesurer : les accélérateurs ne sont exploités que par Core ML / MLX récents |

## 7. Ce qui a été intégré

| quoi | où |
|---|---|
| Catalogue des paliers, un tableau par plateforme (NVIDIA, Mac, CPU/DirectML/XPU), sélection « Auto » (VRAM sur NVIDIA, mémoire unifiée totale sur Mac, seuils tolérants : 7,6 / 11,6 / 15,6 / 23,6 Go tombent bien sur 8 / 12 / 16 / 24), modes Rapide, Qualité max et Personnalisé | `backend/models/tiers.py`, réglage `models.tier` (« auto » par défaut) |
| Sélecteur « Palier de modèles » dans Settings > Models : libellé « Auto (détecté : 24 Go → 24 Go) », tableau des modèles du palier (taille, licence), anciens menus grisés hors « Personnalisé », textes FR/EN | `app/src/components/ModelTiersSection.tsx`, `app/src/i18n/strings.models.ts`, `app/src/modelTiers.ts` |
| Embeddings **versionnés par modèle** : chaque base enregistre son modèle d'embedding ; table `scene_embeddings(scene_id, model, embedding)` (ajout idempotent, sans changer la version du schéma) ; la recherche, les plans similaires et les personnages Search+ utilisent **toujours le modèle de la base** ; une nouvelle vidéo indexée dans une base existante prend le modèle de la base (jamais deux espaces mélangés) | `backend/indexing/reembed.py`, `backend/state.py` |
| **Réindexation des embeddings seuls** : bandeau dans Settings quand le palier veut un autre modèle que celui de la base, bouton, progression (websocket + suivi), annulation ; image médiane de chaque scène relue dans la vidéo (512×288 comme l'indexation), coupes, tags et sélections intacts ; bascule en une transaction à la fin ; les anciens vecteurs sont archivés (revenir en arrière est instantané) ; vidéos introuvables signalées, bascule refusée sauf « forcer » | `backend/api/reembed.py` (`/api/models/reembed`), `ModelTiersSection.tsx` |
| Téléchargement **à la demande avec contrôle de licence** : liste blanche (Apache-2.0, MIT, BSD, CC-BY-4.0, CC0) vérifiée sur la fiche Hugging Face avant le premier téléchargement, et sur le registre des modèles de finition | `backend/model_download.py` |
| Finition : modèle par défaut = celui du palier ; nouveaux modèles **2xHFA2kAVCCompact** (x2 natif, x4 en deux passes, Neural Engine sur Mac), **RIFE 4.26 / 4.25 lite / 4.26 heavy** (un seul graphe paramétré, Core ML sur Mac), **Video Depth Anything Small** (réseau Apache-2.0 copié dans `archs/vda`, flux sur GPU à la résolution du modèle, estimation de durée adaptée), taille d'entrée de la profondeur par palier | `backend/finish/registry.py`, `upscale.py`, `interpolate.py`, `vda.py`, `archs/rife.py`, `archs/vda/`, `ane.py`, `coreml.py`, `runner.py` |
| Corrections trouvées en mesurant : wd-swinv2 sur CUDA (tags vides), programme Core ML du tagger partagé entre modèles | `backend/models/wd_tagger.py`, `backend/finish/coreml.py` |
| Outils de mesure : `AMV_VRAM_LIMIT_GB` (carte plus petite), `bench_pipeline.py --settings / --finish-models`, harnais `tests/eval/` + jugements | `backend/models/device.py`, `tests/bench_pipeline.py`, `tests/eval/` |
| Tests : sélection des paliers (NVIDIA 6 à 48 Go, Mac 8 à 128 Go, CPU, simulation, palier d'une autre plateforme, Personnalisé), tableaux mesurés, modèles enregistrés et licenciés, réindexation de bout en bout (factice), architectures RIFE et upscaler x2, flux VDA (1 à 100 images : une carte par image, dans l'ordre) | `tests/test_model_tiers.py` (56 tests), `app/src/modelTiers.test.ts` |

**Non intégré** (mesuré, perdant ou pas assez gagnant) : PE-Core, SigLIP 2
à résolution fixe, PixAI, wd-swinv2/convnext/vit-large, Real-CUGAN,
ModernSpanimation, FILM. RIFE 4.25 lite est installable depuis la finition mais n'est dans aucun palier.

**Migration des installations existantes.** Le palier par défaut est
« Auto ». Une installation qui avait choisi elle-même un autre modèle que
les défauts passe une fois en « Personnalisé » (son choix est gardé). Une
base indexée avec SigLIP base continue à chercher avec SigLIP base ; le
bandeau propose la réindexation (1 045 scènes en 6 min 07 sur la 3090 hors
téléchargement, dominé par la relecture des images dans les vidéos ; retour
à l'ancien modèle en 1 s).

## 8. Limites et points ouverts

- **Bruit de mesure.** La VM NVIDIA partage son hôte Proxmox : la même
  référence a donné 5 min 08 puis 7 min 01 de pipeline à une heure
  d'intervalle (téléchargements en cours la 1re heure). D'où les runs
  alternés référence / palier du § 4 et des écarts, pas des runs isolés.
  Sur le Mac, deux processus étrangers tournaient pendant toute la session
  (un `ANECompilerService` bloqué à 100 % d'un cœur depuis 1 h du matin et
  Discord) et 3,4 → 5,1 Go de swap étaient déjà occupés avant nos runs.
- **Cartes simulées, pas réelles.** Le lest donne la bonne quantité de VRAM
  libre, mais pas la puissance ni la bande passante d'une 4060 ou d'une
  3060 : leurs débits sont **estimés** (§ 5). Le pilote d'une vraie carte
  de 8 Go qui affiche aussi le bureau laisse ~0,5-1 Go de moins.
- **Mac 24 et 32 Go : extrapolés** (machine de 16 Go). Les M2/M3/M4/M5
  sont **estimés** à partir du M1 Pro (§ 6).
- **Core ML « ALL » et « CPU_AND_NE » non mesurés pour le tagger** : la
  compilation pour le Neural Engine n'a pas abouti en 15 min (le
  compilateur ANE du système était déjà bloqué par une autre session).
  CPU_AND_GPU est mesuré (et c'est ce qu'utilise l'application).
- **Jeux d'évaluation modestes** : 25 requêtes, 60 scènes, 36 images,
  18 plans. Les écarts de 1-2 % (256 vs 128 patches, 4.26 vs 4.25) sont
  dans le bruit ; les écarts retenus pour les choix (so400m +12 %, HFA2k
  LPIPS ÷ 1,9, VDA scintillement ÷ 4,8) sont nets. Black Clover n'a pas
  de vérité « personnages » (l'annotateur n'a pas su nommer les
  personnages de cette nouvelle saison avec certitude). Un seul
  annotateur, pas de double jugement.
- **Personnages récents** : aucun tagger ouvert ne connaît Beerus, Whis ou
  la nouvelle distribution de Black Clover. La réponse est la recherche
  par le nom (so400m) et les personnages Search+ ; un tagger entraîné sur
  un Danbooru récent reste à trouver (cl_tagger, licence Apache-2.0, non
  mesuré).
- **Non mesurés faute de temps** (licence OK) : EMA-VFI, IFRNet, VFIMamba
  (interpolation), DA3 Small/Base, Distill-Any-Depth, MoGe-2 (profondeur),
  cl_tagger, jina-clip-v1, nomic-embed-vision, TIPS (recherche), 2x-Fallin
  (upscale). FILM est mesuré (§ 3.4) et rejeté.
- **Upscale** : le protocole « 540p → 1080p » favorise les modèles
  fidèles ; un monteur qui veut des traits « re-dessinés » peut préférer
  animevideov3 ou Real-CUGAN (toujours disponibles via Personnalisé ou le
  menu de la finition).
- **VDA** : fenêtre de 32 images → latence de 32 images avant la 1re
  sortie et VRAM élevée à 518 px ; non mesuré sur CPU (le palier CPU garde
  Depth Anything V2). Sur les plans noirs, tous les modèles inventent un
  « vignettage » de profondeur (cohérent d'une image à l'autre, mais sans
  sens).
- **so400m et la RAM** : poids fp32 de 4,5 Go chargés puis convertis, d'où
  un pic de RAM du sidecar de +1,7 Go sur la 3090 et du swap sur le Mac
  16 Go (§ 6). Convertir une fois les poids en fp16 sur disque diviserait
  ce pic par deux (non fait).
- **Profondeur, piste ouverte** : avec VDA, le lissage par flot optique de
  l'application (CPU, ~0,08 s par image, la moitié du job) apporte peu
  (0,14 % → 0,11 %) ; le désactiver par défaut pour VDA rendrait le job
  ~2× plus rapide. Non fait (la boîte de dialogue envoie toujours 0,5).
- **Neural Engine du Mac** : les premiers runs ont tourné sans lui (service
  système bloqué) ; seuls les runs où il répondait servent aux
  comparaisons (§ 6.3).
- **Tests sur le PC RTX 3090** : `uv run pytest -m "not slow"` donne 257 succès
  et 3 échecs (`test_sub_segmentation_streams_from_the_single_decode`,
  `test_upscale_job_mocked_model_keeps_audio`, `test_slowmo_audio_stretch`)
  qui échouent **déjà sur le commit de départ** sur cette machine (ffmpeg
  5.1 du système, décodage NVDEC réel) : aucun nouvel échec. Un 4e test
  (`test_status_reports_explicit_gpu_fallback`) a échoué une fois sur deux
  runs complets (passe seul et au 2e run) : instable, à surveiller. Sur le
  VPS : 260 succès, 0 échec.

## 9. Planches visuelles et données brutes

Hors du dépôt, sur la machine de test :

| fichier | contenu |
|---|---|
| `<données>/reports/c4-upscale-planche.png` | 3 extraits 1080p de Black Clover : original, bicubique, animevideov3, 2xHFA2kAVCCompact, Real-CUGAN conservative, x4plus anime 6B, ModernSpanimation V2, animevideo xsx2 |
| `<données>/reports/c4-upscale-planche-small.jpg` | la même, réduite |
| `<données>/reports/c4-interp-planche.png` | 3 images intermédiaires d'action : image avant / vraie / après, fondu, RIFE 4.25, 4.26, 4.26 heavy, 4.25 lite |
| `<données>/reports/c4-depth-planche.png` | 2 plans calmes, début/milieu/fin : Depth Anything V2 518 et 364, Video Depth Anything 518 et 364 (après lissage de l'application) |
| `<données>/c4/w/` | images médianes des 1 045 scènes, embeddings et scores de tags de chaque modèle (`emb_*.npz`, `tags_*.npz`), mesures brutes (`*_runs.jsonl`), scores (`search_scores.json`, `tag_scores.json`), sorties d'upscale / d'interpolation / de profondeur |
| `<données>/c4/bench/` | runs du pipeline complet (`c4-*.json` / `.md`) |
| `~/amv-tools-test/c4/` (Mac) | mêmes mesures sur le M1 Pro (`w/`, `bench/m-*`) |

Reproduire : `tests/eval/eval_models.py` et `tests/eval/eval_finish.py`
(aide en tête de fichier), `tests/eval/report_tables.py <dossier>` pour les
tableaux, `tests/bench_pipeline.py --settings '{"models":{"tier":"24gb"}}'`
pour un palier dans le pipeline, `AMV_VRAM_LIMIT_GB=8` + le lest
(`tests/eval/ballast.py <Go>` lancé à côté) pour une carte de
8 Go.
