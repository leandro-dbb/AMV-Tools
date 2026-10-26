# Chantier 4b — réévaluer les modèles écartés pour leur seule licence

Branche `jarvis/v05-models-nc` (depuis `main`). Mesures du 25 octobre 2026
au soir sur le **PC RTX 3090** du chantier 4 (RTX 3090 24 Go, VM Proxmox
6 vCPU, torch 2.11 + cu126). Le **Mac M1 Pro n'était pas joignable** (une
tentative de connexion, délai dépassé) : rien n'a été mesuré sur Apple
Silicon, les choix Mac sont déduits (voir § 6).

**Décision de départ** : AMV Tools n'est pas commercial, les licences non
commerciales (CC-BY-NC, CC-BY-NC-SA, S-Lab, SAM License…) sont autorisées,
à trois conditions : poids jamais committés ni empaquetés (téléchargés à la
demande depuis la source officielle), licence affichée dans Settings >
Models et dans le README, mention « usage non commercial » visible.

## En bref

- **Upscale : AnimeJaNai HD V3 UltraCompact est intégré** et remplace
  2xHFA2kAVCCompact à tous les paliers à partir de 8 Go et en Qualité max,
  ainsi qu'animevideov3 en mode Rapide (NVIDIA et Mac). Dans le pipeline, il
  est plus rapide qu'HFA2k dans chaque paire de runs alternés. Sur les 36 images 1080p du chantier 4, il est **un tiers
  plus proche de l'original** (LPIPS 0,030 contre 0,044) et gagne +3,4 dB
  sur les plans sombres (48,7 contre 45,3 dB). Il n'a pas le halo cyan
  qu'HFA2k laisse autour des traits sombres, et il fait deux fois moins de
  calcul (8 convolutions au lieu de 16). Téléchargement : 24,7 Mo (archive
  officielle), 2,5 Mo gardés sur le disque. Licence CC-BY-NC-SA-4.0.
- **Profondeur : Video Depth Anything Large à 364 px est intégré en
  Qualité max (NVIDIA)**. Il scintille le moins (0,208 % contre 0,242 % pour
  Small 518 px, soit −14 %) et, sur les planches, donne nettement plus de
  relief (visage, oreille, mèches). Il demande 9,3 Go de VRAM et 1,5 Go de
  téléchargement, et il manque de mémoire à 518 px même sur 24 Go. Licence
  CC-BY-NC-4.0.
- **Rejetés après mesure**, avec le chiffre :
  - AnimeJaNai V3.1 : pas mieux que V3 (LPIPS 0,028 contre 0,030), ONNX
    seulement ;
  - GIMM-VFI : pas meilleur que RIFE 4.26 sur l'action (LPIPS 0,358 contre
    0,357), 22× plus lent, demande CuPy ;
  - AMT S/L/G : moins bon en perception (LPIPS action 0,39-0,43 contre
    0,357), 8 à 14× plus lent, 14-18 Go de VRAM ;
  - VDA Base : damier sur un plan sombre, et Large fait mieux là où la
    mémoire suffit ;
  - PixAI tagger v1.0 : ne connaît toujours ni Beerus ni Whis, F1 des tags
    plus bas (0,67 contre 0,69), environ 50× plus lent que wd-vit. Il
    reconnaît Asta (7 détections sur 7 justes) et Cell.
- **Licences** : la liste d'autorisation accepte maintenant CC-BY-NC-4.0 et
  CC-BY-NC-SA-4.0 ; Settings > Models affiche un badge « usage non
  commercial » à côté de ces modèles, dans le tableau des paliers comme dans
  la liste des modèles de finition ; le README les liste avec leur licence.

## 1. Méthode

C'est **exactement le protocole du chantier 4**
(`docs/reports/chantier-4-modeles.md`, § 1), avec les mêmes jeux et les
mêmes scripts :

| fonction | jeu (identique au chantier 4) | script |
|---|---|---|
| upscale | 36 images 1080p des 3 épisodes (dont 4 très sombres) réduites en 540p, remontées x2, comparées à l'original : PSNR et SSIM (luminance), LPIPS (AlexNet), PSNR des plans sombres, planche visuelle | `tests/eval/eval_finish.py upscale` |
| interpolation | 12 plans (6 calmes, 6 d'action), 13 images 1080p : une image sur deux retirée puis régénérée ; images **dupliquées** (animation « en deux ») comptées à part ; en plus, paire d'images identiques (ralenti entre deux images source égales) | `eval_finish.py interp` + contrôle des doublons |
| profondeur | scintillement sur 18 plans calmes/fixes (216 paires), brut, aligné et après le lissage de l'application ; vitesse sur ~117 images d'action ; planches | `eval_finish.py depth` |
| tags | 60 scènes annotées à la main (36 apparitions de personnages, 21 tags visuels), seuils de l'application (0,35 / 0,85) ; en plus les seuils conseillés par la fiche et le meilleur seuil ; détections de personnages sur Black Clover vérifiées à l'œil | `eval_models.py tag` + balayage des seuils |
| pipeline | section « finition » de `tests/bench_pipeline.py` (upscale x2, RIFE x2, profondeur sur un extrait 1080p de 5 s), runs **alternés** référence / candidat | `bench_pipeline.py --sections finish` |

Les paliers NVIDIA 8 / 12 / 16 Go sont simulés en plafonnant l'allocateur
PyTorch (`--vram-gb`) : la méthode du lest (`ballast.py`) n'a **pas** été
utilisée, car le banc de détourage d'un autre chantier tournait sur la même
carte et le lest l'aurait privé de mémoire.

**Conditions de mesure.** La 3090 était partagée pendant toute la session
avec ce banc de détourage (2,7 à 9 Go de VRAM et un cœur CPU en
permanence). Les débits absolus sont donc plus bas qu'au chantier 4 : HFA2k
34 im/s au lieu de 48, RIFE 4.26 14,6 au lieu de 20, VDA Small 518 px 16 au
lieu de 27. **Les modèles de référence ont été remesurés dans la même
session**, et toutes les comparaisons de vitesse se font entre runs de cette
session. Les métriques de qualité ne dépendent pas de la charge : elles
reproduisent le chantier 4 au centième (HFA2k 42,50 dB / LPIPS 0,0445 ;
RIFE 4.26 45,37 dB ; VDA Small 518 0,242 % ; wd-vit F1 0,685, Goku 7/12).

## 2. Candidats, sources et tailles

Licences vérifiées sur la fiche et le dépôt officiels, poids téléchargés à
10 Mo/s (`curl --limit-rate 10M`) depuis la source officielle, révision
épinglée.

| candidat | source officielle | licence | téléchargement | ≈ temps à 10 Mo/s |
|---|---|---|---:|---:|
| AnimeJaNai HD V3 (Compact, UltraCompact, SuperUltraCompact, Sharp1) | archive « ModelsOnly » de la release 3.0.0 de mpv-upscale-2x_animejanai | CC-BY-NC-SA-4.0 | 24,7 Mo (toutes variantes) ; fichiers 4,8 / 2,5 / 0,4 Mo | 2,5 s |
| AnimeJaNai HD V3.1 (Balanced, Performance, Sharp1 ; architecture SPAN, ONNX fp16 seulement) | archive « overlay » de la release 3.8.0 | CC-BY-NC-SA-4.0 | 39,3 Mo (archive) ; fichiers 2,0 / 0,7 Mo | 4 s |
| GIMM-VFI-R-P (+ RAFT) | Hugging Face GSean/GIMM-VFI | S-Lab 1.0 (non commerciale) | 79,3 + 21,1 Mo, + CuPy (~100 Mo de dépendance) | 10 s |
| AMT-S / L / G | Hugging Face lalala125/AMT | CC-BY-NC-SA-4.0 (fiche) / CC-BY-NC-4.0 (dépôt) | 12,0 / 51,8 / 122,6 Mo | 1 / 5 / 12 s |
| Video Depth Anything Base | Hugging Face depth-anything | CC-BY-NC-4.0 | 458 Mo | 46 s |
| Video Depth Anything Large | Hugging Face depth-anything | CC-BY-NC-4.0 | 1 538 Mo | 2 min 34 |
| PixAI tagger v1.0 | Hugging Face pixai-labs | Apache-2.0 sur la fiche, dérivé de SAM 3 (SAM License, non commerciale) | 1 945 Mo | 3 min 15 |

« AnimeJaNai V3.1 » n'existe qu'en ONNX, avec une autre architecture (SPAN)
que les Compact de V3. Il a été mesuré via ONNX Runtime (CUDA).

## 3. Résultats par fonction (RTX 3090)

### 3.1 Upscale — **AnimeJaNai V3 UltraCompact gagne nettement**

36 images 1080p réduites en 540p puis remontées x2 ; LPIPS : plus bas = plus
proche à l'œil. Débit sous charge partagée (comparer entre lignes, pas au
chantier 4).

| modèle | PSNR (dB) | SSIM | LPIPS | PSNR plans sombres | débit 540p→1080p | VRAM réservée | licence |
|---|---:|---:|---:|---:|---:|---:|---|
| realesr-animevideov3 (Rapide) | 38,09 | 0,975 | 0,082 | 44,1 | 35 im/s | 0,57 Go | BSD-3 |
| **2xHFA2kAVCCompact (gagnant du chantier 4)** | 42,50 | 0,980 | 0,0445 | 45,3 | 34 im/s | 0,56 Go | CC-BY-4.0 |
| AnimeJaNai V3 Compact | **42,73** | 0,985 | 0,0296 | **48,9** | 43 im/s | 0,56 Go | CC-BY-NC-SA |
| **AnimeJaNai V3 UltraCompact** | 42,72 | **0,985** | 0,0298 | 48,7 | **56-75 im/s** | 0,56 Go | CC-BY-NC-SA |
| AnimeJaNai V3 SuperUltraCompact | 41,75 | 0,983 | 0,0399 | 47,4 | 100 im/s | 0,55 Go | CC-BY-NC-SA |
| AnimeJaNai V3 Sharp1 Compact | 42,31 | 0,984 | **0,0269** | 47,0 | 35 im/s | 0,56 Go | CC-BY-NC-SA |
| AnimeJaNai V3.1 Balanced (SPAN, ONNX) | 42,68 | 0,985 | 0,0284 | 48,8 | 40 im/s | – | CC-BY-NC-SA |
| AnimeJaNai V3.1 Performance (SPAN, ONNX) | 42,60 | 0,985 | 0,0287 | 48,5 | 75 im/s | – | CC-BY-NC-SA |
| AnimeJaNai V3.1 Sharp1 Balanced (SPAN, ONNX) | 42,44 | 0,984 | 0,0288 | 46,2 | 39 im/s | – | CC-BY-NC-SA |

**Lecture.** Tous les AnimeJaNai HD sont plus fidèles qu'HFA2k : LPIPS
0,027-0,030 contre 0,044 (−33 %), SSIM 0,985 contre 0,980, et +3,4 dB sur
les 4 images très sombres (ils sont entraînés sur de l'anime 1080p).
UltraCompact donne **la même qualité que Compact** (écart de 0,0002 en
LPIPS, dans le bruit) avec deux fois moins de convolutions, d'où un débit
1,3 à 1,7× plus élevé. V3.1 n'apporte rien de mesurable par rapport à V3
(0,028 contre 0,030) : il n'existe qu'en ONNX fp16, dans une autre
architecture, et demanderait un second moteur d'inférence pour l'upscale.
Les variantes « Sharp1 » ont le meilleur LPIPS mais un PSNR plus bas, surtout
sur les plans sombres (traits plus appuyés que l'original). SuperUltraCompact
reste meilleur qu'HFA2k (0,040) tout en étant 3× plus rapide, mais
UltraCompact est déjà plus rapide que les modèles actuels.

Planche jugée (3 extraits de Black Clover, zoom ×2) : HFA2k laisse un léger
**halo cyan** le long des traits sombres et adoucit un peu les aplats ;
AnimeJaNai V3 UltraCompact est propre et le plus proche de l'original ; les
variantes Sharp1 épaississent un peu les traits ; animevideov3 lisse les
aplats et noircit les traits (comme au chantier 4).

Intégration vérifiée : le chemin de l'application (`Upscaler`,
`SRVGGNetCompact` à 8 convolutions) redonne exactement le même PSNR que le
chargeur d'évaluation (42,7172 dB), et il passe sans erreur sous un plafond
de 8, 12 et 16 Go (0,15 Go réservés, image entière en 540p).

### 3.2 Interpolation — **RIFE 4.26 reste**

12 plans, une image sur deux régénérée (t = 0,5) ; « doublons » = l'image
retirée répétait sa voisine (animation « en deux »).

| modèle | débit 1080p (sous charge) | calme : PSNR / LPIPS | action : PSNR / SSIM / LPIPS | doublons : PSNR | paire identique : PSNR | VRAM max | licence |
|---|---:|---|---|---:|---:|---:|---|
| **RIFE 4.26** | **14,6 im/s** | 45,37 / **0,0117** | 21,02 / 0,802 / 0,357 | 38,6 | **> 80 dB** (recopie exacte) | 0,9 Go | MIT |
| RIFE 4.26 heavy (Qualité max) | 14,6 im/s | 45,65 / 0,0119 | 21,05 / 0,803 / **0,356** | 38,7 | > 83 dB | 1,5 Go | MIT |
| AMT-S | 1,8 im/s | 46,52 / 0,0176 | 21,00 / 0,808 / 0,394 | 38,6 | – | 14,7 Go | CC-BY-NC |
| AMT-L | 1,4 im/s | **47,23** / 0,0177 | 21,02 / 0,811 / 0,406 | **39,4** | 70-80 dB | 15,4 Go | CC-BY-NC |
| AMT-G | 1,0 im/s | 47,20 / 0,0178 | 20,85 / 0,811 / 0,427 | 39,4 | – | 18,3 Go | CC-BY-NC |
| GIMM-VFI-R-P, flot à ½ résolution | 0,67 im/s | 46,79 / 0,0185 | **21,09** / 0,799 / 0,358 | 39,1 | non mesuré | 6,3 Go | S-Lab 1.0 |
| GIMM-VFI-R-P, pleine résolution | **OOM** (> 12 Go à côté d'un autre processus) | | | | | | S-Lab 1.0 |

**Lecture.** Sur l'action de l'anime, aucun candidat ne fait mieux que RIFE.
GIMM fait jeu égal (21,09 contre 21,02 dB, LPIPS 0,358 contre 0,357, dans le
bruit) mais il est **22× plus lent** et demande CuPy (compilation de noyaux
CUDA, rien sur Mac). AMT gagne 1 à 2 dB sur les plans calmes, où tous les
modèles sont déjà au-dessus de 45 dB (écart invisible), mais il est **moins
bon en perception** (LPIPS 0,39 à 0,43 contre 0,357 sur l'action), 8 à 14×
plus lent, et il prend 14 à 18 Go à 1080p, donc aucun palier sous 16 Go.
Planche d'action (même plan rapide d'un démon de Black Clover) : AMT-L
**délave et dédouble** l'œil ; GIMM et RIFE laissent le même léger
dédoublement des traits, 4.26 heavy un peu moins.

Images dupliquées : sur les « doublons », AMT et GIMM inventent un peu moins
de mouvement (39,1-39,4 dB contre 38,6 dB), mais le simple fondu restait au-dessus de tous
(40,2 dB, chantier 4). Dans l'usage réel (ralenti entre deux images source
identiques), RIFE recopie l'image à l'identique (> 80 dB), mieux qu'AMT
(70-80 dB). Les coupes franches sont gérées par l'application avant le
modèle (détection de coupe → recopie), de la même façon pour tous.

### 3.3 Profondeur — **VDA Large 364 px en Qualité max**

18 plans calmes/fixes, 216 paires ; scintillement en % de l'étendue de
profondeur du plan (plus bas = plus stable).

| modèle | débit 1080p (sous charge) | brut | aligné | après lissage de l'application | VRAM max (réservée) | chargement | licence |
|---|---:|---:|---:|---:|---:|---:|---|
| **VDA Small 518 px (16 / 24 Go)** | 16,1 im/s | 0,242 % | 0,143 % | 0,112 % | 11,8 Go | 11 s | Apache-2.0 |
| **VDA Small 364 px (Rapide, 8 / 12 Go)** | **46,9 im/s** | 0,350 % | 0,231 % | 0,157 % | 3,3 Go | 11 s | Apache-2.0 |
| VDA Base 518 px | **OOM** sur 24 Go | | | | > 20 Go | | CC-BY-NC |
| VDA Base 364 px | 23,9 im/s | 0,248 % | 0,178 % | 0,120 % | 6,5 Go | 12 s | CC-BY-NC |
| VDA Large 518 px | **OOM** sur 24 Go | | | | > 20 Go | | CC-BY-NC |
| **VDA Large 364 px** | 8,3 im/s (12 im/s une fois la carte moins chargée) | **0,208 %** | 0,150 % | **0,110 %** | 9,3 Go | 34-71 s | CC-BY-NC |

**Lecture.** Large à 364 px est le plus stable (brut −14 % contre Small
518 px, −41 % contre Small 364 px). Surtout, la **planche** (4 plans de Black
Clover, après le lissage de l'application) montre **plus de relief** : le
visage et l'oreille se détachent du fond, les mèches sont nettes, la bouche
ouverte garde sa forme, là où Small 518 px fait une tache claire. Il coûte
cher : 5,7× plus lent que Small 364 px et environ 2× plus lent que Small
518 px, 9,3 Go de VRAM et 1,5 Go à télécharger (2 min 30 à 10 Mo/s). C'est
le profil des options « Qualité max » du chantier 4 (eva02, RIFE heavy).
À 518 px, Base comme Large réclament plus de 20 Go (fenêtre de 32 images)
et ne passent pas sur 24 Go.

Base à 364 px stabilise presque autant que Small 518 px (0,248 %) pour moitié
moins de VRAM, mais il fait apparaître un **damier** sur un plan presque noir
(les autres modèles y produisent un vignettage lisse) et ne donne pas plus de
relief que Small sur les autres plans : **rejeté**.

### 3.4 Tags — **wd-vit reste**

| modèle | tags P / R / F1 (0,35) | meilleur F1 (seuil) | Goku (12 scènes) à 0,85 / 0,5 | autres personnages | personnages Black Clover (≥ 0,5) | débit (sous charge) | chargement | VRAM | téléchargement |
|---|---|---:|---:|---|---|---:|---:|---:|---:|
| **wd-vit v3** | 0,61 / 0,78 / **0,685** | **0,742** (0,50) | 7 / 11 | Vegeta | aucun juste (10 faux, 1 chacun) | **140 img/s** | 4-15 s | 0,8 Go | 0,4 Go |
| wd-eva02-large v3 (Qualité max) | 0,55 / 0,84 / 0,663 | 0,734 (0,65) | 9 / 11 | Vegeta | aucun juste | (54 img/s au ch. 4) | | 2,0 Go | 1,3 Go |
| PixAI v0.9 | 0,54 / 0,76 / 0,628 | 0,662 (0,45) | 10 / 11 | Vegeta | aucun juste | (18 img/s au ch. 4) | | ~20 Go (ORT) | |
| PixAI v1.0 (bf16) | 0,59 / 0,78 / 0,671 | 0,733 (0,55) | 7 / 8 | Vegeta, **Cell** | **Asta : 7 détections, 7 justes** | **4,8 img/s** | 28 s (106 s à froid) | 6,7 Go | 1,9 Go |

**Lecture.** PixAI v1.0 (30 877 tags, données jusqu'en mai 2026) **ne
connaît toujours ni Beerus, ni Whis, ni Kaiô** : ils sont absents de son
vocabulaire. Il ajoute Asta (`asta_(black_clover)`), Noelle Silva et Cell.
Sur les 260 scènes de Black Clover, les 7 détections d'Asta à ≥ 0,5 sont
toutes justes (Asta en forme démoniaque, vérifié à l'œil) et 3 autres
tombent entre 0,24 et 0,42. Au seuil de l'application (0,85), une seule
passe. Sur les tags visuels, il fait un peu moins bien que wd-vit (F1 0,671
contre 0,685), et il est **~30× plus lent dans la même session (4,8 contre
140 img/s), ~50× contre le débit de wd-vit hors charge**. Sur une saison de
3 épisodes (1 045 scènes), cela représente ~3 min 40 de plus, contre ~16 s
pour eva02 en Qualité max. Il faut aussi télécharger 1,9 Go et ajouter un
second moteur de tagger (code PyTorch propre au modèle, sans ONNX). La
recherche so400m trouvait déjà Asta 7 fois sur 10 au chantier 4. **Rejeté
pour les paliers** ; il reste une piste pour un futur « tagger personnages
récents » optionnel.

Piège rencontré : convertir le modèle en bf16 avec `.to(dtype)` convertit
aussi ses tampons RoPE complexes en réels et ruine les sorties (Goku 0/12).
Il faut le passer sous autocast. Ce défaut est corrigé dans le harnais
d'évaluation (`tests/eval/pixai.py`).

## 4. Dans le vrai pipeline (finition, RTX 3090)

`tests/bench_pipeline.py --sections finish` : upscale x2, RIFE x2 et carte de
profondeur sur un extrait 1080p de 5 s de Dragon Ball Super, dans le vrai
sidecar. Quatre configurations ont été passées en **runs alternés**, 4 séries
chacune (un run « palier 24 Go » a été tué par un signal extérieur et il n'en
reste que 3). La carte était chargée par le banc de détourage d'un autre
chantier, de plus en plus au fil de la soirée : le même job de référence
varie de 33 à 54 s. Les médianes et les **paires alternées** (même minute)
sont donc plus parlantes que les valeurs absolues.

| configuration | upscale x2 | RIFE x2 | profondeur | runs |
|---|---:|---:|---:|---:|
| référence chantier 4, palier 24 Go (HFA2k, RIFE 4.26, VDA Small 518) | 39,9 s [33,3 – 54,0] | 34,2 s | 124,7 s [64 – 161] | 4 |
| **nouveau palier 24 Go** (AnimeJaNai UC, RIFE 4.26, VDA Small 518) | **31,7 s** [26,4 – 44,0] | 29,4 s | 107,0 s [98 – 112] | 3 |
| ancien Rapide (animevideov3, RIFE 4.26, VDA Small 364) | 34,1 s [26,0 – 43,1] | 31,0 s | 75,3 s [62 – 126] | 4 |
| **nouvelle Qualité max** (AnimeJaNai UC, RIFE 4.26 heavy, VDA Large 364) | 25,8 s [19,5 – 56,2] | 37,3 s | 129,8 s [100 – 212] | 4 |

Paires alternées pour l'upscale, HFA2k → AnimeJaNai UC (runs consécutifs) :
45,1 → 31,7 s, 33,3 → 13,9 s, 34,7 → 26,4 s, 54,0 → 44,0 s. **AnimeJaNai
UC est plus rapide dans chaque paire**, de 10 à 19 s. Rapporté au temps de
RIFE du même run (pour neutraliser la charge), l'upscale coûte 1,18× RIFE
avec HFA2k, 1,03× avec AnimeJaNai UC et 1,10× avec animevideov3. Pour la
profondeur, VDA Large 364 px (129,8 s) coûte autant que Small 518 px
(124,7 s), à l'intérieur du bruit : le job est limité par le décodage,
l'écriture des PNG 16 bits et le lissage CPU (0,14 à 0,25 s par image), pas
par le réseau. **Dans le pipeline, les deux modèles retenus ne coûtent donc
rien de mesurable.** Ils coûtent du téléchargement : 24,7 Mo pour
AnimeJaNai (2,5 s à 10 Mo/s) et 1,5 Go pour VDA Large (2 min 34), ce
dernier en Qualité max seulement.

Recherche, tags et indexation ne changent pas (aucun modèle retenu dans ces
fonctions) : le pipeline complet du chantier 4 n'a pas été relancé.

## 5. Recommandation NVIDIA : fonction × palier

Les changements par rapport au chantier 4 sont en gras.

| fonction | Rapide | 8 Go | 12 Go | 16 Go | 24 Go | Qualité max |
|---|---|---|---|---|---|---|
| recherche | SigLIP 2 base 128 | so400m 128 | so400m 128 | so400m 128 | so400m 128 | so400m 256 |
| tags | wd-vit | wd-vit | wd-vit | wd-vit | wd-vit | wd-eva02-large |
| upscale | **AnimeJaNai V3 UC** | **AnimeJaNai V3 UC** | **AnimeJaNai V3 UC** | **AnimeJaNai V3 UC** | **AnimeJaNai V3 UC** | **AnimeJaNai V3 UC** |
| interpolation | RIFE 4.26 | RIFE 4.26 | RIFE 4.26 | RIFE 4.26 | RIFE 4.26 | RIFE 4.26 heavy |
| profondeur | VDA Small 364 | VDA Small 364 | VDA Small 364 | VDA Small 518 | VDA Small 518 | **VDA Large 364** |

Éligibilité mesurée (allocateur plafonné) : AnimeJaNai UC passe à 8, 12 et
16 Go (0,15 Go réservés) ; VDA Large 364 px passe à 12 et 16 Go (9,3 Go
réservés au pic) et **manque de mémoire à 8 Go**. Comme pour VDA Small
518 px au chantier 4 (11,8 Go), la Qualité max suppose donc une carte d'au
moins 12 Go ; le palier « Auto » ne la choisit jamais.

**Mode Rapide** : AnimeJaNai V3 UC remplace aussi animevideov3. Le chantier 4
gardait animevideov3 en Rapide parce qu'il était un peu plus rapide sur le
job. Ce n'est plus vrai : UltraCompact fait du x2 natif avec 8 convolutions
(animevideov3 calcule du x4 puis réduit), son débit isolé est 1,6 à 2× plus
élevé (56-75 contre 35 im/s dans la même session) et son temps dans le job
est équivalent ou meilleur (31,7 contre 34,1 s). La qualité est sans
commune mesure (LPIPS 0,030 contre 0,082). Le seul surcoût est le
téléchargement (24,7 Mo contre 2,5 Mo). Le palier CPU / DirectML / XPU
reste inchangé.

Justification case par case :

- **Upscale = AnimeJaNai V3 UltraCompact partout (Rapide → Qualité max).**
  LPIPS 0,030 contre 0,044 (HFA2k) et 0,082 (animevideov3), +3,4 dB sur les
  plans sombres, pas de halo, même VRAM (0,15 Go réservés en 540p, passe
  à 8 Go), plus rapide dans chaque paire du pipeline. Compact donne la même
  qualité en deux fois plus de calcul et n'est donc pas retenu, même en
  Qualité max.
- **Profondeur, Qualité max = VDA Large 364 px.** Scintillement 0,208 %
  contre 0,242 % (Small 518 px), relief nettement plus détaillé sur les
  planches, même temps de job (129,8 contre 124,7 s), 9,3 Go réservés
  (passe à 12 et 16 Go, pas à 8 Go). Les autres paliers gardent Small :
  Large ne tient pas à 518 px et serait 5,7× plus lent que Small 364 px aux
  paliers 8/12 Go.
- **Interpolation, tags, recherche : inchangés** (§ 3.2 et 3.4).

## 6. Apple Silicon (non mesuré cette fois)

Le Mac n'était pas joignable. Ce qui suit est **déduit** :

| fonction | Rapide | Mac 8 Go | Mac 16 Go | Mac 24 Go | Mac 32 Go + | Qualité max |
|---|---|---|---|---|---|---|
| upscale | **AnimeJaNai V3 UC** | **AnimeJaNai V3 UC** | **AnimeJaNai V3 UC** | **AnimeJaNai V3 UC** | **AnimeJaNai V3 UC** | **AnimeJaNai V3 UC** |
| profondeur | DA V2 364 | DA V2 518 | VDA Small 364 | VDA Small 364 | VDA Small 364 | VDA Small 364 (inchangé) |

(recherche, tags et interpolation : inchangés, voir le chantier 4 § 6.5).

- AnimeJaNai V3 UC a la **même architecture** (SRVGG Compact, x2 natif)
  qu'HFA2k. HFA2k sortait au centième le même résultat sur le Mac qu'en CUDA
  (42,51 contre 42,50 dB) et tournait sur le Neural Engine. Le chemin Neural
  Engine (`ane.py`) lit désormais le nombre de convolutions dans le fichier
  de poids (16 pour Compact, 8 pour UltraCompact). La moitié du calcul
  devrait rendre l'upscale un peu plus rapide sur Mac. **À vérifier sur la
  machine.**
- VDA Large n'est **pas** proposé sur Mac : Small à 364 px occupait déjà
  8 Go d'empreinte sur le M1 Pro 16 Go, et Large réserve 9,3 Go de VRAM à
  364 px sur CUDA (en fp32 sur MPS, il en faudrait davantage).

## 7. Ce qui a été intégré

| quoi | où |
|---|---|
| Licences : `PERMISSIVE_LICENSES` + `NON_COMMERCIAL_LICENSES` (CC-BY-NC-4.0, CC-BY-NC-SA-4.0) ; `is_non_commercial()` ; message d'erreur mis à jour ; GPL, OpenRAIL, « other » et inconnu toujours refusés | `backend/model_download.py` |
| Indicateur `non_commercial` dans l'API des paliers et des modèles de finition ; badge « usage non commercial » (FR/EN) dans le tableau des paliers et dans la liste des modèles de finition | `backend/models/tiers.py`, `backend/finish/registry.py`, `app/src/components/ModelTiersSection.tsx`, `FinishModelsSection.tsx`, `app/src/i18n/strings.models.ts`, `strings.finish.ts`, `app/src/api/types.ts`, `finish.ts` |
| Registre de finition : un fichier peut être **un membre d'une archive zip** officielle (somme SHA-256 de l'archive vérifiée, seul le membre est gardé, extraction atomique) | `backend/finish/registry.py` |
| **AnimeJaNai HD V3 UltraCompact** (`animejanai-v3-uc`) : x2 natif, x4 en deux passes, Neural Engine sur Mac ; Rapide, paliers 8 → 24 Go et Qualité max, NVIDIA et Mac | `registry.py`, `upscale.py`, `ane.py`, `tiers.py` |
| **Video Depth Anything Large** (`video-depth-anything-large`) : même code vendored (Apache-2.0), encodeur choisi par l'identifiant du modèle (`VDA_VARIANTS`) ; Qualité max NVIDIA à 364 px | `backend/finish/vda.py`, `runner.py`, `registry.py`, `tiers.py` |
| README (tableaux des modèles et des paliers, paragraphe « modèles non commerciaux ») et CHANGELOG | `README.md`, `CHANGELOG.md` |
| Harnais : upscalers ONNX (ONNX Runtime CUDA, I/O sur le GPU), AMT, GIMM-VFI, VDA Base/Large, PixAI v1.0 | `tests/eval/eval_finish.py`, `tests/eval/pixai.py`, `tests/eval/eval_models.py` |
| Tests : archive zip (membre seul, archive altérée refusée), modèle UltraCompact x2/x4, licences NC acceptées et signalées, configurations VDA, nouveaux choix des paliers | `tests/test_models_nc.py`, `tests/test_model_tiers.py`, `tests/test_finish.py` |

Une installation qui a choisi elle-même HFA2k (Personnalisé ou menu de la
finition) le garde ; HFA2k et VDA Small restent proposés, pour qui veut
rester sur des licences permissives.

## 8. Limites

- **GPU partagé** pendant toute la session (banc de détourage d'un autre
  chantier) : débits absolus plus bas et bruités. Les choix reposent sur des
  écarts de qualité, qui ne dépendent pas de la charge, et sur des vitesses
  relatives mesurées dans la même session.
- **Cartes simulées par plafonnement de l'allocateur seulement** (pas de lest
  ni de passe complète du pipeline dans une carte réduite) : NVDEC/NVENC
  et ffmpeg ne sont pas contraints. AnimeJaNai UC a la même empreinte
  qu'HFA2k, qui passait tous les paliers au chantier 4.
- **Mac non mesuré** (injoignable) : tout le § 6 est déduit.
- **GIMM-VFI en pleine résolution** non mesuré (OOM à côté d'un autre
  processus) ; à mi-résolution il ne gagnait rien.
- La profondeur n'a pas de vérité terrain : la « qualité » de Large repose
  sur le scintillement et sur des planches jugées à l'œil (4 plans).
- Black Clover n'a toujours pas d'annotation de personnages : la précision
  d'Asta (7/7) est vérifiée à l'œil, son rappel n'est pas connu.
- Les poids et le code de AMT, GIMM-VFI et PixAI v1.0 n'ont servi qu'à la
  mesure, dans un dossier à part sur la machine de test, rien n'est
  committé. Deux correctifs locaux ont été nécessaires pour GIMM-VFI sous
  Python 3.13 et CuPy 13 (valeurs par défaut des dataclasses,
  `cupy.RawModule`).

## 9. Données brutes et planches

Sur la machine de test (hors dépôt) : `<données>/c4b/w/*_runs.jsonl`
(mesures), `tags_pixai-v1.0.npz`, sorties d'upscale, d'interpolation et de
profondeur, `<données>/c4b/bench/` (pipeline), planches
`<données>/c4b/reports/c4b-upscale-planche.png`, `c4b-interp-planche.png`,
`c4b-depth-planche-{0,1,2}.png`, `asta.jpg`.

Reproduire : `tests/eval/eval_finish.py upscale --model <id> [--weights
fichier.pth|.onnx]`, `interp --model amt-l|gimm-r-p-ds05 --weights …`
(`AMT_SRC`, `GIMM_SRC`, `GIMM_RUN`), `depth --model
video-depth-anything-large --input-size 364 --weights …`, `eval_models.py
tag --model pixai-v1.0` (`PIXAI_V1_DIR`), puis
`tests/bench_pipeline.py --sections finish --finish-models …`.

## 10. Tests et état de la machine

- VPS (CPU) : `uv run pytest -m "not slow"` → 350 succès, 1 ignoré (test
  matériel réel), 0 échec ; `cd app && npx tsc --noEmit && npx vitest run
  && npm run build` → OK (28 tests vitest).
- La 3090 a été rendue avec `amv-gpu-release` après les mesures. HyperQwen a
  redémarré, puis il a été arrêté 2 min 30 plus tard, proprement (pas d'OOM),
  quand le chantier de détourage a relancé ses bancs sur la carte. Le
  modèle n'a donc pas été relancé (`qwen-on`) : cela aurait coupé ces bancs
  en cours. **À relancer une fois le détourage terminé.**
