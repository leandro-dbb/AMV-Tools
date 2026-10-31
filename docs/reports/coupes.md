# Détection de coupes : objectif 98 % / 98 %

Branche `jarvis/cuts` (depuis `main` = 0.5.0-beta.2). Toutes les mesures
sont **à l'image exacte** (une coupe trouvée une image trop tôt ou trop tard
compte comme ratée *et* comme fausse coupe), sur la grille d'images de
l'application (`-fps_mode cfr`).

## En bref

**Objectif 98 % / 98 % : non atteint.** Chiffres atteints sur le **jeu de
test** (extraits jamais utilisés pour régler), à l'image exacte, avec le
réseau TransNetV2 téléchargé :

| Série (test) | Avant : rappel / précision | Après (NVIDIA) | Après (CPU, = Mac) |
|---|---|---|---|
| Dragon Ball Super, épisode 02 (2 extraits, 107 coupes) | 68,2 % / 93,6 % | **85,0 % / 91,0 %** | **85,0 % / 91,9 %** |
| Black Clover S02E04 (45 coupes) | 91,1 % / 100 % | **95,6 % / 100 %** | **95,6 % / 100 %** |

Le meilleur extrait de test atteint 95,5 % de rappel (DBS 02, dialogues et
préparation de combat) ; l'extrait de combat pur (`dbs_beerus_02_b`) reste à
77,8 % : plans de 2 images, gros plans de même palette, coupes vers du noir
ou du blanc. Raison prouvée sur planches au § 3 : les coupes restantes ne se
distinguent pas, image contre image, d'un mouvement d'anime (et les fausses
coupes sont des flashs et des mains dans l'objectif) ; les combineurs plus
souples apprennent par cœur nos 4 extraits de réglage (§ 4).

- **Comment** : TransNetV2 (réseau de référence, sur GPU quand il y en a un)
  combiné au détecteur actuel et à la couleur, décision réglée sur 4 extraits
  annotés et mesurée sur 3 autres ; **5 nouveaux extraits annotés image par
  image** (251 coupes).
- **Coût** : +11,4 % sur l'indexation complète d'un épisode sur RTX 3090
  (limite 15 %). Sur CPU (6 cœurs), la passe de coupes passe de 328 à 64
  images/s (2,6× le temps réel) : ~7 min de plus par épisode de 23 min.
- **Optionnel** : sans les poids (30 Mo, Paramètres › Modèles), le
  détecteur actuel tourne comme avant, au bit près.
- **Recherche / synchro** : sur le test, les plans qui cachent une vraie
  coupe passent de 16,4 % à 9,6 % ; 88 % des vrais plans (contre 75 %)
  deviennent un plan à part entière (§ 6).

## 1. Plus de vérité terrain

Il n'y a que trois épisodes sur le NAS (`/srv/data/media/attente` : DBS
Beerus 01 et 02, Black Clover S02E04 ; `anime`, `series`, `movies` sont
vides) : **pas de vieil anime 4:3 disponible**, la variété vient des
passages choisis (dialogues sombres, combats rapides, flashs, magie).

Cinq nouveaux extraits de 150 s (copies du flux sans réencodage, à partir
d'une image clé), pris **ailleurs** que les deux extraits existants (repérés
dans les épisodes : DBS 01 à 118 s, Black Clover à 175 s) :

| Extrait | Source | Jeu | Coupes franches | Fondus | Ambigus |
|---|---|---|---|---|---|
| `dbs_beerus_01_extract` (existant) | DBS 01, 118 s | réglage | 67 | 1 | 9 |
| `black_clover_s02e04_extract` (existant) | BC, 175 s | réglage | 42 | 4 | 2 |
| `dbs_beerus_01_b` | DBS 01, 600 s | réglage | 41 | 0 | 2 |
| `black_clover_s02e04_c` | BC, 1000 s | réglage | 58 | 3 | 3 |
| `dbs_beerus_02_a` | **DBS 02**, 300 s | **test** | 44 | 0 | 3 |
| `dbs_beerus_02_b` | **DBS 02**, 900 s (combat) | **test** | 63 | 0 | 18 |
| `black_clover_s02e04_b` | BC, 500 s | **test** | 45 | 5 | 2 |

Le jeu de test est fait d'un **autre épisode** pour DBS (le 02, jamais vu au
réglage) ; pour Black Clover il n'y a qu'un épisode : l'extrait de test est
un autre passage du même épisode (limite assumée, faute d'autre épisode).
Tous les réglages (coefficients, seuil) sont faits sur le jeu de réglage
seul ; les chiffres « test » ne servent qu'à mesurer.

**Méthode d'annotation** (chaque extrait, `tests/data/cuts_gt/*.json`,
champ `method`) : toutes les images décodées ; les 37 planches de 100 images
regardées dans l'ordre (les fortes variations de mesures simples encadrées,
à titre d'indice seulement) ; chaque passage d'action revu image par image à
320 px ; **chaque coupe vérifiée sur une planche n-2 | n-1 | n | n+1**
(n-1 dans l'ancien plan, n dans le nouveau). Fondus notés en plages,
cas discutables (zooms par paliers de 2 images, caméra qui tourne autour des
combattants, coupe sous un flash) notés « ambigus » : ils ne comptent ni
comme ratés ni comme fausses coupes. L'annotation a été faite par des agents
(un par extrait), indépendamment de tout détecteur ; la revue des 55 erreurs
restantes (§ 3) n'a trouvé **aucune annotation fausse**, seulement 4 cas
discutables.

**Numérotation.** Le flux de Black Clover a un trou d'horodatage (image
2058 de `black_clover_s02e04_c`) : l'application, qui numérote sur une grille
à cadence fixe, y duplique une image. La mesure convertit donc les numéros
d'annotation (ordre de décodage) en numéros de l'application via les
horodatages des paquets (`tests/eval/cuts_eval.py`, `gt()`).

## 2. Ce qui a été fait

**TransNetV2 combiné au détecteur actuel** (`backend/indexing/cutnet.py`,
`shotdetect.py`). TransNetV2 (Souček & Lokoč 2020, architecture MIT ; poids
PyTorch `Sn4kehead/TransNetV2`, Apache-2.0, 30 Mo, **téléchargés à la
demande** depuis Paramètres › Modèles, jamais commités) regarde des fenêtres
de 100 images 48×27 et donne pour chaque image la probabilité d'une
transition. Il voit **toutes les images** (fenêtres de référence : 100
images toutes les 50, les 50 du milieu gardées). Notre portage reproduit la
référence à 2·10⁻⁴ près.

La décision est une **régression logistique** sur 12 mesures : 5 tirées du
réseau (probabilité à l'image, avant, après, « pic », probabilité
« multi-images »), les 6 mesures du détecteur actuel (différence après
recalage, rapport à l'activité locale, distance d'histogramme gris, retour de
l'ancienne image dans les 6 suivantes, nouvelle image déjà vue, règle
actuelle vraie/fausse) et une distance d'histogramme **couleur**. Coefficients
et seuil appris **sur le jeu de réglage seul** (seuil = celui qui maximise le
pire de rappel/précision sur les 4 extraits de réglage). Le curseur « seuil
de détection » des Paramètres règle toujours la sensibilité.

**La couleur dans tous les chemins.** Le réseau et l'histogramme couleur ont
besoin de couleur ; jusqu'ici tous les chemins ne sortaient que le gris
160×90. Quand le réseau est actif : ffmpeg deux passes (CPU Linux/Windows,
CUDA, VideoToolbox), PyAV et `single_pass` (Mac et CPU) empilent l'image
160×90 couleur sous le plan gris (le plan gris reste identique au bit près,
vérifié) ; `onepass` et `nvc_pass` (NVIDIA par défaut) fabriquent l'image
48×27 du réseau à partir du plan gris et de la chrominance de l'image
512×288 qu'ils ont déjà pour chaque image. Sans les poids (ou
`AMV_CUTNET=0`), rien ne change : mêmes tuyaux, même règle, mêmes coupes.

**Mac (lecture de code).** Le Mac indexe par `single_pass` (même code que le
CPU : décodage logiciel ou VideoToolbox, même empilement couleur, même
décision) ; le réseau y tourne **sur le CPU** : le portage de référence
prévient que les opérations de TransNetV2 retombent sur le CPU de façon
incohérente avec MPS (résultats différents), donc mêmes calculs que le
chemin CPU. Débit non mesuré (pas de Mac ici) : il faut s'attendre au même
ordre de surcoût que sur le CPU du NAS (§ 5), à confirmer sur le M1 Pro.

## 2 bis. Résultats par chemin (jeu de test)

Évaluation finale lancée à 14 h 45 (chemins NVIDIA) et relancée à 15 h 23
pour les chemins CPU après une correction (couleur des chemins CPU, voir
plus bas ; le chemin NVIDIA `hw`, qui partage ce code, a été remesuré après
la correction : mêmes chiffres), code final ; `tests/eval/cuts_eval.py`,
médias sur le NAS.

| Chemin | Série (test) | Avant | Après |
|---|---|---|---|
| NVIDIA `nvc_pass` (défaut) | DBS 02 (107 coupes) | 68,2 / 93,6 | 85,0 / 91,0 |
| NVIDIA `nvc_pass` (défaut) | Black Clover (45) | 91,1 / 100 | 95,6 / 100 |
| NVIDIA `onepass` | DBS 02 | 68,2 / 93,6 | 85,0 / 91,0 |
| NVIDIA `onepass` | Black Clover | 91,1 / 100 | 95,6 / 100 |
| NVIDIA ffmpeg CUDA (`hw`) | DBS 02 | 68,2 / 93,6 | 85,0 / 91,0 |
| NVIDIA ffmpeg CUDA (`hw`) | Black Clover | 91,1 / 100 | 95,6 / 100 |
| CPU ffmpeg (Linux / Windows sans GPU) | DBS 02 | 68,2 / 93,6 | 85,0 / 91,9 |
| CPU ffmpeg | Black Clover | 91,1 / 100 | 95,6 / 100 |
| CPU `single_pass` (= Mac) | DBS 02 | 68,2 / 93,6 | 85,0 / 91,9 |
| CPU `single_pass` (= Mac) | Black Clover | 91,1 / 100 | 95,6 / 100 |
| CPU PyAV (secours) | DBS 02 | 67,3 / 93,5 | 85,0 / 91,9 |
| CPU PyAV (secours) | Black Clover | 91,1 / 100 | 95,6 / 100 |

(rappel / précision en %, les deux extraits DBS 02 cumulés.) La seule
différence entre chemins : sur `dbs_beerus_02_a`, les chemins NVIDIA font
une fausse coupe de plus (leur image couleur est tirée de l'image 512×288,
celle des chemins CPU de l'image pleine taille).

Par extrait (`nvc` pour NVIDIA, `sw` pour CPU ; « avant » est identique sur
tous les chemins sauf PyAV, une coupe de moins sur `dbs_beerus_02_a`) :

| Extrait | Jeu | Avant | Après NVIDIA | Après CPU |
|---|---|---|---|---|
| `dbs_beerus_02_a` | test | 84,1 / 94,9 | 95,5 / 91,3 | 95,5 / 93,3 |
| `dbs_beerus_02_b` | test | 57,1 / 92,3 | 77,8 / 90,7 | 77,8 / 90,7 |
| `black_clover_s02e04_b` | test | 91,1 / 100 | 95,6 / 100 | 95,6 / 100 |
| `dbs_beerus_01_extract` | réglage | 80,6 / 93,1 | 89,6 / 92,3 | 89,6 / 93,8 |
| `dbs_beerus_01_b` | réglage | 95,1 / 100 | 97,6 / 100 | 97,6 / 100 |
| `black_clover_s02e04_extract` | réglage | 85,7 / 90,0 | 88,1 / 88,1 | 88,1 / 88,1 |
| `black_clover_s02e04_c` | réglage | 82,8 / 100 | 91,4 / 100 | 91,4 / 100 |

**Correction en cours de mesure (honnêteté).** La première évaluation CPU
donnait sur `dbs_beerus_02_a` 84,1 / 94,9 au lieu de 95,5 : le chemin CPU
convertissait en couleur *après* la réduction à 160×90 (chrominance 80×45),
alors que la décision a été apprise sur une conversion directe depuis
l'image pleine taille. Corrigé pour que l'entrée soit identique au bit près
à celle de l'apprentissage (vérifié), sans toucher ni coefficients ni seuil ;
c'est un défaut de chaîne, pas un réglage sur le test.

## 3. Ce qui reste raté, par catégorie

Toutes les erreurs restantes (chemin NVIDIA, réseau actif, les 7 extraits :
36 ratés, 19 fausses coupes ; mesure de 14 h 15, juste avant la dernière
optimisation de débit, qui déplace une fausse coupe sur deux extraits) ont
été regardées sur planches n-2 | n-1 | n |
n+1 (`docs/reports/coupes/*.jpg` : trois exemples). **Aucune n'est une
erreur d'annotation** ; 2 sont des détections à une image près, 4 sont
discutables (dernier palier d'un zoom par paliers noté coupe, main lancée
dans l'objectif…).

| Ratés (coupes non trouvées) | test | tous |
|---|---|---|
| rafales (plans de 2 images en combat, images d'impact) | 5 | 8 |
| même palette (deux gros plans du même personnage, même fond) | 4 | 9 |
| coupe vers / depuis du noir ou du blanc | 3 | 5 |
| coupe suivie d'un fort mouvement / secousse | 3 | 5 |
| objet qui passe devant l'objectif / plan sombre | 2 | 3 |
| caméra qui tourne autour des combattants, zoom par paliers, flash | 1 | 6 |

| Fausses coupes | test | tous |
|---|---|---|
| flash blanc ou coloré, inversion de couleurs dans le même plan | 4 | 5 |
| objet / main / jambe qui balaie l'objectif | 3 | 6 |
| explosion, poussière, aura dans le même plan | 2 | 3 |
| zoom par paliers, caméra qui tourne | 0 | 4 |

Planches : `coupes/meme-palette.jpg` (Black Clover, cristaux roses des deux
côtés de la coupe), `coupes/rafales.jpg` (DBS, plans de 2 images),
`coupes/occlusion-flash.jpg` (cape qui passe devant l'objectif, flash).

**Pourquoi pas 98 %.** Les ratés restants sont des coupes que ni le réseau
(entraîné sur des films et séries en prise de vues réelles, pas sur de
l'anime) ni les mesures d'image ne séparent d'un mouvement : deux plans de
même palette avec un personnage au même endroit, ou des plans de 2 images
au milieu d'un combat où chaque dessin change déjà tout. Les fausses coupes
sont l'inverse : un flash ou une main dans l'objectif change toute l'image
sans changer de plan. Sur l'extrait de combat (`dbs_beerus_02_b`), même
l'annotation a dû marquer 18 zones ambiguës. Avec un combineur appris sur 4
extraits (208 coupes), aller plus loin apprend par cœur (gradient boosting :
93-100 % sur le réglage, **plus bas** que la régression logistique sur le
test, voir § 4).

## 4. Pistes mesurées et écartées

Toutes mesurées sur le jeu de test, réglées sur le jeu de réglage.

| Piste | Test DBS02_a | Test DBS02_b | Test BC | Verdict |
|---|---|---|---|---|
| Détecteur actuel seul | 84 / 95 | 57 / 92 | 91 / 100 | avant |
| TransNetV2 seul, seuil 0,5 | 91 / 95 | 67 / 93 | 96 / 98 | moins bon que combiné |
| TransNetV2 **ou** détecteur actuel (vote « l'un des deux ») | 93 / 93 | 73 / 87 | 96 / 98 | précision en baisse |
| Détecteur actuel **et** TransNetV2 > 0,1 (confirmation) | 84 / 95 | 57 / 95 | 91 / 100 | rappel inchangé |
| Combiné, TransNetV2 sur l'image **grise** (aucun chemin à toucher) | 84 / 97 | 67 / 93 | 96 / 100 | la couleur compte sur DBS |
| Combiné, gradient boosting au lieu de logistique | 93 / 84 | 76 / 79 | 87 / 100 | apprend par cœur |
| **Combiné, couleur (retenu)** | **98 / 93** | **79 / 91** | **96 / 100** | (calcul hors ligne) |

(rappel / précision en %, calcul hors ligne sur les mêmes images que
l'application ; les chiffres des vrais chemins sont au § 2.)

- **TransNetV2 en ONNX Runtime sur CPU** : 67 img/s contre 74 en PyTorch
  (6 cœurs du NAS) : pas plus rapide, écarté. **int8** : la quantification
  dynamique d'ONNX Runtime ne touche que les couches linéaires, pas les
  convolutions 3D qui font l'essentiel du calcul ; pas d'OpenVINO dans les
  dépendances. Non retenu.
- **AutoShot** : poids distribués hors Hugging Face sans licence claire ;
  non essayé faute de temps (TransNetV2 est sa base).
- **Descripteurs SigLIP** de l'indexation : ils ne sont calculés que pour
  l'image médiane de chaque plan, pas pour chaque image ; les calculer pour
  chaque image coûterait bien plus que les 15 % permis. Écarté.
- **Décoder ou analyser moins d'images** : interdit et non fait ; toutes les
  images passent par le réseau et par les mesures.

## 5. Débit

**RTX 3090 (debian-nas, Ryzen 9 5900X 6 vCPU), indexation complète d'un
épisode** (DBS Beerus 01, 23 min, 33 512 images ; `tests/bench_pipeline.py
--sections season` : coupes, mouvement, images médianes, tags, embeddings,
miniatures, proxys), deux passages alternés de chaque, cache disque chaud :

| | passage 1 | passage 2 | décodage + coupes + mouvement | plans |
|---|---|---|---|---|
| Sans réseau | 70,6 s | 71,3 s | 60,1 / 60,7 s | 399 |
| Avec réseau | 79,4 s | 78,7 s | 67,0 / 66,9 s | 497 |

**+11,4 %** sur l'indexation (limite : 15 %). Le réseau lui-même tient
5 000 images/s sur la 3090 ; le surcoût vient surtout du travail par image
côté CPU (image couleur 48×27, histogramme, décision). Deux optimisations
ont été nécessaires pour passer sous 15 % (premier essai : +37 %) : les
mesures coûteuses ne sont calculées que si le score peut encore franchir le
seuil (décisions identiques, vérifié), et l'image du réseau est construite
directement en 48×27 (un `cv2.resize` par petite image coûtait ~0,2 ms
sous charge).

**CPU.** TransNetV2 en PyTorch sur les 6 vCPU du NAS : **74 images/s**
(ONNX Runtime : 67). Passe de coupes seule, CPU uniquement
(`CUDA_VISIBLE_DEVICES=`), extrait `dbs_beerus_02_a` (3 638 images, 2 min
32 s), machine au calme : **11,1 s sans réseau, 57,3 s avec** (328 → 64
images/s, 2,6× le temps réel). Soit environ **+7 min pour un épisode de
23 min** sur 6 cœurs. Utilisable, mais c'est un choix : sans GPU, mieux vaut
ne télécharger TransNetV2 que si la précision des coupes compte plus que
le temps d'indexation. Mac : même code que le CPU (le réseau y tourne sur
le CPU), débit non mesuré faute de Mac.

## 6. Synchro musique et recherche

Chaque plan détecté est un résultat de recherche (son image médiane est
indexée) et un clip pour la synchro musique. Une coupe ratée cache un
deuxième plan dans le premier : ce plan-là n'est jamais un résultat à lui
seul et la synchro peut montrer un saut au milieu d'un clip.

Mesuré sur le jeu de test (chemin NVIDIA, 155 vrais plans) :

| | Avant | Après |
|---|---|---|
| Plans détectés qui cachent une vraie coupe | 20 sur 122 (16,4 %) | 14 sur 146 (9,6 %) |
| Vrais plans qui commencent un plan détecté (donc trouvables seuls en recherche, utilisables seuls en synchro) | 117 / 155 (75,5 %) | 137 / 155 (88,4 %) |
| Sur l'épisode DBS 01 entier : plans | 399 | 497 |

La synchro musique propose des coupes calées sur les plans détectés : avec
moins de plans « doubles », un clip calé sur un temps fort ne montre plus de
saut au milieu ; la recherche indexe une image par plan : 20 vrais plans de
plus sur 155 deviennent des résultats à part entière. (Mesure faite sur les
plans, pas par un test d'écoute ou de pertinence de recherche.)

## 7. Tests et reproductibilité

- `tests/test_shotdetect.py` : repli sans poids (règle actuelle inchangée),
  réseau en flux = fenêtres de référence, détecteur combiné flux = lot,
  cas synthétiques, et **non-régression sur la vérité terrain** : planchers
  de rappel / précision par extrait (règle seule, chemins `sw`, `pyav`,
  `single_pass` ; avec le réseau, marqué `slow`). S'exécute si les extraits
  sont présents (`AMV_CUTS_MEDIA`).
- `tests/eval/cuts_eval.py` : mesure des vrais chemins (`sw`, `pyav`,
  `single_pass`, `hw`, `onepass`, `nvc`) sur les 7 extraits, à l'image
  exacte, par jeu.
- Médias (non commités, > 1 Mo) : NAS `/tmp/cutsgt`, VPS
  `~/dev/amv-testdata/cuts-new` et `Séries test`.
