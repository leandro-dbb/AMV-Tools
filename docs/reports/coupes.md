# Détection de coupes : objectif 98 % / 98 %

Branche `jarvis/cuts` (depuis `main` = 0.5.0-beta.2). Toutes les mesures
sont **à l'image exacte** (une coupe trouvée une image trop tôt ou trop tard
compte comme ratée *et* comme fausse coupe), sur la grille d'images de
l'application (`-fps_mode cfr`).

## En bref

@@BREF@@

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
chemin CPU. Débit non mesuré (pas de Mac ici) ; à l'échelle du CPU du NAS
(voir § 5), un M1 Pro (8 cœurs performance) devrait être du même ordre.

@@RESULTATS@@

## 3. Ce qui reste raté, par catégorie

Toutes les erreurs restantes (chemin NVIDIA, réseau actif, les 7 extraits :
36 ratés, 19 fausses coupes) ont été regardées sur planches n-2 | n-1 | n |
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

@@DEBIT@@

## 6. Synchro musique et recherche

Chaque plan détecté est un résultat de recherche (son image médiane est
indexée) et un clip pour la synchro musique. Une coupe ratée cache un
deuxième plan dans le premier : ce plan-là n'est jamais un résultat à lui
seul et la synchro peut montrer un saut au milieu d'un clip.

@@SYNCHRO@@

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
