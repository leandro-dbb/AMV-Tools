# Chantier roto : rendre le détourage (Mask Mode) utilisable sur de vrais épisodes

Date : 25 octobre 2026. Machine de mesure : RTX 3090 24 Go (Linux, 6 vCPU, 15 Go de RAM).
Le Mac M1 Pro 16 Go était injoignable pendant tout le chantier : **rien n'a été mesuré sur Mac**.

Priorité fixée en cours de route : le mode **automatique** (aucun clic) d'abord, le mode manuel ensuite
(il ne devait pas régresser).

## 1. Pourquoi refaire l'évaluation

La note de la 0.4 annonçait un IoU de 0,98 pour l'Auto… sur des clips synthétiques (un sprite qui se
déplace sur un fond). Sur de vrais épisodes, c'est faux : l'ancien Auto conseillé par la doc (ISNet
anime) tombe à **J 0,75** avec un échec (personnage perdu) sur 5 images clés, et le Manuel par défaut
(SAM 2.1 tiny à partir d'un clic) à **J 0,60** avec 4 échecs sur 10.

## 2. Protocole

### Jeu de plans

23 plans tirés de trois épisodes réels (deux épisodes de *Dragon Ball Super*, H.264 1080p ; un épisode
de *Black Clover*, HEVC 1080p, en 2,39:1 avec bandes noires), décodés à la taille de travail de Mask
Mode (1080 px de grand côté, soit 1080×608) avec l'extracteur de l'application :

| Catégorie | Plans | Contenu |
|---|---|---|
| calme | 2 | gros plans fixes, fond ciel |
| multi-personnages | 3 | 2 à 4 personnages, dialogue |
| cheveux / contours fins | 3 | Super Saiyan 3 (mèches longues), gros plan, cheveux longs blonds sur ciel |
| action rapide / flou | 5 | coups, flou de mouvement, éclairs, fumée de vitesse |
| effets | 3 | auras, particules et cristaux, forme démoniaque |
| occlusion | 2 | personnage qui sort de la fumée, personnage derrière un mur |
| sombre | 3 | couloir de nuit, palais sombre, pluie de nuit |
| fond chargé | 1 | deux personnages collés à un tracteur |
| plan long | 1 | 11 s continues (264 images), panoramique, deux personnages |

Les plans ont été découpés aux coupes franches (détection sur les images puis vérification visuelle
de chaque plan sur une planche d'images), d'où des durées de 1,3 à 3,5 s (11 s pour le plan long).

### Vérité terrain

- **Cible** : tous les personnages visibles du plan, corps et cheveux compris ; **l'aura, les
  éclairs, les rayons d'énergie et la fumée sont exclus** (en montage AMV on les garde ou on les
  recrée à part ; un détourage « personnage » ne doit pas les avaler). Les objets tenus (bâton,
  cuillère) font partie du personnage.
- **Outil** : SAM 2.1 large (le meilleur segmenteur disponible) sur 6 images clés par plan, d'abord
  à partir des boîtes d'un détecteur, puis — pour 14 plans sur 23 — avec des boîtes et des points
  placés à la main par personnage, puis nettoyage des grains (< 0,15 % de l'image).
- **Contrôle** : chaque planche (image avec grille de coordonnées / superposition masque + contour)
  a été regardée et corrigée jusqu'à être juste ; 2 images clés ambiguës (éclair blanc qui couvre
  tout, rayon translucide devant le personnage) sont exclues du score. Au final : 135 masques de
  référence, dont 112 notés (sur chaque plan, l'image clé où le personnage est le plus grand sert de
  « clic » simulé pour le mode manuel et n'est pas notée).
- **Limites** : la référence est binaire (pas d'alpha) ; elle est construite avec SAM 2.1 large, ce
  qui peut avantager légèrement les méthodes de la même famille sur le tracé exact des bords ; sur
  les contours sombres sur fond sombre (cheveux noirs de nuit) le bord de référence est incertain de
  quelques pixels ; un plan (tracteur) a une cible que tous les détecteurs confondent avec l'objet.

### Métriques

- **J** : IoU de alpha > 0,5 avec la référence ; **F** : F-mesure des contours (tolérance DAVIS,
  0,8 % de la diagonale ≈ 10 px) ; **J&F** : moyenne des deux.
- **Bord** : erreur absolue moyenne de l'alpha dans une bande de ±8 px autour du contour de
  référence (×100, plus bas = mieux).
- **Scintillement** : variation temporelle de l'alpha compensée par le flot optique,
  Σ|aₜ − warp(aₜ₋₁)| / Σaₜ en %, sur toutes les images (pas seulement les images clés) ; les paires où
  le flot n'explique pas l'image (coupes, action trop rapide) sont exclues.
- **Échecs** : images clés avec J < 0,5 (personnage perdu, avalé par le fond, ou autre chose
  détouré).
- **Vitesse et mémoire** : ms par image (après chauffe), pic de VRAM (nvidia-smi, inclut ONNX
  Runtime) et pic de RAM du processus.
- **Verdict visuel par plan** sur des planches alpha composées sur damier et sur fond vert :
  *utilisable* (exportable tel quel), *retouche légère* (quelques images ou un bord à reprendre au
  pinceau), *inutilisable*.

## 3. Résultats

### Mode Auto (aucun clic) — tableau principal

Cible notée : tous les personnages du plan (option « Tous les personnages »). 22 plans notés
(un plan exclu, voir limites), 107 images clés.

| catégorie | méthode | J | J&F | bord | scint. % | échecs |
|---|---|---|---|---|---|---|
| calme | ISNet anime (0.4, conseillé par la doc) | 0.978 | 0.919 | 9.3 | 0.3 | 0/10 |
| calme | BiRefNet seul (0.4, réglage par défaut) | 0.987 | 0.910 | 8.1 | 0.1 | 0/10 |
| calme | ToonOut seul | 0.988 | 0.907 | 7.5 | 0.1 | 0/10 |
| calme | MatAnyone 2 (amorce auto) | 0.988 | 0.911 | 7.8 | 0.1 | 0/10 |
| calme | **Auto 0.5** | 0.987 | 0.911 | 7.1 | 0.1 | 0/10 |
| multi-personnages | ISNet anime (0.4, conseillé par la doc) | 0.888 | 0.874 | 14.7 | 0.4 | 0/15 |
| multi-personnages | BiRefNet seul (0.4, réglage par défaut) | 0.954 | 0.945 | 8.9 | 0.2 | 0/15 |
| multi-personnages | ToonOut seul | 0.890 | 0.865 | 12.2 | 0.4 | 0/15 |
| multi-personnages | MatAnyone 2 (amorce auto) | 0.946 | 0.933 | 9.4 | 0.2 | 0/15 |
| multi-personnages | **Auto 0.5** | 0.948 | 0.937 | 8.6 | 0.3 | 0/15 |
| cheveux | ISNet anime (0.4, conseillé par la doc) | 0.984 | 0.963 | 7.8 | 0.9 | 0/10 |
| cheveux | BiRefNet seul (0.4, réglage par défaut) | 0.983 | 0.967 | 7.0 | 0.8 | 0/10 |
| cheveux | ToonOut seul | 0.850 | 0.865 | 8.7 | 1.1 | 0/10 |
| cheveux | MatAnyone 2 (amorce auto) | 0.926 | 0.900 | 10.3 | 0.7 | 0/10 |
| cheveux | **Auto 0.5** | 0.983 | 0.967 | 5.8 | 0.8 | 0/10 |
| action floue | ISNet anime (0.4, conseillé par la doc) | 0.724 | 0.670 | 27.7 | 25.5 | 4/23 |
| action floue | BiRefNet seul (0.4, réglage par défaut) | 0.860 | 0.843 | 15.5 | 4.2 | 0/23 |
| action floue | ToonOut seul | 0.788 | 0.741 | 16.4 | 6.9 | 1/23 |
| action floue | MatAnyone 2 (amorce auto) | 0.782 | 0.725 | 19.8 | 3.7 | 3/23 |
| action floue | **Auto 0.5** | 0.876 | 0.855 | 14.3 | 5.5 | 0/23 |
| effets | ISNet anime (0.4, conseillé par la doc) | 0.523 | 0.472 | 36.7 | 32.4 | 5/14 |
| effets | BiRefNet seul (0.4, réglage par défaut) | 0.713 | 0.687 | 24.9 | 19.6 | 1/14 |
| effets | ToonOut seul | 0.307 | 0.229 | 47.7 | 6.6 | 13/14 |
| effets | MatAnyone 2 (amorce auto) | 0.621 | 0.598 | 26.9 | 1.8 | 4/14 |
| effets | **Auto 0.5** | 0.732 | 0.687 | 25.3 | 14.0 | 0/14 |
| occlusion | ISNet anime (0.4, conseillé par la doc) | 0.750 | 0.707 | 17.1 | 5.3 | 3/10 |
| occlusion | BiRefNet seul (0.4, réglage par défaut) | 0.871 | 0.828 | 14.2 | 3.2 | 1/10 |
| occlusion | ToonOut seul | 0.639 | 0.561 | 24.4 | 2.2 | 4/10 |
| occlusion | MatAnyone 2 (amorce auto) | 0.734 | 0.646 | 28.3 | 1.6 | 2/10 |
| occlusion | **Auto 0.5** | 0.872 | 0.834 | 12.9 | 3.0 | 1/10 |
| sombre | ISNet anime (0.4, conseillé par la doc) | 0.653 | 0.606 | 28.1 | 13.1 | 4/15 |
| sombre | BiRefNet seul (0.4, réglage par défaut) | 0.788 | 0.758 | 18.2 | 4.2 | 1/15 |
| sombre | ToonOut seul | 0.715 | 0.658 | 21.0 | 2.9 | 0/15 |
| sombre | MatAnyone 2 (amorce auto) | 0.812 | 0.776 | 16.6 | 2.2 | 0/15 |
| sombre | **Auto 0.5** | 0.787 | 0.757 | 17.4 | 4.1 | 1/15 |
| fond chargé | ISNet anime (0.4, conseillé par la doc) | 0.335 | 0.374 | 25.8 | 0.0 | 5/5 |
| fond chargé | BiRefNet seul (0.4, réglage par défaut) | 0.337 | 0.380 | 25.3 | 0.0 | 5/5 |
| fond chargé | ToonOut seul | 0.338 | 0.378 | 24.9 | 0.0 | 5/5 |
| fond chargé | MatAnyone 2 (amorce auto) | 0.337 | 0.378 | 24.8 | 0.0 | 5/5 |
| fond chargé | **Auto 0.5** | 0.337 | 0.373 | 24.3 | 0.0 | 5/5 |
| plan long 11 s | ISNet anime (0.4, conseillé par la doc) | 0.940 | 0.856 | 18.3 | 7.9 | 0/5 |
| plan long 11 s | BiRefNet seul (0.4, réglage par défaut) | 0.961 | 0.884 | 15.9 | 1.0 | 0/5 |
| plan long 11 s | ToonOut seul | 0.847 | 0.768 | 19.0 | 1.1 | 0/5 |
| plan long 11 s | MatAnyone 2 (amorce auto) | 0.962 | 0.895 | 11.5 | 0.8 | 0/5 |
| plan long 11 s | **Auto 0.5** | 0.962 | 0.887 | 14.9 | 0.9 | 0/5 |
| **tous (22 plans)** | ISNet anime (0.4, conseillé par la doc) | 0.751 | 0.710 | 22.3 | 13.0 | 21/107 |
| **tous (22 plans)** | BiRefNet seul (0.4, réglage par défaut) | 0.847 | 0.821 | 15.1 | 4.6 | 8/107 |
| **tous (22 plans)** | ToonOut seul | 0.719 | 0.671 | 20.5 | 3.3 | 23/107 |
| **tous (22 plans)** | MatAnyone 2 (amorce auto) | 0.802 | 0.761 | 17.6 | 1.7 | 14/107 |
| **tous (22 plans)** | **Auto 0.5** | 0.853 | 0.823 | 14.4 | 4.2 | 7/107 |

Lecture honnête :

- **Contre ce que la doc 0.4 conseillait (ISNet anime)** : J 0,75 → 0,85, échecs 21 → 7 sur 107,
  scintillement 13 % → 4 %, bords 22 → 14. C'est le saut que ressentira quelqu'un qui suivait la doc.
- **Contre le réglage par défaut réel de la 0.4 (BiRefNet seul, image par image)** : la qualité
  moyenne est **à peine meilleure** (J 0,853 contre 0,847, bords 14,4 contre 15,1, scintillement
  4,2 % contre 4,6 %, 7 échecs contre 8). Le suivi apporte des bords un peu plus propres, un
  peu moins de scintillement, aucun échec sur les effets, et surtout ce que BiRefNet seul ne sait pas
  faire : choisir le personnage principal, ignorer un objet que le détecteur prend pour un
  personnage *sur certaines images seulement*, traverser une coupe. Il coûte du temps (≈ 2,5× plus
  lent que BiRefNet seul sur la même machine chargée).
- **Le vrai plafond est le détecteur.** Quand BiRefNet prend un objet pour un personnage sur toutes
  les images (le tracteur, le pilier et les fruits, le décor sombre de *Black Clover*), le suivi le
  suit fidèlement. Aucun des détecteurs testés ne sépare un personnage d'un objet collé à lui.
- **ToonOut**, affiné sur l'anime, est le meilleur sur les bords des plans calmes mais garde les
  auras et les effets (13 échecs sur 14 images « effets ») : rejeté comme défaut.
- **MatAnyone 2** (non commercial), amorcé par la même sélection automatique : le plus stable
  (scintillement 1,7 %) et bon sur le calme et le sombre, mais il perd le personnage sur l'action
  rapide et les effets (J 0,80, 14 échecs) : rejeté par la mesure.

### Vitesse et mémoire (RTX 3090)

| méthode | ms/image (banc chargé) | VRAM max | RAM max |
|---|---|---|---|
| ISNet anime (0.4, conseillé par la doc) | 69 | 1.3 Go | 2.6 Go |
| BiRefNet seul (0.4, réglage par défaut) | 209 | 2.6 Go | 3.2 Go |
| ToonOut seul | 434 | 2.6 Go | 3.2 Go |
| MatAnyone 2 (amorce auto) | 662 | 2.1 Go | 4.3 Go |
| **Auto 0.5** | 551 | 2.9 Go | 4.1 Go |
| SAM 2.1 tiny, 1 clic (0.4) | 418 | 6.1 Go | 4.8 Go |
| SAM 2.1 large, 1 clic | 519 | 8.8 Go | 4.5 Go |
| **Manuel 0.5** | 402 | 2.9 Go | 4.0 Go |

Les temps ont été mesurés pendant que d'autres calculs tournaient sur la même machine (jusqu'à
trois processus GPU et une autre session) : ils sont gonflés d'un facteur 2 à 4 et ne valent que
pour comparer les méthodes entre elles. VRAM et RAM sont des pics par processus.

Plafond de durée : sur le plan de 11 s, le suivi SAM 2.1 large de la 0.4 (tout le plan dans un seul
état) monte à **9,0 Go de VRAM et 4,6 Go de RAM** ; le suivi par fenêtres de 96 images de la 0.5
reste à **2,1 Go / 3,2 Go**, plus vite, avec une qualité équivalente (J 0,95 contre 0,95). La mémoire
ne dépend plus de la durée du plan ; seules les images extraites (sur disque) et les masques 8 bits
grandissent avec lui.

### Mode Manuel (clic) — mesuré en second

Clic simulé : un point au cœur de chaque personnage de la référence, sur l'image clé où il est le
plus visible.

| catégorie | méthode | J | J&F | bord | scint. % | échecs |
|---|---|---|---|---|---|---|
| action floue | SAM 2.1 tiny, 1 clic (0.4) | 0.761 | 0.746 | 18.9 | 5.1 | 5/23 |
| action floue | SAM 2.1 large, 1 clic | 0.551 | 0.571 | 27.2 | 6.8 | 10/23 |
| action floue | **Manuel 0.5** | 0.752 | 0.742 | 19.3 | 5.7 | 3/23 |
| effets | SAM 2.1 tiny, 1 clic (0.4) | 0.139 | 0.149 | 50.0 | 11.2 | 14/14 |
| effets | SAM 2.1 large, 1 clic | 0.165 | 0.156 | 48.0 | 4.8 | 14/14 |
| effets | **Manuel 0.5** | 0.781 | 0.737 | 25.0 | 11.9 | 0/14 |
| **tous (22 plans)** | SAM 2.1 tiny, 1 clic (0.4) | 0.601 | 0.580 | 29.0 | 5.5 | 41/107 |
| **tous (22 plans)** | SAM 2.1 large, 1 clic | 0.689 | 0.677 | 21.0 | 3.1 | 30/107 |
| **tous (22 plans)** | **Manuel 0.5** | 0.828 | 0.801 | 15.5 | 3.8 | 10/107 |

Sur l'anime, SAM 2 seul sélectionne la plage de couleur sous le clic (un œil, une chaussure, une
mèche) : c'est la principale cause des 41 échecs de la 0.4. En 0.5, le clic choisit le personnage
entier trouvé par le détecteur, puis le suivi fait le reste : J 0,60 → 0,83 et échecs 41 → 10. Pas
de régression : la boîte (SAM 2 direct) et le pinceau fonctionnent comme avant.

### Verdict visuel par plan (Auto, sortie alpha sur damier et sur fond vert)

U = utilisable tel quel, R = retouche légère (quelques images ou un objet à masquer), X = inutilisable.

| plan | contenu | ISNet anime 0.4 | BiRefNet 0.4 | **Auto 0.5** |
|---|---|---|---|---|
| s01 | calme | U | U | U |
| s02 | calme | U | U | U |
| s04 | multi (4 personnages) | X | R | R |
| s05 | multi (2) | U | U | U |
| b5 | multi (2) | U | U | U |
| s06 | cheveux SSJ3 | U | U | U |
| b1 | cheveux longs sur ciel | U | U | U |
| s09 | action, éclairs | R | R | R |
| s10 | action, deux corps | R | R | R |
| s11 | action, rayon d'énergie | R | R | R |
| s12 | action, flou rouge | X | U | U |
| b6 | action, gros plan sombre | X | X | X |
| s13 | aura + débris | U | R | R |
| s15 | aura sur fond noir | X | U | U |
| b2 | forme démon, particules | X | R | R |
| s16 | sort de la fumée | U | U | U |
| s17 | derrière un mur | R | U | U |
| s18 | nuit, couloir | X | U | U |
| s19 | palais sombre, pilier | U | R | R |
| b4 | pluie de nuit, 2 personnages | R | X | X |
| s20 | collé à un tracteur | X | X | X |
| L1 | plan long 11 s | U | U | U |
| **total** | | 10 U / 5 R / 7 X | 12 U / 7 R / 3 X | 12 U / 7 R / 3 X |

Le verdict visuel de l'Auto 0.5 est **le même que celui de BiRefNet seul** plan par plan : 19 plans
sur 22 sont exploitables (12 tels quels, 7 avec une retouche légère : un objet fixe à masquer, des
éclairs ou une aura gardés sur quelques images), 3 sont inutilisables (le tracteur, le décor sombre
de *Black Clover* pris pour le personnage, un personnage perdu en fin de plan sous la pluie). Ce que
la 0.5 change à l'œil : moins de « respiration » du contour, plus d'image vide ni d'image
entièrement opaque sur un plan à fumée, et le personnage principal sélectionnable. ISNet, conseillé
par la doc 0.4, rendait 7 plans inutilisables.

Défaut trouvé pendant ce verdict et corrigé avant la mesure finale du Manuel : sur un plan
d'action à éclairs, les flashs sont pris pour des coupes et les sous-plans sans clic restaient vides ;
le personnage cliqué est maintenant reporté d'un sous-plan au suivant.


## 4. Candidats : gardés et rejetés

| Candidat | Licence (code / poids) | Mesure | Décision |
|---|---|---|---|
| **BiRefNet general** | MIT / MIT | meilleur détecteur image par image : J 0,85, 8 échecs | **gardé** : détecteur de l'Auto sur GPU et Mac |
| ISNet anime-seg | Apache-2.0 / Apache-2.0 (jeu de données CC0 annoncé) | J 0,75, 21 échecs, scintillement 12,5 % ; exclut bien les auras | gardé seulement sur CPU (≈ 10× plus léger) |
| ToonOut (BiRefNet affiné anime) | MIT / MIT, jeu de données CC-BY-4.0 | J 0,72, 23 échecs : garde auras et effets (13 échecs sur 14 images « effets ») | rejeté par la mesure, laissé en variante |
| Ensemble BiRefNet + ISNet | — | moyenne : J 0,850 (+0,004) mais scintillement 7,4 % ; minimum : J 0,77 | rejeté (gain nul) |
| **SAM 2.1 tiny / base+ / large** | Apache-2.0 / Apache-2.0 | seul, à partir d'un clic : J 0,58 / – / 0,66 ; comme suivi derrière le détecteur, tiny ≈ large (J 0,753 vs 0,750 sur 6 plans) | **gardé** comme suivi, taille selon la mémoire |
| SAMURAI | Apache-2.0, poids SAM 2.1 | non mesuré (amorce par boîte seulement, pas le temps) | à essayer plus tard |
| Cutie | MIT / poids entraînés notamment sur MOSE (CC-BY-NC-SA) | non mesuré | non prioritaire |
| SAM2Long / DAM4SAM / SeC | CC-BY-NC / Apache / Apache mais LLM Qwen sous licence de recherche | non mesurés | — |
| ViTMatte | MIT / Apache-2.0, entraîné sur Adobe DIM (conditions non publiques, probablement non commerciales) | non mesuré | — |
| **MatAnyone 2** | NTU S-Lab 1.0 (non commercial) / idem | voir §3 | voir §3 |
| **SAM 3 / 3.1** | SAM License (Meta) | **non mesurable** : dépôt Hugging Face à accès validé manuellement (HTTP 401 sans compte approuvé) | voir §6 |

MatAnyone 1 : le code actuel ne l'installe pas par défaut (`matanyone` absent des dépendances, simple
commentaire dans `pyproject.toml`) ; vérifié.

## 5. Recommandation par palier de mémoire

Rien à régler pour l'utilisateur : les deux réglages de modèle de Mask Mode valent *Auto* par défaut
et suivent le palier choisi automatiquement (tableau visible dans Settings > Models avec les licences).

| Palier | Détecteur | Suivi | Commentaire |
|---|---|---|---|
| NVIDIA 8 Go / Rapide | BiRefNet general | SAM 2.1 tiny | pic mesuré ≈ 3 Go de VRAM |
| NVIDIA 12 / 16 Go | BiRefNet general | SAM 2.1 base+ | |
| NVIDIA 24 Go / Max | BiRefNet general | SAM 2.1 large | le suivi « large » n'améliore pas l'Auto (J 0,750 contre 0,753 sur 6 plans) mais rend le Manuel par boîte plus fiable |
| Mac 8 / 16 Go | BiRefNet general | SAM 2.1 tiny | **non mesuré** (Mac injoignable) |
| Mac 24 / 32 Go | BiRefNet general | SAM 2.1 base+ | **non mesuré** |
| CPU | ISNet anime | aucun | BiRefNet et SAM 2 sont trop lents sur CPU ; comportement 0.4 |

Repli : si SAM 2 n'est pas installé, l'Auto retombe sur le détecteur image par image (0.4) ; si un
modèle manque, il est téléchargé à la demande (10 Mo/s, révision épinglée pour BiRefNet et ToonOut).

## 6. SAM 3 : licence et gain attendu

**Mesure** : impossible pendant ce chantier. Les poids `facebook/sam3` et `facebook/sam3.1` sont
derrière une validation manuelle sur Hugging Face (formulaire : nom, date de naissance, pays,
affiliation, fonction ; réponse 401 sans compte approuvé). Il faut donc un compte HF approuvé et son
jeton pour l'essayer ; le gain annoncé par Meta sur les bancs vidéo est de +4 à +5 points J&F sur
MOSE par rapport à SAM 2.1 (79,6 contre 74,5), mesuré sur des vidéos réelles, pas sur de l'anime.
Ici, le suivi n'est pas le goulot : passer de SAM 2.1 tiny à large ne change presque rien derrière le
détecteur. Le gain probable de SAM 3 serait surtout sa détection par concept texte (« anime
character »), qui pourrait remplacer ou compléter BiRefNet sur les cas où celui-ci échoue (aura,
objet collé au personnage). À mesurer avec ce banc dès qu'un accès est obtenu.

**Licence (SAM License, 19 novembre 2025)**, ce qu'elle dit vraiment :

- licence propriétaire de Meta (pas open source standard) couvrant code, poids et documentation ;
  utiliser ou distribuer vaut acceptation ;
- droit non exclusif, mondial, gratuit d'utiliser, reproduire, distribuer, modifier et créer des
  dérivés ; **aucune restriction non commerciale** (les mots « commercial » / « non-commercial »
  n'y figurent pas) : l'usage commercial est permis ;
- redistribution des poids et des dérivés permise **uniquement sous la même licence**, avec une
  copie du texte (une application MIT qui télécharge le modèle n'est pas « contaminée », mais du
  code SAM 3 intégré au dépôt resterait sous SAM License) ;
- obligations : citer SAM dans les publications de recherche ; respecter le contrôle des
  exportations et les sanctions ; **interdiction** des usages militaires, nucléaires, d'espionnage,
  d'armement et des activités ITAR ; pas d'ingénierie inverse ;
- pas de politique d'usage séparée ; Meta ne revendique rien sur les masques produits ; une action en
  contrefaçon contre Meta met fin à la licence ; l'utilisateur garantit Meta contre les réclamations
  de tiers ; droit californien ; Meta peut modifier la licence unilatéralement.

Pour AMV Tools (gratuit, poids non redistribués, téléchargés par l'utilisateur) : compatible, à
condition d'afficher ces conditions dans Settings > Models et de faire passer l'utilisateur par son
propre compte Hugging Face approuvé. **Non intégré** (alors que la licence le permettait pour un
outil non commercial) : poids inaccessibles sans compte approuvé, donc aucun gain mesuré sur lequel
fonder le choix. Intégration possible ensuite derrière le même pipeline (détecteur → suivi).


## 7. Mask Mode activable par défaut ?

**Non, pas encore.** Sur 22 plans réels, l'Auto donne une sortie utilisable telle quelle ou avec une
retouche légère sur la grande majorité des plans calmes, de dialogue, multi-personnages et d'action
(voir le verdict), ce qui suffit pour un outil « bêta » qu'on active soi-même. Mais trois plans sur
22 restent inutilisables sans intervention (objet collé au personnage, décor sombre pris pour le
personnage, personnage secondaire perdu), le gain sur le réglage par défaut réel de la 0.4 est
faible en moyenne, rien n'a été validé sur Mac, et le traitement est ≈ 2,5× plus lent que la 0.4.
Prochaine étape la plus rentable : un meilleur **détecteur** (SAM 3 avec le concept texte
« anime character » dès qu'un accès aux poids est obtenu, ou un BiRefNet affiné sur des personnages
avec auras), mesurable tel quel avec ce banc.

## 8. Ce qui a changé dans le code

- `backend/models/auto_roto.py` (nouveau) : sélection automatique des personnages, amorçage et
  ré-amorçage du suivi, découpe aux coupes franches, fusion détecteur/suivi, modèles par palier.
- `backend/models/sam2.py` : suivi par fenêtres (mémoire bornée) et réutilisation de l'état SAM 2
  entre deux passes sur le même plan.
- `backend/api/segment.py` : moteur Auto = détection + suivi ; Manuel = clic → personnage détecté ;
  corrections au pinceau = images clés.
- `backend/finish/runner.py` : le travail « Cutout » de la boîte de finition utilise le même pipeline.
- `backend/models/birefnet.py` : variante ToonOut (poids MIT, révision épinglée).
- UI : bascule *Personnage principal / Tous les personnages* dans Mask Mode, choix *Auto* des
  modèles et lignes « Détourage » dans le tableau des paliers (FR/EN).
- `tests/eval/roto_eval.py` : le banc complet ; `tests/test_roto.py` : tests unitaires (modèles simulés).

## 9. Planches (hors dépôt, sur la machine de mesure)

- Avant (référence, BiRefNet, ISNet anime, SAM 2.1 tiny au clic) : `/srv/data/amv-tools/reports/roto/avant/<plan>.jpg`
- Après (référence, BiRefNet 0.4, Auto 0.5, Manuel 0.5) : `/srv/data/amv-tools/reports/roto/apres/<plan>.jpg`
- Vérité terrain (planches de contrôle image + superposition) : `/srv/data/amv-tools/roto/review/<plan>.jpg`

Chaque planche montre 3 images clés, chacune composée sur damier puis sur fond vert.

## Résumé

- Les chiffres « 0,98 d'IoU » de la 0.4 venaient de clips synthétiques. Sur 22 plans réels notés
  contre 135 masques de référence vérifiés à l'œil, l'Auto conseillé par la doc 0.4 (ISNet) donnait
  J 0,75 et perdait le personnage sur 1 image clé sur 5 ; le Manuel (SAM 2.1 tiny) J 0,60.
- **Auto 0.5** = BiRefNet (meilleur détecteur mesuré) + suivi SAM 2.1 amorcé et ré-amorcé
  automatiquement, options *Personnage principal / Tous les personnages*, corrections au pinceau
  re-propagées : J 0,85, 7 échecs sur 107, scintillement 4,2 %. C'est nettement mieux que ISNet,
  mais seulement un peu mieux que BiRefNet seul (réglage par défaut réel de la 0.4) ; verdict visuel
  identique : 19 plans sur 22 exploitables, 3 inutilisables.
- **Manuel 0.5** : le clic prend le personnage détecté entier ; J 0,60 → 0,83, échecs 41 → 10.
- **Plus de plafond de 4 s** : suivi par fenêtres, 11 s en 2,1 Go de VRAM au lieu de 9,0 Go.
- Candidats écartés à la mesure : ToonOut (garde les auras), MatAnyone 2 (perd le personnage en
  action), ensemble de détecteurs (gain nul). SAM 3 non mesurable (accès Hugging Face validé à la
  main) ; sa licence permet l'usage, même commercial, sous conditions (voir §6).
- Paliers : BiRefNet + SAM 2.1 tiny / base+ / large selon la VRAM, ISNet sans suivi sur CPU ; Mac non
  mesuré (injoignable).
- **Recommandation : laisser Mask Mode désactivé par défaut** pour l'instant ; le prochain gain viendra
  d'un meilleur détecteur, mesurable avec le banc livré (`tests/eval/roto_eval.py`).
