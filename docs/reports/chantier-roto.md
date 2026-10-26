# Chantier roto : rendre le détourage (Mask Mode) utilisable sur de vrais épisodes

Ce rapport a quatre parties, la plus récente en tête : **partie 4 (objectifs 5 % et 1 s, audit de la
métrique de contamination)**, partie 3 (qualité max), partie 2 (SAM 3 et fond dans les cheveux),
partie 1 (chantier roto, 0.5, le banc).

# Partie 4 — Objectifs chiffrés : ≤ 5 % de fond et de contamination, < 1 s par image

Date : 26 octobre 2026, 2 h – 7 h. Objectifs fixés : fond le long du contour ≤ 5 %, fond enfermé
≤ 5 %, contamination couleur ≤ 5 % (moyenne des 22 plans, sans plan aberrant), puis moins d'une
seconde par image sur la RTX 3090 sans perdre plus de 0,3 point.

## 1. Audit de la métrique de contamination

**Ce qu'elle mesurait** (parties 2 et 3) : pour chaque pixel semi-transparent près du contour, la
couleur exportée C est projetée sur l'axe F → B, où B est la couleur moyenne du fond de référence
autour et **F est deviné** : la couleur du pixel de référence le plus proche situé 3 px ou plus à
l'intérieur du contour.

**Le défaut, prouvé de trois façons :**

1. **Décomposition** des 19,0 % de la qualité max (104 images clés) : **52,7 %** de la contamination
   comptée vient de pixels exportés *plus sombres que F et que B* — du trait d'encre. Un pixel de bord
   correctement exporté en noir d'encre, sur un fond plus sombre que les cheveux, se projette vers le
   fond : il est compté comme « couleur du fond » alors qu'il est juste.
2. **Plancher sur une réponse parfaite** : chaque image clé est recomposée avec des couleurs connues
   (personnage tel qu'il est dans la référence, prolongé vers l'extérieur par la couleur de son pixel
   le plus proche ; fond réel de l'image ; alpha connu, la référence adoucie sur ~2 px). Exporter
   exactement les vraies couleurs donne une contamination vraie de 0 — l'ancienne mesure y affiche
   **25,9 %** (médiane 24,4 %), et une variante qui ajoute l'encre comme couleur candidate encore
   12,2 %. Pire : l'export qualité max y obtient un *meilleur* score (8,4 %) que la réponse parfaite,
   la mesure récompensait la sur-correction.
3. Conséquence : les 19 % de la partie 3 ne mesuraient pas la couleur de fond restante.

**Nouvelle mesure (à vérité connue)** : l'export du pipeline est appliqué au composite de chaque
image clé ; on connaît la vraie couleur F et le vrai fond B de chaque pixel, la part de fond de la
couleur exportée est exacte (0 pour une réponse parfaite) ; on mesure aussi l'erreur de couleur
|C − F| pour attraper une sur-correction. Deux variantes :

- **contamination couleur** (l'export reçoit l'alpha vrai) : ce que l'étape de couleur laisse de fond ;
- contamination avec l'alpha du pipeline (calculé sur la vraie image) : y ajoute les erreurs de
  *position* de l'alpha. Celle-ci a un plancher de mesure prouvé : avec un alpha **parfait décalé de
  1 px vers l'extérieur**, l'export retenu passe de 2,5 % à **20,2 %** (38,9 % à 2 px) ; or la
  référence n'est juste qu'à ~2 px (partie 2, §1). Elle est donnée à titre indicatif, pas comme
  objectif.

L'ancienne mesure reste dans les tableaux (« contam. ancienne ») pour la comparaison.

## 2. Phase 1 — résultats (22 plans, 104 à 107 images clés)

Moyenne / médiane / pire plan. « contam. couleur » = nouvelle mesure, alpha exact (§1) ; « contam.
alpha pipeline » = même mesure avec l'alpha du pipeline (plancher de mesure ≈ 20 % pour 1 px
d'écart) ; « contam. ancienne » = l'estimation des parties 2-3, gardée pour comparaison.

| méthode | J | échecs | scint. % | fond contour % | fond intérieur % | mangé % | contam. couleur % | contam. alpha pipeline % | contam. ancienne % |
|---|---|---|---|---|---|---|---|---|---|
| Auto 0.5 (export 0.5) | 0,853 | 7/107 | 4,2 | 18,6 / 14,2 / 48,2 (b6) | 24,8 / 17,1 / 79,6 (s09) | 10,6 | 13,0 / 14,4 / 22,3 | 26,6 | 28,4 |
| filtre SAM 3 (0.6 standard) | 0,920 | 1/107 | 2,8 | 10,6 / 8,1 / 44,5 (b6) | 7,8 / 2,8 / 26,5 (b6) | 7,6 | 13,5 / 13,8 / 26,1 | 33,2 | 30,9 |
| qualité max, sans décontamination | 0,924 | 1/107 | 2,5 | 6,7 / 2,2 / 38,3 (b6) | 5,3 / 0,7 / 25,0 (s18) | 8,0 | 32,3 / 32,4 / 33,1 | 38,9 | 37,4 |
| qualité max, export standard (estimation du premier plan) | 0,924 | 1/107 | 2,5 | 6,7 / 2,2 / 38,3 | 5,3 / 0,7 / 25,0 | 8,0 | 13,5 / 13,8 / 26,1 | 32,2 | 27,4 |
| qualité max, export de la partie 3 | 0,924 | 1/107 | 2,5 | 6,1 / 1,9 / 36,0 (b6) | 5,1 / 0,5 / 25,0 (s18) | 8,4 | 4,7 / 4,6 / 9,3 (b5) | 20,0 | 19,1 |
| **qualité max, export démélangé (partie 4)** | **0,924** | **1/107** | **2,5** | **6,1 / 1,9 / 36,0 (b6)** | **5,1 / 0,5 / 25,0 (s18)** | 8,4 | **2,6 / 2,7 / 4,6 (s20)** | 22,2 | 22,3 |

**Export de la partie 4** : après la résolution de l'alpha par la couleur (encre protégée), la couleur
de chaque pixel semi-transparent est **démélangée contre le fond visible autour** (moyenne des
pixels de fond à moins de 5 px ; le plus proche au-delà) : F = (C − (1 − α) B) / α, jusqu'à α = 0,05,
avec la couleur opaque voisine en dessous. Vérifié sur les zooms fond noir et blanc (b1, s06, s01,
s13) : même rendu que l'export de la partie 3, aucun trait mangé ni pointillé. Variantes mesurées :
démélange sans résolution de l'alpha (2,5 → 4,9 %), seuils α 0,10-0,40 (2,9 %), 0,15-0,50 (4,1 %).

**Objectifs de la phase 1 :**

- **contamination couleur ≤ 5 % : atteint** — 2,6 % en moyenne, 2,7 % en médiane, **4,6 % au pire
  plan** (aucun plan au-dessus de 5 %) ;
- **fond le long du contour ≤ 5 % : non atteint en moyenne (6,1 %)**, atteint en médiane (1,9 %) ;
- **fond enfermé ≤ 5 % : non atteint de justesse en moyenne (5,1 %)**, médiane 0,5 %.

Les moyennes sont tirées par les plans de nuit, d'action floue et d'effets (10 plans sur 22 au-dessus
de 5 % de fond contour, 7 au-dessus de 5 % de fond enfermé, liste au §4), avec deux extrêmes :
**b6** (fond contour 36 % : cheveux presque noirs sur fond presque noir — le détecteur, SAM 3 et la
couleur ne séparent pas les deux, et la référence y est elle-même incertaine de plusieurs pixels,
partie 1 §2) et **s18** (fond enfermé 25 % : couloir de nuit). Aucune des variantes essayées en
partie 3 (rayons, seuils, prompts, votes, ensembles) ne les fait baisser sans manger le personnage
ailleurs.

## 3. Phase 2 — moins d'une seconde par image

**Mesure** : le pipeline qualité max tel que l'application l'enchaîne (BiRefNet → SAM 3 zoomé →
suivi SAM 2.1 tiny → contour serré → stabilisation temporelle), sur **les 22 plans, toutes les
images sorties** (chaque image a son masque, aucune n'est sautée), plus l'export « max » à la
résolution source (1920 × 1080, mêmes threads que l'application, mesuré sur 24 images par plan et
ramené à toutes les images). Ancien et nouveau code ont tourné l'un après l'autre, jamais en même
temps, et la VRAM des autres processus a été relevée au début et à la fin de chaque mesure (une autre
session partage la carte). Temps en ms par image, sur toutes les images des 22 plans :

| étape | ancien code (partie 3) | **nouveau code** |
|---|---|---|
| BiRefNet | 135 | 107 |
| SAM 3, deux passes (image entière + zoom par personnage) | 433 | 276 (dont ≈ 43 de compilation, 71 s une fois, amortie sur les 1 638 images) |
| suivi SAM 2.1 tiny | 130 | 94 |
| contour serré + fragments | 21 | 16 |
| stabilisation temporelle | 55 | 45 |
| **pipeline (masques de toutes les images)** | **775** | **539** |
| export « max » 1920 × 1080 (CPU) | 426 | 166 |
| **total, export compris** | **1 201** | **705** |
| pic de VRAM du processus (les trois modèles chargés ensemble) | 3,8 Go | 7,2 Go |

Conditions : 22 plans, 1 638 images ; aucune autre application sur la carte au début ni à la fin des
deux mesures (l'autre session était arrêtée à ce moment-là ; pendant les essais précédents elle
occupait 0,8 à 5 Go et les temps variaient de 20 à 60 %). Le suivi et la stabilisation, dont le code
n'a pas changé, sont 20 à 30 % plus rapides dans la seconde mesure : c'est l'ordre de grandeur du
bruit de mesure de cette machine (charge CPU de l'hôte), à garder en tête pour ces deux lignes. Les
gains de SAM 3 (× 1,6) et de l'export (× 2,6) sont nettement au-dessus. Dans l'application, sur les
cartes de 16 Go et plus, la compilation de SAM 3 (~1 min) n'est payée qu'au premier clip de la
session ; sur 8 et 12 Go il n'y a pas de compilation et les lots sont de 2 (non mesuré sur ces
cartes).

**Ce qui a fait gagner** :

- SAM 3 : texte encodé une fois par prompt au lieu de chaque image, images et recadrages du zoom
  traités par lots (8 sur les cartes de 12 Go et plus), encodeur d'image compilé (`torch.compile`,
  lots toujours pleins pour une seule compilation ; ≈ 1 min une fois par chargement, le modèle reste
  chargé entre les clips sur les cartes de 16 Go et plus ; retour automatique au mode normal en cas
  d'erreur). Masques identiques (IoU ≥ 0,993 contre la version image par image).
- BiRefNet : 4 images par passage.
- Export : la résolution de l'alpha par la couleur ne travaille que sur le cadre de la bande de
  contour (alpha identique au pixel près), le démélange uniquement sur les pixels semi-transparents,
  jusqu'à 5 threads.
- **Refusé** : ne zoomer que les recadrages de moins des deux tiers de l'image (SAM 3 deux fois plus
  rapide encore) — le gros plan sombre s19 perdait la passe zoomée et son fond enfermé passait de
  2,9 à 14,5 % (moyenne 5,1 → 5,7 %, au-delà des 0,3 point tolérés).

**Qualité inchangée** (même banc, même mesure, moyenne / médiane / pire plan) :

| | J | F | échecs | scint. % | fond contour % | fond intérieur % | mangé % | contam. couleur % |
|---|---|---|---|---|---|---|---|---|
| avant optimisation | 0,924 | 0,863 | 1/107 | 2,5 | 6,1 / 1,9 / 36,0 | 5,1 / 0,5 / 25,0 | 8,4 | 2,6 / 2,7 / 4,6 |
| **après optimisation** | **0,924** | **0,862** | **1/107** | **2,6** | **6,1 / 1,9 / 35,8** | **5,1 / 0,5 / 25,0** | **8,4** | **2,7 / 2,7 / 4,6** |

Écart maximal 0,1 point.

## 4. Bilan des objectifs

| objectif | atteint ? | chiffre | pourquoi, si non |
|---|---|---|---|
| contamination couleur ≤ 5 % | **oui** | 2,7 % (médiane 2,7 %, pire plan 4,6 %) | — |
| fond le long du contour ≤ 5 % | **non** | 6,1 % (médiane 1,9 %) ; au-dessus de 5 % sur 10 plans : b6 35,8, s20 12,3, s18 11,6, s09 10,7, s17 10,0, b4 9,8, b2 9,4, s11 7,6, s19 7,1, L1 6,1 | les plans calmes, de dialogue et de cheveux sont sous 5 % (sauf le plan long, 6,1) ; au-dessus : nuit (b6, s18, b4 : cheveux ou vêtements sombres sur fond sombre, que ni BiRefNet, ni SAM 3, ni la couleur ne séparent, et où la référence est incertaine de plusieurs pixels), action floue et effets (flou de mouvement, éclairs, particules mêlés au contour), le tracteur (s20) |
| fond enfermé ≤ 5 % | **non, de justesse** | 5,1 % (médiane 0,5 %) ; au-dessus de 5 % sur 7 plans : s18 25,0, L1 20,0, b6 13,4, s11 12,8, s20 10,5, b2 9,9, s09 9,7 | nuit (s18, b6), rayon et éclairs (s11, s09), particules (b2), tracteur (s20) ; sur le plan long, la référence marque les mains jointes de Goku comme « fond » (§ partie 3) |
| contamination couleur, aucun plan > 5 % | **oui** | pire plan 4,6 % (s20) | — |
| < 1 s par image sur RTX 3090 | **oui** | 539 ms (masques), 705 ms export 1080p compris | — |
| qualité à ± 0,3 point | **oui** | écart max 0,1 point | — |

**Défaut visible restant** (planches, déjà présent en partie 3) : une **mèche fine isolée** qui part
de la tête (s01, 1 à 2 px de large) est coupée par le contour serré sur SAM 3, avec quelques
pixels blancs dans le trait à sa jonction ; BiRefNet seul la garde. Deux corrections essayées :
rendre les pointes dès 60 % / 75 % d'opacité (la mèche revient en partie, mais le fond enfermé passe
de 5,1 à 5,6 / 5,4 %), rendre les structures fines quelle que soit leur opacité (même effet, 5,6 %) —
refusées, au-delà des 0,3 point tolérés.

Planches de la partie 4 (sur la machine de mesure, `reports/roto/roto4/`) : `black/` et `white/`,
zooms ×4 sur le contour le plus découpé de 3 images clés par plan — export de la partie 3 contre
export final démélangé, sur fond noir et sur fond blanc.

Ce qui n'a pas été fait faute de temps : matting appris sur la bande de contour (ViTMatte, BiRefNet
HR-matting, MatAnyone 2) combiné à l'encre protégée — inutile pour la contamination couleur une fois
le démélange en place (2,7 %), mais pas essayé contre le fond des plans de nuit ; plaque de fond
temporelle (décor reconstruit sur la durée du plan) ; TensorRT.




# Partie 3 — Qualité max : le moins de fond possible, sans contrainte de temps

Date : nuit du 25 au 26 octobre 2026. Même machine (RTX 3090, bridée en horloge), même banc de 22 plans
notés, mêmes métriques (partie 2, §2). Consigne : faire baisser ensemble le fond dans le masque
(contour et intérieur), la contamination de couleur des bords, les échecs et le scintillement, sans
perdre de cheveux ; le temps de calcul et la VRAM ne comptent plus ; **une variante meilleure au
chiffre mais avec un artefact visible est refusée** (planches zoomées ×3 à ×6 sur fond magenta,
noir et blanc, et planches de plans entiers).

## 1. Le pipeline « Qualité max »

1. **BiRefNet** détoure chaque image (le contour : cheveux, trait d'encre, alpha doux).
2. **SAM 3 zoomé** (concept « anime character ») : passe sur l'image entière, puis seconde passe sur
   le recadrage de chaque personnage trouvé — SAM 3 travaille à 1008 px fixes, un personnage
   recadré y gagne 1,5 à 3 fois en résolution.
3. Filtre large (2,5 % de la diagonale) autour de SAM 3 avant le suivi, **suivi SAM 2.1** comme en
   partie 2.
4. **Contour serré à 0,4 % du contour SAM 3, pointes rendues** : un morceau du masque BiRefNet hors
   de ce contour serré revient seulement s'il est opaque (une mèche, un doigt — pas une lueur ni un
   halo semi-transparents), reste à moins de 2,5 % de SAM 3, et **dépasse** du personnage (moins de
   30 % de son pourtour touche le personnage : une poche de fond enfermée entre les bras touche de
   partout et reste coupée).
5. **Fragments isolés supprimés** : petits morceaux détachés du personnage et hors de SAM 3 (bouts
   d'auréole, d'aura).
6. **Export** (ProRes 4444, VP9, PNG) : l'alpha d'une bande de 2 px à l'intérieur et 12 px à
   l'extérieur du contour est **recalculé à partir des couleurs locales** (personnage / fond), sauf
   sur les pixels d'encre (plus sombres que le personnage et le fond autour : un trait noir n'est
   ni l'un ni l'autre), puis estimation de la couleur de premier plan.

## 2. Du 0.4 à la qualité max (22 plans notés, alpha exporté)

| étape | J | F | échecs | scint. % | fond contour % | fond intérieur % | mangé % | contam. % | temps / image |
|---|---|---|---|---|---|---|---|---|---|
| 0.4, ISNet anime (conseillé par la doc) | 0,751 | 0,669 | 21/107 | 13,0 | 21,8 | 24,9 | 22,4 | 40,8 | ≈ 70 ms (chargé) |
| 0.4, BiRefNet seul (réglage par défaut) | 0,847 | 0,794 | 8/107 | 4,6 | 19,7 | 23,5 | 10,2 | 29,7 | 119 ms |
| Auto BiRefNet + suivi SAM 2 (0.5) | 0,853 | 0,793 | 7/107 | 4,2 | 18,6 | 24,8 | 10,6 | 28,4 | 225 ms |
| filtre SAM 3 (0.6 standard, partie 2) | 0,920 | 0,862 | 1/107 | 2,8 | 10,6 | 7,8 | 7,6 | 30,9 | 435 ms |
| filtre SAM 3 serré 0,4 % (variante notée tgate4c0) | 0,915 | 0,857 | 2/107 | 2,6 | 6,6 | 5,9 | 7,7 | 28,0 | ≈ 440 ms |
| **qualité max** (tel que l'application, stabilisation temporelle comprise) | **0,924** | **0,863** | **1/107** | **2,5** | **6,1** | **5,1** | 8,4 | **19,1** | 651 ms (+ export 580 ms / image 1080p) |

Plans simples (calmes, dialogue, multi-personnages, cheveux, plan long) :

| étape | J | F | échecs | fond contour % | fond intérieur % | mangé % | contam. % |
|---|---|---|---|---|---|---|---|
| 0.4, ISNet anime | 0,941 | 0,869 | 0/40 | 15,3 | 15,9 | 9,2 | 28,0 |
| 0.4, BiRefNet seul | 0,970 | 0,898 | 0/40 | 11,0 | 10,7 | 6,8 | 25,9 |
| Auto BiRefNet + suivi (0.5) | 0,968 | 0,895 | 0/40 | 9,2 | 10,1 | 7,8 | 23,7 |
| filtre SAM 3 (0.6 standard) | 0,974 | 0,911 | 0/40 | 2,7 | 3,0 | 3,4 | 23,9 |
| **qualité max** | **0,974** | **0,917** | 0/40 | **1,7** | **2,5** | 4,2 | **16,3** |

Les lignes 0.4 et 0.5 sont mesurées avec l'export de leur époque (filtre guidé) ; les suivantes avec
l'export 0.6 ou « max ». Temps : passe au calme sur un plan de 84 images (sauf mention « chargé ») ;
qualité max = BiRefNet 118 ms + SAM 3 zoomé 419 ms (deux passes ; 216 ms pour une) + suivi SAM 2
98 ms + contour serré et fragments 15 ms ; l'export « max » prend 580 ms par image 1080p sur CPU
(294 ms pour l'export standard). Pic de VRAM : 2,0 Go (SAM 3), les modèles étant chargés l'un après
l'autre — la qualité max tient sur une carte de 8 Go. Sans la stabilisation temporelle de
l'application (le banc brut), le scintillement de la qualité max est de 2,7 %.

## 3. Ce qui a été essayé, gardé ou refusé

Toutes les lignes ci-dessous sont comparées à la même base (filtre SAM 3 suivi, 22 plans) ; « vu »
= vérifié sur les planches zoomées.

| essai | mesure (22 plans) | à l'œil | décision |
|---|---|---|---|
| contour serré 0,4 % sans rendre les pointes | fond contour 6,6 %, intérieur 5,9 % | **rogne les pointes de mèches** que SAM 3 sous-segmente (s06) | remplacé par ↓ |
| **contour serré 0,4 % + pointes opaques rendues** (seuil de contact 30 %) | J 0,923, fond contour 6,8 %, intérieur 6,1 % | pointes gardées, lueurs et poches coupées | **gardé** |
| seuils de contact 45 % / 60 %, opacité 0,8 | fond contour 6,5 / 10,6 %, J jusqu'à 0,912 | — | refusés (plus de fond) |
| **suppression des fragments isolés** | identique au chiffre près | bouts d'auréole flottants autour de Whis supprimés (s05) | **gardé** |
| rendre les objets tenus (poche opaque dont la couleur n'est pas celle du fond autour) | fond intérieur 5,7 → 7,2 % | rend le bol de Beerus mais aussi des poches de fond | refusé |
| résolution couleur de l'alpha, bande ±10 px (la variante c5 de la partie 2) | contamination 30,9 → 10,6 % | **mange le trait d'encre**, mangé 18,6 % | refusé |
| idem, bande 2 px dedans / 8 px dehors | contamination 17,3 % | **trait d'encre en pointillés** (s06) | refusé |
| **idem + encre protégée**, bande 2 / 12 px | contamination 27,5 → 20,7 %, mangé 8,2 % | trait continu, plus de liseré bleu sur fond noir (b1) | **gardé** |
| bande 3 px dedans | contamination 15,4 %, mangé 10,1 % | **ronge les pointes jaunes sur l'aura jaune** (s06) | refusé |
| couleur du pixel opaque le plus proche pour tous les pixels du bord | contamination 18,8 % | **contour crénelé** (b1) | refusé |
| **encre protégée + démélange contre le fond le plus proche + estimation du premier plan** | contamination 20,6 → 19,1 %, alpha identique | trait plus net contre le blanc, aucun artefact (b1, s02, s06) | **gardé** |
| démélange seul (sans estimation du premier plan) | contamination 23,9 % | — | refusé |
| portée des pointes rendues 1,0 / 1,5 % au lieu de 2,5 % | identique (fond contour 6,0 %) | — | refusé (garde les longues pointes) |
| niveaux sur l'alpha (seuils 10-90 %, 20-80 %) | contamination 34 à 37 % | — | refusé |
| **SAM 3 zoomé** (seconde passe par personnage) | fond intérieur 5,7 → 5,1 % (sombre 13,4 → 9,5 %) | identique | **gardé** (2 passes SAM 3) |
| fusion de prompts « anime character » + « person » | fond contour 7,0 %, J 0,922 | — | refusé |
| fusion « anime character » + « hair » | fond contour 6,6 % (cheveux 1,3 → 1,8 %), J 0,922 | — | refusé |
| « person » seul | J 0,917 | — | refusé |
| SAM 3 vidéo comme sélecteur (cohérence temporelle) | mangé 11,5 %, J 0,910 | — | refusé |
| union SAM 3 image + vidéo | J 0,922, fond contour 7,3 % | — | refusé |
| vote temporel du sélecteur (comblement, majorité sur 3 images) | identique ou pire (scint. 2,7 → 3,1 %) | — | refusé |
| **ensemble BiRefNet + BiRefNet HR** (moyenne) | J 0,924 → **0,938** (effets 0,80 → 0,89) | **ailes du démon à moitié transparentes** (b2) : la moyenne de deux avis contraires donne un fantôme | refusé |
| BiRefNet HR seul / HR-matting (entrée 2048) | HR : J 0,848, contamination 34,7 % ; HR-matting : J 0,816, 11 échecs, scint. 8,1 % | — | refusés |

Ce qui reste dans les chiffres vient en partie de la **référence** (SAM 2.1, binaire) : sur le plan
long, les mains jointes de Goku y sont marquées « fond » (une image clé à 92 % de « fond
intérieur » alors que le pipeline a raison) et l'auréole de Kaio y compte comme personnage ; sur s13,
du sol entre les jambes y est inclus. Le « mangé » des effets (29 %) est surtout l'aura qui enveloppe
les mains et les objets tenus que SAM 3 exclut.

## 4. Verdict visuel par plan (qualité max contre filtre SAM 3)

La qualité max est **égale ou meilleure sur chaque plan** : plus de bouts d'auréole autour de Whis
(s05), bords nets sans liseré sur fond noir ou blanc (b1, s01, s06), lueur de l'aura retirée autour
des pointes (s06), bords plus propres sur b6. Restent : le bâton de Whis et le bol de Beerus coupés
(objets tenus, et le bol laisse un trou dans le torse sur s19), un amas de particules gardé sur une
image de b2 (SAM 3 le compte comme personnage), une lueur verte aux pieds sur s13, le personnage
secondaire perdu en fin de plan sous la pluie (b4). Compte inchangé : **13 U / 8 R / 1 X** (comme
le filtre SAM 3) — les gains sont à l'intérieur des plans (liseré, lueurs, fragments, poches), pas
assez pour faire changer de classe un plan dont il manque un objet tenu (s05, s19).

## 5. Intégration

- Réglage `mask_quality` (*Détourage qualité max*, Settings > Models) : *Auto* = qualité max sur les
  paliers NVIDIA 8, 12, 16, 24 Go et Max **quand SAM 3 est accessible** ; sinon, et sur le palier
  Rapide, Mac et CPU, **repli automatique** sur le pipeline standard (ou sur l'Auto 0.5 sans SAM 3).
  L'export suit (`mask_edge_mode` = *auto*).
- VRAM : les modèles sont chargés l'un après l'autre (BiRefNet 1,6 Go, SAM 3 2,0 Go, SAM 2 1,7 Go) ;
  la qualité max ne coûte pas plus de VRAM que le standard, seulement du temps (deux passes SAM 3,
  étapes CPU).
- Mode manuel : le clic passe par le même chemin (SAM 3 zoomé, contour serré, export max) — **pas de
  régression** : J 0,879 → 0,883, échecs 5 → 5, fond contour 10,0 → 5,8 %, fond intérieur 7,8 → 5,1 %,
  contamination 31,2 → 19,2 %, mangé 10,8 → 11,6 %.

## 6. Mask Mode Auto activable par défaut ?

**Oui, sur les cartes NVIDIA de 8 Go et plus où SAM 3 est accessible : l'Auto « qualité max » est
activable par défaut.** Sur 22 plans réels, il ne perd le personnage qu'une fois sur 107 images
clés, et le défaut qui motivait ces trois chantiers — du fond dans les cheveux et autour des
personnages, même sur des plans simples — est divisé par cinq à six depuis la 0.5 sur les plans
simples (fond le long du contour 9,2 → 1,7 %, fond enfermé 10,1 → 2,5 %, couleur de fond dans les
bords 23,7 → 16,3 %), sans perte de cheveux à l'œil sur fond noir, blanc et magenta. Restent : les
objets tenus coupés, la lueur d'aura aux pieds sur un plan, un personnage secondaire perdu sous la
pluie de nuit, et un traitement 2,9 fois plus long qu'en 0.5 (≈ 0,65 s par image sur RTX 3090).

**Non ailleurs** : sans SAM 3 (Mac non mesuré, CPU, palier Rapide, jeton absent), l'application
retombe automatiquement sur le pipeline standard ou sur l'Auto 0.5, dont les verdicts des parties 2
et 1 restent valables. Comme l'accès à SAM 3 dépend du jeton Hugging Face de chaque utilisateur,
`enable_mask` reste désactivé dans le code ; l'activer automatiquement quand le palier a SAM 3 et
que le jeton ouvre ses poids est la dernière marche.

## 7. Planches

Dans `reports/roto/roto3/` sur la machine de mesure : `cheveux-avant-apres/` (zooms ×3, filtre SAM 3
contre qualité max, sur magenta), `noir/` et `blanc/` (mêmes zooms ×4 sur fond noir et blanc, pour le
liseré), `plans/` (images entières), `verdict/` (planches compactes du §4).


---


# Partie 2 — SAM 3, détecteur anime, et le fond qui reste dans les cheveux

Date : 25 octobre 2026 (soir). Machine de mesure : RTX 3090 24 Go (Linux, 6 vCPU). Mac injoignable
(une tentative) : **rien n'a été mesuré sur Mac**.

Problème de départ, en vrai usage : « il y avait tout le temps un morceau du fond dans les cheveux ou
ailleurs, même dans des scènes simples ». Cible : le fond qui reste dans le détourage (trous entre
les mèches, fond entre les bras ou sous le menton, liseré autour du contour, couleur de fond qui bave
dans les pixels semi-transparents), y compris sur les plans calmes.

## 1. Le banc : d'où viennent les références, et ce qu'elles valent dans les cheveux

Même banc que la partie 1 : 23 plans réels (22 notés), 135 masques de référence sur 6 images clés par
plan, taille de travail 1080 × 608.

**Construction des références (vérifiée dans le code et les fichiers de correction)** : SAM 2.1 large,
amorcé par les boîtes des composantes de ToonOut (un BiRefNet affiné sur l'anime) ou, pour 14 plans
sur 23, par des boîtes et des points placés à la main, puis nettoyage des grains. Aucun masque n'est
pris de BiRefNet ni de ToonOut eux-mêmes (le mode « sam+toon », qui aurait recopié le contour de
ToonOut, n'est utilisé sur aucun plan). Conséquences :

- **pas de biais pour BiRefNet** (le détourage retenu) ; ToonOut ne sert qu'à cadrer ;
- **un biais pour la famille SAM** sur le tracé exact du bord : les références sont binaires, en
  léger escalier, et à ±2 px près. Contrôle à l'œil des zones cheveux (planches zoomées ×3) : sur
  les mèches épaisses (s06, b1, s02) la référence suit bien le contour ; elle **oublie les mèches
  fines isolées** (une mèche d'un ou deux pixels au-dessus de la tête dans s01, que BiRefNet garde à
  raison) et, sur le plan d'aura s13, elle inclut à tort des morceaux de sol entre les jambes.
  SAM 3 (même famille, mêmes défauts : il retire le trait d'encre et rogne les pointes) est donc
  avantagé par la mesure du bord ; c'est pourquoi le choix final repose aussi sur les planches.

Correction appliquée à la métrique plutôt qu'aux références (pas le temps de redessiner 135 masques
au pixel) : les 2 premiers pixels de part et d'autre du contour de référence ne sont pas notés.

## 2. Nouvelle métrique : le fond dans le masque

Sur chaque image clé notée, sur l'alpha **exporté** (après le traitement de l'export) :

- **fond contour** : alpha moyen des pixels de fond situés entre 2 et 8 px **à l'extérieur** du
  contour de référence (trous entre les mèches, liseré, halo) ;
- **fond intérieur** : alpha moyen des pixels de fond **enfermés dans l'enveloppe** du personnage
  (silhouette fermée par un disque de 41 px puis trous remplis) à plus de 8 px du contour : fond
  entre les bras ou les jambes, sous le menton, grands trous dans les cheveux ;
- **mangé** : 1 − alpha moyen des pixels du personnage entre 2 et 8 px **à l'intérieur** du contour
  (vérifie qu'on n'enlève pas le fond en mangeant les cheveux) ;
- **contamination** : pour chaque pixel semi-transparent du bord (0,05 < α < 0,95), position de sa
  couleur exportée entre la couleur du personnage F (pixel de référence le plus proche, ≥ 3 px à
  l'intérieur : le trait d'encre s'il y en a un) et celle du fond B (moyenne locale du fond de
  référence) : 0 = couleur du personnage, 1 = couleur du fond, moyenne pondérée par α, seulement là
  où F et B diffèrent visiblement. Les masques binaires n'ont pas de pixel semi-transparent (« – »).

« Plans simples » = les 9 plans calmes, de dialogue, multi-personnages, cheveux et le plan long :
c'est là que la plainte « même dans des scènes simples » se mesure.

## 3. Première trouvaille : le liseré venait de l'export

L'export alpha de la 0.4/0.5 agrandit le masque avec un **filtre guidé** puis nettoie la couleur du
bord. Mesuré sur la sortie de l'Auto 0.5 :

| plans simples | fond contour % | fond intérieur % | mangé % | contam. % |
|---|---|---|---|---|
| alpha brut (sans traitement) | 3,6 | 8,4 | 1,6 | 33,9 |
| **export 0.5** (filtre guidé + nettoyage couleur) | 8,9 | 8,6 | 6,5 | 23,7 |
| export 0.6 (masque tel quel + estimation du premier plan) | 3,6 | 8,4 | 1,6 | 24,6 |

Le filtre guidé **multiplie par 2,5 le fond le long du contour** (par 4 sur les plans « cheveux » :
2,2 → 9,3 %) et mange le trait d'encre (1,6 → 6,5 %) : sur les planches zoomées, chaque mèche est
entourée d'un halo flou où bave la couleur du ciel ou de l'aura. C'est très probablement le « morceau
de fond dans les cheveux » ressenti sur les plans calmes, où le détourage lui-même était bon.

Export 0.6 : le masque est agrandi tel quel (bilinéaire) et seule la **couleur** des pixels
semi-transparents est corrigée par **estimation du premier plan** (« blur fusion », Germer et al.
2021, deux passes de flou, quelques ms par image). Variantes essayées : rayons 41/91/181, inversion
de l'équation de composition, couleur du pixel opaque le plus proche pour les pixels faibles,
nettoyage couleur 0.5 sans filtre guidé (27,8 %) : aucune ne fait mieux de plus d'un point ; la
variante publiée est gardée. Lecture honnête de la contamination : par rapport aux couleurs brutes,
l'estimation retire environ un quart de la couleur de fond (cheveux : 38,9 → 29,7 %) ; l'export 0.5
affiche un chiffre plus bas sur les cheveux (24,6 %) parce que son halo ajoute une large frange de
pixels faiblement opaques repeints à la couleur du personnage, qui diluent la moyenne — c'est
justement la lueur visible.

## 4. Candidats mesurés

Mêmes 22 plans notés, mêmes 107 images clés, tout mesuré **après l'export 0.6**, sauf les deux
lignes « 0.5 » (export 0.5, ce que l'utilisateur obtenait). Colonnes « fond », « mangé », « contam. »
en %, plus bas = mieux ; « – » = masque binaire, sans pixel semi-transparent.

### Tous les plans (22)

| méthode | J | F | échecs | scint. % | fond contour | fond intérieur | mangé | contam. |
|---|---|---|---|---|---|---|---|---|
| ISNet anime (0.4, conseillé par la doc), export 0.5 | 0,751 | 0,669 | 21/107 | 13,0 | 21,8 | 24,9 | 22,4 | 40,8 |
| BiRefNet seul (réglage 0.4), export 0.5 | 0,847 | 0,794 | 8/107 | 4,6 | 19,7 | 23,5 | 10,2 | 29,7 |
| **Auto 0.5**, export 0.5 | 0,853 | 0,793 | 7/107 | 4,2 | 18,6 | 24,8 | 10,6 | 28,4 |
| Auto 0.5 + export 0.6 | 0,853 | 0,793 | 7/107 | 4,2 | 14,3 | 24,6 | 6,3 | 30,5 |
| BiRefNet HR (entrée 2048) | 0,848 | 0,784 | 8/107 | 4,6 | 15,5 | 23,1 | 5,6 | 34,7 |
| SAM 3 image seul (« anime character ») | 0,894 | 0,827 | 9/107 | 4,0 | 5,6 | 5,7 | 5,2 | – |
| SAM 3 vidéo seul (concept + son propre suivi) | 0,879 | 0,775 | 10/107 | 3,0 | 3,3 | 3,1 | 14,5 | – |
| SAM 3 vidéo → BiRefNet (filtre large, sans SAM 2) | 0,892 | 0,794 | 2/107 | 3,5 | 13,6 | 19,7 | 8,7 | 32,9 |
| deepghs (personnes anime) → BiRefNet dans les boîtes | 0,814 | 0,765 | 6/107 | 6,0 | 14,0 | 22,0 | 12,5 | 29,4 |
| deepghs → BiRefNet → suivi SAM 2 | 0,813 | 0,753 | 6/107 | 6,4 | 13,1 | 21,3 | 13,5 | 30,0 |
| deepghs → SAM 2.1 large dans les boîtes | 0,807 | 0,747 | 12/107 | 9,7 | 6,5 | 6,9 | 16,4 | – |
| Auto 0.5 + ViTMatte (trimap ±10 px) | 0,855 | 0,805 | 7/107 | 4,1 | 15,4 | 25,8 | 6,4 | 27,6 |
| Auto 0.5 + résolution couleur de la bande | 0,844 | 0,798 | 7/107 | 4,6 | 13,1 | 23,8 | 16,9 | 19,0 |
| SAM 3 → BiRefNet (filtre large) → suivi SAM 2 | 0,908 | 0,818 | 1/107 | 3,2 | 12,8 | 19,9 | 7,1 | 32,9 |
| idem + contour serré sur SAM 3 à 0,4 % | 0,915 | 0,857 | 2/107 | 2,6 | 6,6 | 5,9 | 7,7 | 28,0 |
| **Auto 0.6** = idem + contour serré à 0,8 % | **0,920** | **0,862** | **1/107** | **2,8** | 10,6 | **7,8** | 7,6 | 30,9 |
| Manuel 0.5 (clic), export 0.5 | 0,828 | 0,774 | 10/107 | 3,8 | 16,8 | 20,9 | 14,2 | 28,7 |
| **Manuel 0.6** (clic → SAM 3 → suivi → contour serré) | **0,879** | **0,823** | **5/107** | 2,9 | 10,0 | 7,8 | 10,8 | 31,2 |

### Plans simples (9 : calmes, dialogue, multi-personnages, cheveux, plan long)

| méthode | J | F | échecs | fond contour | fond intérieur | mangé | contam. |
|---|---|---|---|---|---|---|---|
| ISNet anime, export 0.5 | 0,941 | 0,869 | 0/40 | 15,3 | 15,9 | 9,2 | 28,0 |
| BiRefNet seul, export 0.5 | 0,970 | 0,898 | 0/40 | 11,0 | 10,7 | 6,8 | 25,9 |
| **Auto 0.5**, export 0.5 | 0,968 | 0,895 | 0/40 | 9,2 | 10,1 | 7,8 | 23,7 |
| Auto 0.5 + export 0.6 | 0,968 | 0,895 | 0/40 | 4,0 | 9,9 | 3,0 | 24,5 |
| BiRefNet HR (2048) | 0,972 | 0,900 | 0/40 | 4,9 | 6,5 | 2,4 | 25,6 |
| SAM 3 image seul | 0,974 | 0,928 | 0/40 | 0,9 | 2,5 | 4,8 | – |
| SAM 3 vidéo seul | 0,957 | 0,855 | 0/40 | 0,8 | 2,7 | 8,9 | – |
| deepghs → BiRefNet → suivi | 0,919 | 0,853 | 0/40 | 4,0 | 9,9 | 10,0 | 25,7 |
| Auto 0.5 + ViTMatte | 0,970 | 0,896 | 0/40 | 4,3 | 10,4 | 3,1 | 24,2 |
| Auto 0.5 + résolution couleur | 0,957 | 0,898 | 0/40 | 3,1 | 8,4 | 13,7 | 10,5 |
| contour serré 0,4 % | 0,974 | 0,919 | 0/40 | 1,8 | 2,5 | 3,7 | 23,4 |
| **Auto 0.6** (contour serré 0,8 %) | **0,974** | 0,911 | 0/40 | **2,7** | **3,0** | 3,4 | 23,9 |
| Manuel 0.5, export 0.5 | 0,970 | 0,900 | 0/40 | 9,2 | 10,0 | 7,3 | 23,9 |
| **Manuel 0.6** | 0,974 | 0,909 | 0/40 | 2,7 | 3,0 | 3,4 | 23,9 |

### Auto 0.5 → Auto 0.6 par catégorie

| catégorie | J | échecs | scint. % | fond contour | fond intérieur | mangé |
|---|---|---|---|---|---|---|
| calme | 0,987 → 0,989 | 0 → 0 | 0,1 → 0,1 | 5,7 → 1,3 | 3,5 → 0,2 | 4,2 → 0,1 |
| multi-personnages | 0,948 → 0,963 | 0 → 0 | 0,3 → 0,3 | 10,7 → 2,7 | 15,3 → 1,0 | 8,7 → 4,4 |
| cheveux | 0,983 → 0,983 | 0 → 0 | 0,8 → 0,8 | 9,3 → 2,0 | 3,7 → 0,2 | 5,6 → 0,3 |
| action floue | 0,876 → 0,941 | 0 → 0 | 5,5 → 3,6 | 23,3 → 17,1 | 45,4 → 12,8 | 7,8 → 2,7 |
| effets | 0,732 → 0,795 | 0 → 0 | 14,0 → 7,1 | 20,6 → 9,5 | 28,7 → 4,0 | 26,1 → 28,3 |
| occlusion | 0,872 → 0,977 | 1 → 0 | 3,0 → 2,8 | 21,3 → 11,8 | 24,1 → 10,2 | 4,6 → 0,8 |
| sombre | 0,787 → 0,843 | 1 → 1 | 4,1 → 4,1 | 22,1 → 12,3 | 30,5 → 14,4 | 13,1 → 13,3 |
| fond chargé (tracteur) | 0,337 → 0,872 | 5 → 0 | 0,0 → 0,1 | 48,1 → 37,3 | 11,4 → 11,2 | 4,1 → 0,1 |
| plan long 11 s | 0,962 → 0,963 | 0 → 0 | 0,9 → 0,8 | 11,3 → 6,8 | 20,9 → 20,0 | 17,2 → 12,8 |

### Lecture

- **Le détecteur était bien le plafond.** SAM 3 avec le seul concept texte « anime character » sait
  ce qu'est un personnage : il ignore le tracteur (J 0,34 → 0,94 seul), le pilier et les fruits, le
  décor sombre de *Black Clover*, et surtout le fond enfermé entre les bras et les jambes (fond
  intérieur 24,6 → 5,7 %). « person » donne le même masque que « anime character » sur les essais
  (le second est gardé, plus explicite). Mais **seul, il ne suffit pas** : bords binaires en
  escalier qui retirent le trait d'encre et rognent les pointes de mèches (planches), 9 échecs
  (il perd Beerus dans le couloir sombre et Goku dans l'aura sur fond noir), et ses bons chiffres de
  contour viennent en partie de la référence, de la même famille SAM. Son mode vidéo est plus stable
  (scintillement 3,0 %) mais mange davantage (14,5 %), demande 5,5 Go de VRAM et 10 Go de RAM.
- **Le bon montage est hybride** : BiRefNet dessine le contour (cheveux, trait d'encre, alpha doux),
  SAM 3 dit *où est le personnage*. Avant le suivi, le masque BiRefNet n'est gardé qu'à moins de
  2,5 % de la diagonale de ce que SAM 3 appelle personnage (l'aura, le tracteur, le pilier tombent) ;
  après le suivi, il est serré à 0,8 % du contour de SAM 3 (le fond entre les bras, sous le menton,
  entre les pointes tombe). Une image où SAM 3 ne voit personne garde le masque BiRefNet : c'est ce
  qui évite ses échecs dans le noir. Rayon 0,4 % : chiffres un peu meilleurs (la référence est
  SAM), mais il rogne les pointes de mèches que SAM 3 sous-segmente (planche s06) ; 0,8 % les garde.
  Attendre que SAM 3 couvre tout le personnage avant de serrer mesure moins bien (J 0,901).
- **Détecteur anime deepghs** (YOLOv8, MIT) : il cadre bien les personnages, mais une boîte
  contient autant de décor qu'un masque BiRefNet et coupe les membres tendus (mangé 12,5 %) ; il ne
  sépare pas un personnage d'un objet collé à lui. SAM 2.1 large dans ses boîtes (image par image) :
  12 échecs, scintillement 9,7 %. Rejeté.
- **SAM 3 vidéo comme sélecteur devant BiRefNet** (sans SAM 2) : J 0,892, 2 échecs, mais le fond
  enfermé reste à 19,7 % et il coûte 5,5 Go de VRAM ; le mode image image par image, suivi par
  SAM 2, fait mieux pour moins cher.
- **BiRefNet HR (2048)** : à peine différent sur des images de travail de 1080 px (J 0,848), 8,1 Go
  de VRAM, 3 à 4 fois plus lent. Rejeté. (La variante « HR-matting » a été téléchargée mais pas
  mesurée, faute de temps.)
- **Affinage des bords** : ViTMatte sur une bande de ±10 px ne baisse pas le fond du contour
  (15,4 % contre 14,3 %) et triple le temps ; la résolution de l'alpha par la couleur locale
  (équation de composition) donne la meilleure contamination (19 %) mais **mange le trait d'encre**
  (mangé 16,9 %, un trait noir n'est ni le ciel ni les cheveux). La suppression des pixels « couleur
  du fond local » dans la bande ne change rien de mesurable (3,98 % contre 4,04 % sur les plans
  simples) et mange un peu : rejetée. MatAnyone 2 sur la seule bande de contour n'a pas été retesté,
  faute de temps. Le vrai gain sur le contour venait de **l'export** (§3) et de **SAM 3** (fond
  intérieur).
- **Ce qui reste** : la contamination couleur ne baisse pas avec l'Auto 0.6 (30,9 %) — elle mesure
  les pixels semi-transparents restants, que l'estimation du premier plan ne corrige qu'en partie ;
  à l'œil, sur damier et sur couleur vive, plus de liseré coloré visible sur les plans calmes. Le
  « mangé » monte sur les effets (28 %) : SAM 3 ne compte pas les **objets tenus** comme partie du
  personnage (bâton et auréole de Whis, bol de Beerus : coupés), et coupe les mains prises dans une
  aura.

## 5. Vitesse et mémoire (RTX 3090, passe au calme, plan de 84 images à 1080 × 608)

| étape | ms / image | pic VRAM (PyTorch) |
|---|---|---|
| BiRefNet (détection, contour) | 119 | 1,6 Go |
| SAM 3 image, concept « anime character » | 212 | 2,0 Go |
| suivi SAM 2.1 tiny (+ ré-amorçages) | 101 à 106 | 1,7 Go |
| contour serré | 4 | – |
| **Auto 0.5** (BiRefNet + suivi) | **225** | 1,7 Go |
| **Auto 0.6** (BiRefNet + SAM 3 + suivi + contour serré) | **435** (× 1,9) | 2,0 Go |
| export : estimation du premier plan, image 1080p, CPU | 206 | – |

Les modèles sont chargés l'un après l'autre (BiRefNet, puis SAM 3, puis SAM 2), le pic de VRAM est
celui du plus gros : ≈ 2,5 Go avec le contexte CUDA. Le banc complet (tableaux du §4) tournait avec
3 à 5 calculs en parallèle sur la même carte ; ses temps (Auto 0.6 : 2,8 s/image) ne servent qu'à
comparer.

## 6. Verdict visuel par plan

Planches de 3 images clés par plan, alpha exporté sur damier (avant = Auto 0.5 avec export 0.5,
après = Auto 0.6), et zooms ×3 sur la zone de contour la plus découpée (cheveux en général), alpha
composé sur magenta pour que tout reste de fond saute aux yeux. U = utilisable tel quel,
R = retouche légère, X = inutilisable.

| plan | contenu | Auto 0.5 | **Auto 0.6** | ce qui change |
|---|---|---|---|---|
| s01 | calme, ciel | U | U | plus de halo violet le long des mèches |
| s02 | calme, gros plan | U | U | idem |
| s04 | 4 personnages | R | U | fond entre les personnages retiré (fond intérieur 44,9 → 2,8 %) |
| s05 | Whis et Beerus | U | **R** | **régression** : bâton et auréole de Whis coupés (objets tenus) |
| b5 | 2 personnages | U | U | — |
| s06 | cheveux SSJ3 | U | U | halo jaune de l'aura autour des pointes disparu, pointes gardées |
| b1 | cheveux longs sur ciel | U | U | halo bleu flou autour des cheveux disparu |
| s09 | action, éclairs | R | R | moins d'éclairs gardés |
| s10 | action, deux corps | R | R | — |
| s11 | action, rayon | R | R | rayon d'énergie encore partiellement gardé |
| s12 | action, flou rouge | U | U | — |
| b6 | gros plan sombre | X | **R** | le bloc de décor sombre n'est plus pris pour le personnage |
| s13 | aura + débris | R | **U** | aura et sol entre les jambes retirés |
| s15 | aura sur fond noir | U | U | — |
| b2 | forme démon, particules | R | R | un amas de particules reste sur une image |
| s16 | sort de la fumée | U | U | — |
| s17 | derrière un mur | U | U | — |
| s18 | nuit, couloir | U | U | — |
| s19 | palais sombre | R | R | pilier et fruits retirés, mais le bol tenu par Beerus est coupé |
| b4 | pluie de nuit | X | X | personnage secondaire toujours perdu en fin de plan |
| s20 | collé au tracteur | X | **R** | tracteur retiré ; Goten, assis derrière le volant, n'est visible que par la tête (juste) |
| L1 | plan long 11 s | U | U | — |
| **total** | | 12 U / 7 R / 3 X | **13 U / 8 R / 1 X** | |

## 7. Intégration (ce qui a changé dans l'application)

- **Auto par défaut, un clic, rien à régler** : BiRefNet → **SAM 3** (concept « anime
  character ») → suivi SAM 2.1 → **contour serré sur SAM 3** → export décontaminé. Le *Manuel*
  (clic) passe par le même chemin (le clic choisit le personnage, SAM 3 nettoie) ; la boîte et le
  pinceau ne changent pas.
- **Paliers** : SAM 3 sur NVIDIA 8, 12, 16 et 24 Go (pas sur le palier « Rapide », où il double le
  temps) ; **pas sur Mac** (non mesuré, Mac injoignable) ni sur CPU. Tableau des paliers dans
  Settings > Models avec la nouvelle ligne « Détourage (sélection des personnages) » et sa licence.
- **Repli propre** : poids SAM 3 inaccessibles (dépôt à accès contrôlé, pas de jeton approuvé),
  erreur ou manque de mémoire pendant la passe SAM 3 → l'Auto tourne exactement comme en 0.5 (message
  dans le journal, jamais d'échec du travail) ; interrupteur « Sélection des personnages par SAM 3 »
  dans Settings > Models.
- **Export alpha** (ProRes 4444, VP9, PNG) : masque exporté tel quel + estimation du premier plan ;
  interrupteur « Couleurs de bord propres à l'export » (désactivé = export 0.5).
- **Licence SAM 3** (SAM License, Meta) : pas une licence « non commerciale » (l'usage commercial est
  permis), interdit les usages militaires, nucléaires, ITAR et l'ingénierie inverse ; les poids ne
  sont jamais redistribués par AMV Tools, ils sont téléchargés par Hugging Face avec le jeton de
  l'utilisateur après acceptation de la licence sur la page du modèle (3,4 Go). Le détecteur deepghs
  (MIT) et ViTMatte (Apache-2.0, entraîné sur Composition-1k) n'ont servi qu'au banc.

## 8. Mask Mode Auto activable par défaut ?

**Oui, sur les cartes NVIDIA de 8 Go et plus où SAM 3 est accessible.** Sur 22 plans réels, l'Auto
0.6 ne perd plus le personnage qu'une fois sur 107 images clés (7 en 0.5), 21 plans sur 22 sont
exploitables (13 tels quels), et le défaut ressenti en premier — le fond dans les cheveux et autour
des personnages sur des scènes simples — est divisé par trois à quatre (fond le long du contour 9,2 →
2,7 %, fond enfermé 10,1 → 3,0 % sur les plans simples), sans manger les cheveux (3,4 % contre
7,8 %). Restent à connaître : les objets tenus coupés (bâton, bol), les effets lumineux en partie
gardés, un plan de nuit sous la pluie, et un traitement deux fois plus long qu'en 0.5.

**Non ailleurs** : sans SAM 3 (Mac, CPU, palier Rapide, jeton absent), on retombe sur l'Auto 0.5
(seul l'export s'améliore), dont le verdict de la partie 1 reste valable ; et rien n'a été mesuré sur
Mac. Comme l'accès à SAM 3 dépend du jeton Hugging Face personnel de chaque utilisateur, le réglage
`enable_mask` reste désactivé par défaut dans le code : le basculer pour tous mettrait l'Auto 0.5
devant ceux qui n'ont pas SAM 3. L'étape suivante naturelle : l'activer automatiquement quand le
palier NVIDIA a SAM 3 et que le jeton ouvre ses poids.

## 9. Planches

Sur la machine de mesure, dans le dossier des rapports `reports/roto/roto2/` :

- `cheveux-avant-apres/<plan>.jpg` : zooms ×3 sur la zone de contour la plus découpée de 3 images
  clés — image, référence (contour cyan), Auto 0.5 (export 0.5), Auto 0.6 — sur magenta ;
- `plans-avant-apres/<plan>.jpg` : images entières, référence / Auto 0.5 / Auto 0.6, sur damier et
  sur vert ;
- `verdict/verdict_*.jpg` : planches compactes utilisées pour le verdict du §6.

## Résumé (partie 2)

- **Le liseré venait de l'export** : le filtre guidé de l'export 0.4/0.5 ajoutait un anneau de fond
  autour des cheveux (×2,5 à ×4). Export 0.6 : masque tel quel + estimation de la couleur du premier
  plan (ProRes 4444, VP9, PNG).
- **SAM 3 est le meilleur « détecteur de personnage » mesuré**, mais pas un bon détoureur seul (bords
  en escalier, échecs dans le noir). **Gagnant : BiRefNet pour le contour + SAM 3 pour savoir où est
  le personnage + suivi SAM 2 + contour serré à 0,8 %.** Auto : J 0,853 → 0,920, échecs 7 → 1 sur
  107, scintillement 4,2 → 2,8 %, fond le long du contour 18,6 → 10,6 %, fond enfermé 24,8 → 7,8 %
  (plans simples : 9,2 → 2,7 % et 10,1 → 3,0 %). Manuel : J 0,828 → 0,879, échecs 10 → 5.
- Rejetés à la mesure : détecteur anime deepghs, BiRefNet HR, ViTMatte, résolution couleur de
  l'alpha, SAM 3 seul (image ou vidéo).
- **Mask Mode Auto activable par défaut : oui sur NVIDIA ≥ 8 Go avec SAM 3, non ailleurs.**



---

# Partie 1 — chantier roto (0.5)


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
trois processus GPU et une autre session) : ils sont gonflés de façon variable et ne valent que
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

- Avant (référence, BiRefNet, ISNet anime, SAM 2.1 tiny au clic) : `reports/roto/avant/<plan>.jpg`
- Après (référence, BiRefNet 0.4, Auto 0.5, Manuel 0.5) : `reports/roto/apres/<plan>.jpg`
- Vérité terrain (planches de contrôle image + superposition) : `roto/review/<plan>.jpg`

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
