# Chantier roto « objets tenus » : Mask Mode Auto qualité max

Date : 26 octobre 2026 (nuit). Machine de mesure : RTX 3090 24 Go (Linux, 6 vCPU), partagée pendant
toute la mesure avec une autre session de banc (temps gonflés, voir §6). Rien n'a été mesuré sur Mac.

Point de départ : le mode Auto 0.6 en **qualité max** (BiRefNet → SAM 3 « anime character » avec passe
zoomée → suivi SAM 2.1 → contour serré sur SAM 3 avec pointes rendues → fragments supprimés →
stabilisation → export « max »), décrit dans `chantier-roto.md`. Les deux défauts restants visés : les
**objets tenus coupés** et les **plans d'action / flou de mouvement**.

# Objets tenus et action

## 1. Pourquoi l'option « objet tenu » de la session principale était restée désactivée

SAM 3 ne compte pas ce qu'un personnage tient ou porte comme faisant partie de lui (bâton et auréole de
Whis, bol de Beerus). La session principale avait essayé de les rendre dans le contour serré
(`hug_with_tips(rgb=…)`) : une poche opaque *enfermée* par le personnage revenait si ses couleurs ne
ressemblaient pas au fond autour. Mesuré, ce critère de couleur rendait aussi des poches de fond (fond
intérieur 5,7 → 7,2 %) et ne traitait pas les objets qui *dépassent* du personnage (un bâton) : ils
sont coupés dès la porte large (2,5 % de la diagonale) avant le suivi. Il faut donc un critère qui dise
*ce qu'est* l'objet, pas seulement qu'il n'a pas la couleur du fond.

## 2. Banc : 5 plans ajoutés, références des objets, métrique « objet tenu conservé »

**Plans ajoutés** (mêmes épisodes que le banc, 3,5 s chacun, 84 images, catégorie « objet ») :

| plan | contenu | objet |
|---|---|---|
| h1 | main gantée qui tient un marteau (gros plan, le marteau est plus grand que la main) | marteau |
| h2 | Whis et Beerus en pied, ciel rouge | bâton (fin, toute la hauteur), auréole |
| h3 | Whis et Beerus en vol | bâton en diagonale, auréole |
| h5 | gros plan de Whis | grande auréole qui encadre le visage |
| h7 | Beerus de profil, Whis derrière lui tient son bâton | bâton tenu par une main, coupé par le cadre |

Avec s05 (bâton et auréole de Whis) et s19 (bol de Beerus), cela fait **7 plans à objets tenus**
(35 images clés notées) ; le banc passe à **27 plans notés, 132 images clés**.

**Références** (6 images clés par plan, toutes regardées sur planche, deux reprises : bas du bâton de
h3, cape de h5 comptée à tort comme objet) :

- personnages des nouveaux plans : SAM 2.1 large amorcé par des points placés à la main par personnage
  et par image (h1, h2, h5, h7) ou la référence automatique du banc vérifiée (h3) ;
- objets : zone tracée à la main (bande le long du bâton, boîte autour de l'auréole ou du marteau) ∩
  masque BiRefNet > 0,5, moins le personnage ; le bol de s19 par SAM 2.1 large sur une boîte ;
- s05 et s19 : leurs références d'origine **n'incluaient pas** le bâton, l'auréole et le bol (SAM 2.1
  avait suivi le corps) alors que la règle du banc dit « les objets tenus font partie du personnage » :
  elles sont complétées par ces objets. Les chiffres de ces deux plans changent donc pour toutes les
  méthodes, référence comprise (J de l'Auto qualité max sur les 22 plans d'origine : 0,924 avant
  correction, 0,921 après).

Biais connu : le contour des objets vient de BiRefNet, le même modèle qui fournit le bord du masque
final ; la métrique « objet tenu » mesure donc surtout *si l'objet est gardé*, pas la finesse de son
bord. La cuillère de s19 (2 à 4 px de large) n'est pas dans la référence objet.

**Nouvelle métrique** : *objet tenu conservé* = alpha exporté moyen sur la référence objet (ses 2 px
extérieurs non notés), en %, sur les images clés qui en ont une. Toutes les autres métriques sont
celles du banc (J, F, échecs, scintillement, fond contour, fond intérieur, mangé, contamination),
toutes sur l'alpha exporté en mode « max », après la stabilisation de l'application.

## 3. Méthode retenue : SAM 3 nomme les objets, BiRefNet les dessine

Sur la même passe SAM 3 que le concept « anime character » (l'encodeur d'image est partagé, seul le
décodeur tourne une fois de plus par mot), 7 concepts d'objets : **weapon, sword, staff, halo, bowl,
hammer, cup**. « held object » (essayé) ne renvoie rien ; « cape » renvoie les vêtements, déjà dans le
personnage. Une instance d'objet (score ≥ 0,4) est rendue au masque si :

1. elle **touche** ce que SAM 3 appelle personnage (à 0,6 % de la diagonale près) ;
2. **BiRefNet la voit au premier plan** (alpha moyen ≥ 0,6) : un pilier ou un arbre étiqueté par
   erreur reste du fond pour lui ;
3. elle n'est **pas plus grande que 2 fois le personnage** ;
4. elle **ne brille pas** : si plus de 30 % de ses pixels sont presque blancs (luminance > 0,85) ou
   vifs (canal max > 0,85 et saturation > 0,5), c'est un effet lumineux, pas un accessoire ;
5. elle est **persistante** : un morceau n'est gardé que s'il recouvre (≥ 30 %) l'objet retenu de
   l'image précédente ou suivante.

L'alpha rendu est celui de **BiRefNet brut** (avant la porte SAM 3) dans l'objet élargi de 2 px, ajouté
après le contour serré ; dans l'application il n'est gardé que s'il touche le personnage suivi (en mode
« personnage principal », l'accessoire d'un autre personnage reste dehors).

Le filtre « lueur » vient de la mesure : sans lui, le rayon d'énergie qui traverse Vegeta (s12) était
pris pour une « weapon » (fond intérieur de s12 0 → 46 %), et une flamme de ki bleue collée à la jambe de
Goku (s13) apparaissait sur quelques images (visible sur planche, scintillement 10,7 → 11,2 %). Part de
pixels lumineux mesurée sur toutes les instances retenues du banc : effets 49 à 97 %, vrais objets 18 %
au plus (le pommeau doré d'un bâton) ; seuil à 30 %.

## 4. Résultats

Avant = Auto qualité max actuel (`stab25~hug…` du banc : contour serré, pointes, fragments,
stabilisation, export « max ») ; après = la même chaîne + objets tenus. Mêmes images clés, mêmes
références (s05/s19 complétées), alpha exporté.

| groupe | J | F | échecs | scint. % | fond contour % | fond intérieur % | mangé % | contam. % | objet tenu % |
|---|---|---|---|---|---|---|---|---|---|
| **tous (27)** avant | 0,902 | 0,838 | 3/132 | 2,59 | 5,84 | 4,26 | 11,67 | 19,76 | 22,7 |
| **tous (27)** après | **0,926** | **0,869** | **2/132** | **2,48** | 5,97 | 4,27 | **7,34** | **19,21** | **95,2** |
| 22 d'origine avant | 0,921 | 0,860 | 1/107 | 2,53 | 6,07 | 5,21 | 8,77 | 19,05 | 2,5 |
| 22 d'origine après | **0,928** | **0,870** | 1/107 | 2,52 | 6,09 | 5,20 | **7,42** | 19,32 | **93,4** |
| objets tenus (7) avant | 0,839 | 0,757 | 2/35 | 2,32 | 4,58 | 1,02 | 21,61 | 26,98 | 22,7 |
| objets tenus (7) après | **0,928** | **0,875** | **1/35** | **1,90** | 5,00 | 1,05 | **5,47** | **24,88** | **95,2** |
| action (5) avant = après | 0,946 | 0,878 | 0/23 | 3,61 | 11,34 | 7,19 | 3,28 | 22,79 | – |
| simples (9) avant | 0,968 | 0,907 | 0/40 | 0,45 | 1,60 | 2,56 | 6,37 | 16,18 | 4,4 |
| simples (9) après | **0,981** | **0,932** | 0/40 | 0,43 | 1,63 | 2,56 | **2,70** | 16,94 | **100** |

Par plan à objet (objet tenu conservé, avant → après) : s05 4 → 100 % (bâton et auréole), s19 1 → 87 %
(bol ; BiRefNet laisse un trou dans la gelée rose, voir planche), h1 38 → 84 % (marteau ; une image clé
reste un échec : main et marteau pris dans un flou de bougé), h2 15 → 99 %, h3 27 → 99 %, h5 74 → 99 %
(auréole), h7 0 → 98 % (bâton). Sur les plans **sans objet annoté**, 697 images sur 2 058 changent au
total, presque toutes dans les plans à objets ; ailleurs, à l'œil : le katana de Yami à la hanche (b5,
J 0,96 → 0,98, il était coupé), la queue de Beerus dans le couloir sombre (s18) et dans le combat
(s10, 4 images), la cigarette de b1, deux images de b2 (cheveux du personnage du fond, inchangés à
l'œil). Aucun fond ajouté visible.

Ce qui monte un peu, et pourquoi :

- **fond contour** +0,13 pt sur les 27 plans, entièrement sur les plans à objets (h7 2,9 → 5,5 %) : le
  bord des bâtons vient de l'alpha doux de BiRefNet, et la référence objet (BiRefNet > 0,5) compte sa
  frange 0,2-0,5 comme du fond. À l'œil (export zoomé ×3 sur magenta), pas de liseré de ciel ;
- **contamination** +0,3 pt sur les 22 d'origine, tout sur s05 (14,2 → 20,4 %) : les pixels
  semi-transparents du bâton et de l'auréole, fins et bordés d'encre, gardent un peu de violet de ciel
  après l'export ; sur l'ensemble des plans à objets elle baisse (27,0 → 24,9 %). Sur la planche
  zoomée, le trait d'encre de l'auréole paraît par endroits un peu pointillé.

Variantes mesurées et écartées (même banc, avant stabilisation) :

| variante | effet | décision |
|---|---|---|
| sans filtre « lueur » | rayon d'énergie de s12 rendu (fond intérieur de l'action 7,2 → 16,5 %), flamme de ki sur s13 (visible) | filtre ajouté |
| filtre « lueur » blanc seul (20 %) | s12 corrigé, flamme bleue de s13 encore rendue sur 4 images (scintillement 10,7 → 11,2 %) | élargi aux couleurs vives, 30 % |
| sans persistance temporelle | identique sur ce banc (les objets y sont détectés de façon stable) | gardée : protège d'une détection isolée |
| intérieur de l'objet SAM 3 rendu opaque (trou du bol) | identique à 0,01 près | écarté |
| + rendre les personnages que SAM 3 voit mais que le suivi a perdus (b4) | J des 22 plans 0,928 → 0,917, un échec de plus | écarté |

## 5. Action rapide et flou de mouvement

Ce que montrent les planches des plans d'action (s09, s10, s11, s12, b6) et des derniers échecs : les
défauts restants n'y sont **pas des pertes sur le flou**. Sur ces 5 plans, l'Auto qualité max n'a aucun
échec (0/23) et ne mange presque rien (3,3 %) ; le fond qui reste (fond contour 11 %, intérieur 7 %)
vient des **effets** gardés (éclairs de s09, rayon de s11, bloc sombre à droite du visage de b6 où les
cheveux noirs se fondent dans le fond noir). L'échec restant sur les 22 plans d'origine est b4 (personnage
secondaire de nuit sous la pluie : SAM 3 le voit, mais le masque suivi ne le contient pas sur la
plupart des images — très probablement écarté au choix des cibles, trop petit à côté du gros plan) ;
b2 n'est pas un échec mais reste à J 0,59 (ailes de la forme démoniaque que SAM 3 ne compte pas comme
personnage).

Mesuré :

- **Comblement temporel du masque SAM 3 sur les images floues** : sur une image où SAM 3 voit moins de
  80 % de ce qu'il voit sur une voisine (membre étalé par le flou), on ajoute ce que les deux voisines,
  recalées par flot optique, voient toutes les deux (jamais à travers une coupe). Il ne touche que 35
  images sur 2 058 (1,7 % ; s15 9, s09 5, s13 3, s16 3…) : **SAM 3 perd rarement le personnage sur le
  flou de ce banc**, et les scores sont identiques (J 0,902, échecs 3/132, mangé 11,67 %), scintillement
  2,59 → 2,54 %. Écarté (gain non significatif).
- **Vote temporel du sélecteur** (image t gardée si t-1 et t+1 d'accord) et **union SAM 3 image + SAM 3
  vidéo** : déjà mesurés par la session principale sur le même banc, J 0,923 / 0,922 contre 0,922, rien de
  significatif ; non repris.
- **Récupérer les personnages que SAM 3 voit et que le suivi a perdus** (visait b4) : le personnage
  secondaire revient sur certaines images seulement, et ailleurs des morceaux d'autres silhouettes
  reviennent aussi : J des 22 plans 0,928 → 0,917, un échec de plus. Écarté.

Conclusion pour l'action : rien de mesuré ne gagne ; l'effort utile est ailleurs (effets lumineux
gardés, choix des cibles en « tous les personnages »). Le filtre « lueur » des objets tenus, lui, garantit
que la nouvelle option ne rajoute pas d'effets sur ces plans (action identique avant/après).

## 6. Vitesse et mémoire

Passe SAM 3 de la qualité max, 40 images de s05 à 1080 × 608, RTX 3090 partagée (une autre session
occupait 6 Go de la carte, charge CPU ≈ 5) :

| passe SAM 3 | ms / image | pic VRAM (PyTorch) |
|---|---|---|
| « anime character » + passe zoomée (avant) | 973 | 2,00 Go |
| + 7 concepts d'objets sur le même encodeur (après) | 1 402 (+ 44 %) | 2,11 Go |

Le masque personnage est identique avec ou sans les objets (IoU 1,0 sur 5 images contrôlées). Le reste
de la chaîne (BiRefNet, suivi, contour serré, stabilisation, export) ne change pas ; le calcul des objets
retenus et leur ajout coûtent quelques ms par image sur CPU. Une première mesure faite pendant que
d'autres calculs chargeaient la machine (charge ≈ 22) donnait 1 166 → 2 160 ms : ces temps ne valent que
pour comparer.

## 7. Intégration

- **Option** `mask_keep_held` (Settings > Models > *Garder les objets tenus (qualité max)*, FR/EN),
  valeur « auto » = **activée** : elle gagne nettement sur les objets et ne change rien ailleurs. Elle
  ne s'applique qu'en qualité max (SAM 3 présent) ; désactivée, ou sans SAM 3, la chaîne est exactement
  celle d'avant.
- `Sam3Concept.segment_with_objects` : un seul passage de l'encodeur d'image pour le concept
  personnage et les concepts d'objets ; la passe zoomée réutilise le masque plein cadre.
- `auto_roto.held_regions` / `persist_held` / `keep_held` ; `apply_selector(held=…)` calcule l'alpha
  des objets sur le masque BiRefNet brut ; `tighten(held=…)` l'ajoute après le contour serré, seulement
  s'il touche le personnage suivi. Même chemin pour le travail *Cutout* de la boîte de finition.
- Vérifié sur la carte avec le vrai SAM 3, par le chemin de l'application (pas celui du banc) : s05,
  objet tenu 9,8 → 97,4 % ; s12 (rayon), rien d'ajouté.
- Banc : `roto_eval.py` méthodes `held…` (prompts, seuils, persistance, filtre lueur) et `sfill…`
  (comblement temporel), métrique « objet tenu » (`<root>/gt_held`), colonne dans le tableau.
- Tests : `tests/test_roto.py` (objets retenus/écartés, lueur, persistance, chemin selector → tighten,
  mode personnage principal, réglage limité à la qualité max) ; `uv run pytest -m "not slow"` 381 passés ;
  `tsc`, `vitest` (28) et `npm run build` OK.

## 8. Planches

Sur la machine de mesure, dans le dossier de sortie du chantier (`b/`) : `held_<plan>.jpg` (image /
référence / Auto qualité max / avec objets tenus, sur magenta), `z_<plan>.jpg` (export « max » zoomé ×3
autour de l'objet), `l20_<plan>.jpg` / `fin_<plan>.jpg` (plans sans objet annoté où quelque chose a été rendu), `gt_<plan>.jpg`
(références : personnage en vert, objet en magenta).

## Résumé

- **Objets tenus** : SAM 3 nomme les accessoires (weapon, sword, staff, halo, bowl, hammer, cup) sur sa
  passe existante, BiRefNet les dessine ; gardés s'ils touchent un personnage, que BiRefNet les voit, qu'ils
  ne brillent pas et qu'ils persistent. Sur les 7 plans à objets : **objet tenu conservé 22,7 → 95,2 %,
  J 0,839 → 0,928, mangé 21,6 → 5,5 %, échecs 2 → 1**. Sur les 27 plans : J 0,902 → 0,926, F 0,838 →
  0,869, échecs 3 → 2, mangé 11,7 → 7,3 %, scintillement 2,59 → 2,48 % ; fond contour +0,13 pt et
  contamination de s05 en hausse (bords des bâtons), sans liseré visible. **Activé par défaut en qualité
  max**, désactivable.
- **Action / flou** : rien de mesuré ne gagne. SAM 3 ne perd le personnage que sur 1,7 % des images de ce
  banc ; le comblement temporel par flot optique, le vote et l'union image + vidéo ne changent pas les
  scores ; récupérer les personnages perdus par le suivi dégrade. Les défauts restants des plans d'action
  sont les effets lumineux gardés, pas le flou.
- Coût : passe SAM 3 + 44 % (≈ +0,4 s par image sur RTX 3090 partagée), +0,1 Go de VRAM.
- Banc : 27 plans notés (5 nouveaux plans à objets), références de s05/s19 complétées, nouvelle métrique
  « objet tenu conservé ».
