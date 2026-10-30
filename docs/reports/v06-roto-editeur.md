# v06 — Éditeur de détourage : « l'IA fait le premier jet, le monteur finit »

Date : 30 octobre 2026, 2 h – 11 h. Branche `jarvis/v06-roto-editor` (jarvis/v06 + fusion de
`jarvis/v05-roto-stable`). Machine de mesure : RTX 3090 (debian-nas) ; interface pilotée dans un
Chromium à moi sous Xvfb, et l'appli Electron complète sur CPU pour le test de bout en bout.

<!-- RESUME -->

## 1. Ce que fait l'éditeur

Dans le mini-éditeur d'un plan, un nouveau bouton **« Détourer ce plan »** ouvre l'éditeur en plein
écran (le Mask Mode auto reste désactivé par défaut ; l'éditeur est le point d'entrée, il est
visible même quand Mask Mode est coupé).

- **Premier jet IA en un clic** (« Détourer ») : la chaîne Stable v5 de la branche roto (bord doux de
  BiRefNet, SAM 3 choisit les personnages, suivi SAM 2.1, garde temporelle, objets tenus, trait noir).
  Au départ on choisit qui détourer : *tous les personnages*, *le personnage principal*, ou *cliquer
  sur le personnage*. Barre de progression, bouton Annuler, l'interface reste utilisable.
- **Corriger, deux façons** :
  - pinceau : vert = ajouter au personnage, rouge = retirer (Alt inverse), taille avec `[` `]` ; gomme
    (`E`) = remet la passe IA sous le trait ;
  - clics (SAM 2.1) : clic = ajouter la partie cliquée, Alt+clic ou clic droit = la retirer. SAM propose
    trois régions emboîtées : un clic positif prend la plus sûre, un clic négatif la plus petite sûre
    et seulement le morceau sous le curseur (sinon un Alt+clic sur une flamme d'aura emportait Goku
    avec elle — vu et corrigé pendant le chantier) ;
  - annuler / rétablir (Ctrl+Z / Ctrl+Shift+Z), « Valider cette image » (`V`).
  - Après chaque geste, l'image courante est recalculée et **le temps mesuré est affiché** (« Image
    corrigée en 134 ms (IA 95 ms) »). Les gestes s'enchaînent dans une file : rien n'est perdu si on
    va plus vite que la machine.
- **Propager** (← ↔ →, `P`) : le suivi vidéo SAM 2.1 (large sur NVIDIA) repart des images corrigées,
  sa mémoire initialisée sur toutes les images verrouillées de la plage ; il s'arrête à la prochaine
  image verrouillée dans chaque sens et **n'écrit jamais une image corrigée ou validée**. Sur chaque
  image, le résultat garde le bord doux de la passe IA là où il est d'accord avec l'objet suivi (à
  5 px près : ce que l'utilisateur a retiré reste retiré) et prend l'objet suivi là où l'IA n'avait
  rien (ce qu'il a ajouté, un personnage perdu). Sans GPU : flot optique.
- **Images à vérifier** : sous la vidéo, une case par image ; les images corrigées par l'utilisateur
  ont un losange jaune ; les **images suspectes sont en rouge**, calculées avec les métriques du banc
  (perte d'aire > 40 % = personnage perdu, saut d'aire > 25 %, îlots / pixels parasites, morceau de
  l'image précédente disparu = objet perdu, trous, bord en escalier). `N` / `Shift+N` : suivante /
  précédente ; flèches image par image, espace lecture.
- **Bords** (aucun réglage obligatoire, appliqués à l'aperçu en direct, ~70 ms) : *Garder le trait
  noir* (activé), *Adoucir ↔ Durcir*, *Rétrécir ↔ Élargir* (± 6 px), *Lissage dans le temps*.
- **Vues** `1`–`5` : surimpression rouge, personnage sur noir / blanc / damier, original ; zoom à la
  molette, déplacement espace+glisser, ajuster (`0`), plein écran (`F`), aide (`?`).
- **Plusieurs personnages** : un calque par personnage (« + Personnage », clic dessus, Détourer).
- **Sauvegarde automatique** (dossier `roto_projects` des données de l'appli) : on ferme, on rouvre le
  même plan, on reprend où on en était. FR / EN.
- **Export** : ProRes 4444 avec transparence (fond propre, couleurs du bord décontaminées : l'export
  Stable v5), séquence PNG avec alpha, masque seul noir et blanc (ProRes 422 HQ) ; le plan est ajouté à
  la sélection (dérush) et **l'export timeline place le clip détouré** à la place de l'épisode
  (option `use_cutouts`, activée).

Code : `backend/roto/` (projet + sauvegarde, bords, images suspectes, moteur, export, lien timeline),
`backend/api/roto.py` (contrat : `docs/roto-editor-api.md`), `app/src/components/Roto*.tsx`,
`app/src/i18n/strings.roto.ts`. Tests : `tests/test_roto_editor.py` (14), `app/src/components/rotoUtils.test.ts`,
`app/e2e/roto.e2e.mjs`. Mesure : `tests/eval/roto_editor_sim.py`, `tests/eval/roto_editor_examples.py`.

## 2. Correction en direct : temps mesurés (RTX 3090)

Image de travail 1080×608 (le 1080p de l'épisode, réduit comme pour tout le détourage de l'appli),
plan b1 (cigarette), une seule mesure à la fois sur la carte, médiane de 5 gestes
(`python -m tests.eval.roto_editor_sim latency`) :

| geste | temps serveur | cible |
|---|---|---|
| clic (SAM 2.1 large) sur une **nouvelle** image (calcul de l'image + masque) | **112 ms** | < 300 ms |
| clic suivant sur la **même** image (image déjà calculée) | **21 ms** | |
| coup de pinceau | **6 ms** | |
| aperçu de l'image avec les réglages de bord (trait noir compris) | **73 ms** | |
| propagation SAM 2.1 large | **180 ms par image** (48 images en 8,7 s) | |

Dans l'appli, l'éditeur affiche le temps **total** (serveur + transfert + affichage de la nouvelle
image), mesuré dans la démo (§4), carte au calme : clics sur une image déjà calculée **120 à 180 ms**,
coup de pinceau **139 ms**, premier clic après la passe IA 407 ms, premier clic sur une autre image
juste après une propagation 1,2 s (toutes les images ont changé, l'aperçu est recalculé). Le premier
clic chargeait le modèle et préparait la carte (16 à 55 s mesurés en cours de chantier) : le modèle
de clic est maintenant préchargé, avec un calcul à blanc, à la fin de la passe IA et à l'ouverture
d'un plan déjà détouré. Mémoire vidéo de l'éditeur seul (clics + propagation) : 2,4 Go.

Sur CPU (machine sans GPU, test de bout en bout) : la passe IA utilise ISNet anime sans suivi, les
clics la pièce détectée sous le curseur, la propagation le flot optique ; tout tourne en tâche de
fond (l'interface reste utilisable, la progression et le bouton Annuler restent là) mais c'est lent :
voir §4.


## 3. Est-ce que ça fait « 90 % du boulot » ? Le monteur simulé

### Méthode (`tests/eval/roto_editor_sim.py`)

Un « monteur » automatique utilise **les outils de l'éditeur** (le code de `backend/roto`, pas un
raccourci), guidé par la vérité terrain du banc roto (références vérifiées à l'œil, 6 à 13 images par
plan) :

1. **Premier choix**, comme un monteur : *tous les personnages*, sauf si la passe IA « tous » prend
   des personnages hors référence (plus d'un tiers de l'aire de la référence loin d'elle : la foule
   derrière Beerus sur s18) → *cliquer sur le personnage* (un clic au cœur de chaque personnage de la
   référence, vraie passe IA de l'appli). Passe IA « tous » : celle du banc (`runs/app_stable`, la même
   chaîne Stable v5).
2. **Boucle** : il prend la plus grosse erreur de la pire image — une image perdue d'abord (clic sur
   le personnage là où la voisine l'a ; une image perdue à côté d'une référence presque vide, fumée ou
   sortie de champ, est juste et laissée), sinon le plus gros morceau faux sur une image de référence
   (morceau manquant ou en trop, au-delà des 2 px de précision de la référence), un trou que la
   référence n'a pas, ou un bout de trait noir perdu — et le corrige d'**un geste** :
   - un **clic** (positif / négatif) si le morceau est gros et épais (≥ 600 px, ≥ 7 px d'épaisseur) ;
     s'il corrige moins d'un quart du morceau, il l'**annule** (un geste de plus) et peint ;
   - sinon un **coup de pinceau** le long du morceau (son axe), rayon = sa demi-épaisseur médiane,
     comme un monteur qui zoome ;
   - quand il ne reste que des erreurs de bord (moins de 120 px), il essaie une fois les curseurs
     *Élargir* (+0,5, +1, −0,5 px) et *Durcir / Adoucir* (−0,4, +0,3) — chaque position essayée compte
     pour un geste ;
   puis il **propage dans les deux sens** depuis l'image corrigée, et recommence.
3. Il s'arrête quand le plan est **« propre »** ou après **30 gestes**.

« Propre » (la demande) : encre de contour ≥ 97 %, 0 image perdue, 0 trou, rugosité ≤ l'ancien mode
sur ce plan, fond le long du contour ≤ 3 %, et pour b1 la cigarette présente sur ses 84 images —
mesurés comme au banc, sur l'alpha que l'export écrit (bords Stable v5, trait noir) à la taille de
travail, sur les images de référence ; les images perdues sur toutes les images. Les trous ne sont
comptés que sur les images de référence, quand la référence dit « personnage » à cet endroit : un
creux de fond entre le bras et le corps est aussi un « petit trou » et il est juste (sur les autres
images on ne peut pas les distinguer ; leur nombre est donné à part dans `chiffres.txt`).

Ce que la simulation ne sait pas faire : regarder. Elle ne voit l'erreur que sur les 6 à 13 images de
référence de chaque plan (un vrai monteur la voit partout, mais ne sait pas la mesurer) ; un geste
réel coûte un coup d'œil et un mouvement de souris, compté 5 s dans le temps total.

<!-- SIMULATION-RESULTATS -->

## 5. Ce qui reste pénible

- **Les critères « propre » au pixel près ne sont pas atteints en 30 gestes** (voir §3) : ce qui reste
  après les gros morceaux est réparti le long de tout le contour (fond de 2 à 8 px autour du trait,
  trait noir à moitié transparent), par petits bouts de 1 à 2 px sur chaque image. Un pinceau n'est pas
  le bon outil pour ça ; les curseurs de bord (globaux) aident un peu (b1 : *Durcir* −0,4 fait passer
  l'encre de 94,4 à 96,4 %), pas assez. Il manque un outil de **bord local** (« affiner le bord ici »,
  comme le Refine Edge d'After Effects) qui recalcule le bord dans une zone peinte à partir des
  couleurs : c'est le prochain chantier si l'éditeur plaît.
- **Une propagation recalcule les images entre deux images verrouillées** à partir de la passe IA et de
  l'objet suivi : une correction propagée plus tôt sur ces images peut être en partie perdue si on
  propage ensuite depuis une autre image. Parade dans l'interface : valider (`V`) les images qui sont
  bonnes, ou propager dans un seul sens. À améliorer : propager la *correction* elle-même (ce qui a été
  ajouté / retiré) plutôt que de repartir de la passe IA.
- **Plans à effets (aura SSJ3)** : la passe IA garde des morceaux d'aura et de sol ; trois clics droits
  enlèvent l'aura d'une image et la propagation les reporte, mais la lueur sombre autour des mèches
  (déjà signalée parties 6-7 du rapport roto) reste : aucun geste du monteur simulé ne l'a enlevée, et
  presque toutes les images y sont « à vérifier » (70 à 77 sur 79), ce qui rend la liste inutile sur ce
  type de plan.
- **Clic négatif sur une aura translucide** : SAM y voit parfois l'aura *et* le personnage ; l'éditeur
  prend la plus petite région sûre et seulement le morceau sous le curseur (corrigé pendant le
  chantier : avant, un clic droit sur l'aura emportait Goku), mais il arrive encore qu'un clic n'enlève
  qu'un bout : on recommence ou on peint.
- **Sur CPU** la passe IA est lente (ISNet anime sans suivi, voir §4) et rien n'a été mesuré sur Mac
  ni sous Windows.
- Annuler / rétablir ne survit pas à une fermeture de l'appli (les détourages, si : ils sont sauvés).
- Pas de renommage des personnages (calques) ; « Refaire la passe IA » (ajouté en fin de chantier)
  garde les images corrigées mais remplace la passe IA de toutes les autres.


## 6. Livrables

Sur la machine de mesure, `/srv/data/amv-tools/reports/roto/editeur/` :

- `demo/demo-editeur.mp4` : l'éditeur utilisé (enregistrement de l'écran Xvfb, ~2 min ; passe IA,
  propagations et export accélérés, avec leur durée réelle à l'écran) sur le plan SSJ3 : passe IA,
  deux clics droits sur les rochers, propagation, trois clics droits sur l'aura et un coup de pinceau
  rouge, zoom, propagation, image suivante à vérifier, vues sur noir / sur blanc, lecture, export ;
  `demo/captures/` : 4 captures.
- Pour chacun des 3 plans des exemples (`01-calme-cigarette-black-clover`,
  `02-quatre-personnages-plan-large`, `03-aura-ssj3-camera-qui-tremble`) : `comparaison.mp4`
  (Original / IA seule / après corrections sur noir / après corrections sur rouge, puis zoom ×3
  ralenti), `detourage-alpha.mov` (ProRes 4444 1080p du plan corrigé, l'export de l'éditeur),
  `contour-sur-blanc.png` et `contour-sur-rouge.png` (3 images, zoom ×3 : original / IA seule / après
  corrections), `chiffres.txt` (gestes, temps, mesures avant / après).

Dans le dépôt : le code (§1), `docs/roto-editor-api.md`, les tests, `tests/eval/roto_editor_sim.py`
(simulation, mesure de latence, tableau), `tests/eval/roto_editor_examples.py` (les 3 exemples),
`app/e2e/roto.e2e.mjs` (test de bout en bout), `app/e2e/explore/roto-demo*.{mjs,sh,py}` (la démo).

## 7. Ce que Léandro doit tester

1. **Le parcours de base**, sur un plan que tu connais : Recherche → ouvrir un plan → « Détourer ce
   plan » → *Tous les personnages* → Détourer. Pendant la passe IA, l'interface doit rester
   utilisable (vues, navigation) et Annuler doit marcher.
2. **Les deux façons de corriger**, sur une image où l'IA a pris du décor : Alt+clic (ou clic droit)
   sur le morceau de décor, puis un coup de pinceau rouge sur ce qui reste ; ensuite un clic gauche
   sur une partie oubliée (une main, un objet tenu). Regarde le temps affiché en bas à gauche du
   panneau (« Image corrigée en … ms ») et dis-moi si ça te paraît instantané.
3. **Propager ↔** depuis l'image corrigée, puis lire le plan (espace) : la correction doit suivre le
   personnage, et les images que tu as corrigées ne doivent jamais bouger.
4. **Les images à vérifier** (rouge sous la vidéo, `N`) : sur un plan calme il doit y en avoir peu
   et elles doivent valoir le coup d'œil ; sur un plan à aura (SSJ3) il y en a trop (§5).
5. **Les bords** : décoche « Garder le trait noir », bouge *Rétrécir ↔ Élargir* et *Adoucir ↔ Durcir* en
   vue « Sur blanc » (`3`) zoomée : le changement doit se voir tout de suite.
6. **Plusieurs personnages** : sur le plan large DBS (Goku, Kaio, Beerus, Whis), *Cliquer sur le
   personnage* sur Beerus seul, puis « + Personnage » et clic sur Whis.
7. **Fermer l'éditeur au milieu**, rouvrir le même plan : tout doit être là (corrections, images
   validées, réglages).
8. **Export** : les trois cases cochées → ouvrir le `_alpha.mov` dans After Effects / Premiere /
   Resolve / Vegas (fond transparent, pas de liseré), le `_matte.mov` en luma matte ; puis exporter la
   timeline de ta sélection : le plan détouré doit y être à la place de l'épisode.
9. **Sur le PC Windows sans lancer de GPU lourd en même temps**, et si possible sur le Mac : l'éditeur
   doit s'ouvrir et corriger partout (pas mesuré sur Mac ni sur Windows dans ce chantier).

