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

Dans l'appli, l'éditeur affiche le temps total (serveur + transfert + affichage) ; mesuré pendant le
chantier avec trois simulations qui tournaient en même temps sur la carte : clics 98 à 149 ms,
pinceau 170 ms. Le **premier** clic chargeait le modèle (20 à 55 s sous charge) : le modèle de clic
est maintenant préchargé en arrière-plan à l'ouverture d'un plan déjà détouré et à la fin de la
passe IA. Mémoire vidéo de l'éditeur seul (clics + propagation) : 2,4 Go.

Sur CPU (machine sans GPU, test de bout en bout) : la passe IA utilise ISNet anime sans suivi, les
clics la pièce détectée sous le curseur, la propagation le flot optique ; tout tourne en tâche de
fond (l'interface reste utilisable, la progression et le bouton Annuler restent là) mais c'est lent :
voir §4.


<!-- SIMULATION -->

<!-- RESTE -->

## 6. Ce que Léandro doit tester

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

