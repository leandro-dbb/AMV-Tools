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

<!-- LATENCE -->

<!-- SIMULATION -->

<!-- RESTE -->

<!-- TESTER -->
