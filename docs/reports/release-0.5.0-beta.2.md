# Release 0.5.0-beta.2 — bêta testable Windows / macOS / Linux, avec l'éditeur de roto

30 octobre 2026, 7 h – 8 h. Branche `jarvis/v06-roto-editor` (éditeur de
détourage + pipeline roto Stable v5 de `jarvis/v05-roto-stable`, 88 commits
au-dessus de la base v06) fusionnée dans `main` par-dessus la 0.5.0-beta.1,
tag `v0.5.0-beta.2`, release GitHub en **brouillon**.

## Fusionner ou non ?

Le rapport de la branche (`docs/reports/v06-roto-editeur.md`) dit l'éditeur
**utilisable** : parcours complet testé de bout en bout dans l'appli Electron
(5/5 sur CPU), corrections en 0,1-0,2 s sur RTX 3090, propagation qui
n'écrase jamais une image corrigée, export ProRes 4444 vérifié. Sa limite est
assumée et documentée : les gros défauts partent en quelques gestes, la
finition au pixel du bord non. Rien de cassé : **fusionné**, et marqué
« bêta » dans l'appli. Le Mask Mode auto reste désactivé par défaut
(`enable_mask: False` dans `backend/config.py`, inchangé).

## Relecture avant fusion

Diff `origin/main...jarvis/v06-roto-editor` (62 fichiers, +15 k lignes),
passé au crible : secrets / jetons, chemins absolus personnels, IP privées,
noms de fichiers d'épisodes, code de debug, `.only`, fichiers volumineux,
binaires, poids de modèles.

| Constat | Correction |
| --- | --- |
| `tests/eval/roto_editor_examples.py` : racine du banc `/srv/data/…` et **trois noms de fichiers d'épisodes** complets (release groups compris) en dur | remplacés par des options `--root` et `--src key=/chemin` comme `roto_eval.py` (`377d356`) |
| `tests/eval/roto_editor_sim.py table` : `--root` par défaut sur `/srv/data/…` | option obligatoire (`377d356`) |
| `app/e2e/explore/roto-*` : 6 scripts ponctuels d'enregistrement de la démo, liés à une bibliothèque indexée locale (`sceneId: 6`, requêtes sur des plans précis) | supprimés, comme les scripts d'exploration retirés en beta.1 (`e69f549`) |
| Rapports roto (`v06-roto-editeur.md`, `chantier-roto.md`) : chemins `/srv/data/…` | remplacés par des descriptions neutres (`c811251`) |
| L'éditeur n'était pas signalé comme bêta dans l'interface | badge « bêta » sur le bouton *Détourer ce plan* et dans la barre de l'éditeur, FR / EN (`7316477`) ; README « Roto Editor (beta) » |

Rien trouvé : aucun secret, aucune IP privée, aucun `.only` / `debugger` /
`print` ou `console.log` de debug dans `backend/` ou `app/src/`, aucun
binaire, aucun poids de modèle (le plus gros fichier ajouté est
`tests/eval/roto_eval.py`, 1 622 lignes). Nouveaux modèles téléchargés à la
demande et listés avec leur licence dans le README : ToonOut (MIT, données
CC-BY-4.0), SAM 3 (SAM License, accès restreint, jeton Hugging Face de
l'utilisateur). Gardé : les noms de séries et de personnages dans la prose
des rapports (comme sur main).

## Fusion

La branche partait de la base v06 (avant la beta.1) : `main` (beta.1) a d'abord
été fusionnée dans la branche, **sans conflit** (`665ba68`), puis version et
CHANGELOG (`9a77958`), puis `merge --no-ff` dans `main` depuis le dépôt
principal (`a5585fc`, arbre identique à la branche testée), push.

**Correctif après fusion** (`64f8b66`) : la CI de `main` a échoué sur
`tests/test_finish.py::test_cancelled_cutout_stops_encoding_and_leaves_no_file`
(`'FakeState' object has no attribute 'get_sam2'`). Sur le runner, le paquet
`sam2` est installé : le détourage passe alors par le suivi SAM 2 de la
pipeline Stable, ce que le faux état du test ne prévoyait pas ; sur la machine
de dev, `sam2` est absent et le test passait. Le test porte sur l'annulation
pendant l'encodage : il coupe maintenant le suivi explicitement
(`mask_auto_track: False`). Pas de bug dans l'appli (le vrai `AppState` a
`get_sam2`).

## Version

`0.5.0-beta.2` dans `app/package.json` / `package-lock.json`, `0.5.0b2`
(PEP 440) dans `pyproject.toml`, `uv.lock`, `backend/__init__.py` et le titre
FastAPI (vérifié : `/openapi.json` → `0.5.0b2`) ; exemple de
`docs/RELEASING.md`. CHANGELOG : section `0.5.0-beta.2 — 2026-10-30` avec un
résumé monteur FR + EN (éditeur de détourage, détourage plus fiable, rappel
de la beta.1 : export timeline Premiere / Resolve / After Effects / Vegas /
FCP, synchro musique, AnimeJaNai, profondeur, corrections), puis le détail
de l'éditeur et des chantiers roto (déplacés depuis la section beta.1, où la
fusion les avait rangés alors qu'ils n'y étaient pas).

## Tests

| Où | Résultat |
| --- | --- |
| Local (CPU), branche fusionnée | `uv run pytest -m "not slow"` : **524 réussis**, 9 sautés (matériel réel / médias réels absents), puis `tests/test_finish.py` 40/40 après le correctif ; `tsc --noEmit` OK ; vitest **84/84** ; `npm run build` OK ; sidecar démarré (`/api/status` répond, 18 routes `/api/roto`) |
| CI `ci` sur `main` `64f8b66` | ✅ [41013177856](https://github.com/leandro-dbb/AMV-Tools/actions/runs/41013177856) (le run du commit de fusion `a5585fc`, [41012784271](https://github.com/leandro-dbb/AMV-Tools/actions/runs/41012784271), avait échoué : voir correctif ci-dessus) |
| `build` à blanc sur `main` (sans tag) | ✅ [41012793649](https://github.com/leandro-dbb/AMV-Tools/actions/runs/41012793649), 3 OS |
| `build` sur le tag `v0.5.0-beta.2` | ✅ [41013419667](https://github.com/leandro-dbb/AMV-Tools/actions/runs/41013419667) : Windows, macOS, Linux, release brouillon |
| Signature macOS | ✅ étape *Check macOS signature* : `Signature=adhoc`, « valid on disk », « satisfies its Designated Requirement » |
| `windows-smoke` sur `main` `64f8b66` | ✅ [41013419672](https://github.com/leandro-dbb/AMV-Tools/actions/runs/41013419672) : installeur 0.5.0-beta.2 installé en silencieux sur VM Windows, sidecar prêt (CPU), indexation, recherche, export, fermeture (aussi vert sur `a5585fc` : [41012795922](https://github.com/leandro-dbb/AMV-Tools/actions/runs/41012795922)) |
| `e2e-linux` sur `main` (Electron complet, Xvfb, CPU) | ✅ **15/15** à la 2e tentative de [41014339336](https://github.com/leandro-dbb/AMV-Tools/actions/runs/41014339336) : drop 3/3, parcours 7/7, **éditeur de roto 5/5** (indexation, passe IA, pinceau + clic, propagation, export ProRes 4444 ; badge bêta visible sur les captures). 1re tentative : 8/9, voir ci-dessous |
| AppImage téléchargée du brouillon, VPS sous Xvfb | ✅ se lance, interface affichée, `uv` embarqué exécuté ; sidecar arrêté faute de Python **parce que le test coupe le réseau** (HOME jetable, `UV_OFFLINE=1`, proxy mort, pour éviter un téléchargement de PyTorch) : message d'erreur clair à l'écran. `.deb` : paquet `amv-tools` `0.5.0~beta.2` amd64, dépendances GTK / NSS habituelles |

**`e2e-linux`** : ce workflow (manuel) n'avait jamais tourné. 1re tentative
sur `main` : `parcours 1. premier lancement` a attendu l'écran d'accueil
3 min pendant que l'appli restait sur « Premier lancement — récupération du
sidecar Python » (sidecar pas encore à l'écoute), le reste du parcours a été
sauté (série) ; drop 3/3 et éditeur de roto 5/5 passaient dans le même run.
Le même workflow sur le tag `v0.5.0-beta.1` : 15/15
([41015087125](https://github.com/leandro-dbb/AMV-Tools/actions/runs/41015087125)),
premier lancement en 4 s. 2e tentative sur `main`, sans aucun changement :
**15/15**, premier lancement en 4,2 s. Rien dans la fusion ne touche le
démarrage du sidecar ni l'écran d'accueil (diff `v0.5.0-beta.1..v0.5.0-beta.2`
sur `app/electron` : aucun). **Échec intermittent, non reproduit**, cause
non trouvée (lenteur ponctuelle du runner ?) : à surveiller. Si Léandro voit
le premier lancement rester plus de 3 min sur « récupération du sidecar
Python », c'est à signaler avec le journal affiché.

## Installeurs (brouillon)

Page : <https://github.com/leandro-dbb/AMV-Tools/releases> (brouillon
« AMV Tools 0.5.0-beta.2 (bêta de test) » ; l'URL `untagged-…` d'un brouillon
change à chaque modification). `builder-debug.yml` retiré du brouillon.

| Fichier | Taille |
| --- | --- |
| `AMV.Tools.Setup.0.5.0-beta.2.exe` (Windows x64, NSIS) | 90,8 Mo |
| `AMV.Tools-0.5.0-beta.2-arm64.dmg` (macOS Apple Silicon) | 112,7 Mo |
| `AMV.Tools-0.5.0-beta.2-arm64-mac.zip` (macOS Apple Silicon) | 107,8 Mo |
| `AMV.Tools-0.5.0-beta.2.AppImage` (Linux x64) | 124,0 Mo |
| `amv-tools_0.5.0-beta.2_amd64.deb` (Linux x64) | 85,2 Mo |

(+ les `.blockmap` d'electron-builder.) Le brouillon de la 0.5.0-beta.1 est
laissé tel quel.

Notes de release du brouillon (FR + EN) : nouveautés, installation par
système (SmartScreen, clic droit → Ouvrir / `xattr`, `chmod +x` / `.deb`,
PyTorch CUDA 13 pour les RTX 50), configuration requise, problèmes connus,
checklist de test de 30 min.

## Ce qui n'a pas été vérifié

- L'installeur macOS et le `.deb` sur une vraie machine (construits et
  signés ad hoc, pas lancés).
- L'**éditeur de détourage sous Windows et sur Mac** : mesuré sur RTX 3090
  (Linux) et testé de bout en bout sur CPU Linux seulement. `windows-smoke`
  n'ouvre pas l'éditeur.
- Les moteurs GPU (CUDA, DirectML, Apple Silicon) via l'installeur, une RTX 50.
- Sur le runner Linux (CPU), une correction par clic sur un clip synthétique
  a pris 5,6 s (affiché par l'éditeur) ; le rapport de l'éditeur donne 12 à
  180 ms sur CPU sur un vrai plan : à regarder sur une machine sans GPU.

## Ce que Léandro doit vérifier

La checklist de 30 min est dans les notes du brouillon. En priorité :

1. Installer sur Windows, Mac et Linux : l'appli s'ouvre, le moteur proposé
   correspond à la carte, Réglages > Diagnostic affiche **0.5.0b2**.
2. Sur le PC NVIDIA : *Détourer ce plan* (badge bêta), passe IA, un clic droit
   sur du décor, un coup de pinceau, un clic sur une partie oubliée, propager,
   export ProRes 4444 ouvert dans le logiciel de montage, timeline exportée
   avec le plan détouré.
3. Sur le Mac : l'éditeur s'ouvre et corrige (jamais essayé sur Mac).
4. Export timeline vers son logiciel et synchro musique (inchangés depuis la
   beta.1, à confirmer sur l'installeur).
5. Décider de publier ou non le brouillon.
