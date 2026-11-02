# Release 0.5.0-beta.3 — bêta testable Windows / macOS / Linux

2 novembre 2026, 18 h 36 – 19 h 05. Branche `jarvis/windows-test`
(contient `jarvis/cuts` : coupes TransNetV2 optimisées, et
`jarvis/mac-install-fixes` : corrections du test sur le MacBook M1 Pro ;
50 commits au-dessus de la 0.5.0-beta.2) fusionnée dans `main`, tag
`v0.5.0-beta.3`, release GitHub en **brouillon**.

## Relecture avant fusion

Diff `origin/main...jarvis/windows-test` (74 fichiers, +7 k lignes) passé
au crible : secrets / jetons, chemins absolus personnels, IP privées, noms
de fichiers d'épisodes, code de debug, `.only`, fichiers volumineux,
binaires, poids de modèles. Rapports lus : `coupes.md`,
`windows-rtx3090.md`, `test-installeur-mac.md` (les `v06-*.md` étaient déjà
sur `main` et relus pour les beta.1 / beta.2).

| Constat | Correction (`6a6008a`) |
| --- | --- |
| `windows-rtx3090.md` : **emplacement du fichier contenant le mot de passe** du compte de la VM de test (sur le VPS), mode d'emploi d'accès à la VM (commandes de l'hyperviseur, SSH), détail des services de la machine hôte pendant les coupures, chemin `/home/…` des captures | emplacement du secret et mode d'emploi retirés (« tenu hors dépôt »), tableaux des coupures de la machine hôte remplacés par une phrase, chemin retiré. Aucun mot de passe ni clé n'était dans le dépôt, seulement le chemin du fichier |
| Noms de fichiers d'épisodes réels (groupe de release + CRC) dans `windows-rtx3090.md`, `CHANGELOG.md`, `backend/export/naming.py` (commentaires), `tests/test_windows_fixes.py` et dans le champ `source` des 5 nouvelles vérités terrain `tests/data/cuts_gt/*.json` | remplacés par un nom de fansub anonymisé (`[Groupe] … [ABCD1234]`) ou par « DBS Beerus 01 episode, from 600 s » ; le test du nom des clips garde le même cas (crochets, parenthèses, CRC) |
| `coupes.md` : chemins `/srv/data/…` du NAS et des dossiers de test du VPS ; `test-installeur-mac.md` : dossier `~/dev/…` des captures | descriptions neutres |

Rien trouvé : aucun secret, aucune IP privée, aucun `.only` / `debugger` /
`print` ou `console.log` de debug (les deux `console.log` ajoutés dans
`app/electron/main.ts` sont le journal habituel du processus principal),
aucun poids de modèle. Seuls binaires : 3 planches JPEG de
`docs/reports/coupes/` (186–235 Ko chacune, illustrations du rapport),
gardées. TransNetV2 est téléchargé à la demande (30 Mo), licence dans le
README (architecture MIT, poids Apache-2.0). Mask Mode auto toujours
désactivé par défaut (`enable_mask: False`, `backend/config.py`).

Le chemin du fichier secret reste dans l'historique de la branche fusionnée
(commits de `jarvis/windows-test`) : ce n'est qu'un chemin sur une machine
privée, pas le secret ; réécrire l'historique n'en valait pas la peine.

## Version

`0.5.0-beta.3` dans `app/package.json` / `package-lock.json`, `0.5.0b3`
(PEP 440) dans `pyproject.toml`, `uv.lock`, `backend/__init__.py` et le
titre FastAPI (vérifié : `/openapi.json` → `0.5.0b3`, et dans l'AppImage
du brouillon) ; exemple de `docs/RELEASING.md`. CHANGELOG : la section
« Unreleased » devient `0.5.0-beta.3 — 2026-11-02`, avec un résumé monteur
FR + EN en tête (coupes précises, Windows, Mac, export, finition, rappel :
export timeline Premiere / Resolve / After Effects / Vegas / FCP, synchro
musique, AnimeJaNai, profondeur, éditeur de détourage) et la relecture
(`53667a1`).

## Tests

| Où | Résultat |
| --- | --- |
| Local (CPU), venv à jour (`uv sync --extra cpu`, librosa et OpenTimelineIO présents) | `uv run pytest -m "not slow"` : **606 réussis**, 19 sautés (matériel réel / médias réels absents), 0 échec — plus d'échec « librosa / OpenTimelineIO manquants » ; `tests/test_windows_fixes.py` + `tests/test_shotdetect.py` rejoués après la relecture : 69 réussis ; `tsc --noEmit` OK ; vitest **88/88** ; `npm run build` OK ; sidecar démarré (`/api/status` répond) |
| CI `ci` sur `main` `e7bd696` | ✅ [41495170931](https://github.com/leandro-dbb/AMV-Tools/actions/runs/41495170931) |
| `build` sur le tag `v0.5.0-beta.3` | ✅ [41495764020](https://github.com/leandro-dbb/AMV-Tools/actions/runs/41495764020) : Windows, macOS, Linux, release brouillon |
| Signature macOS | ✅ étape *Check macOS signature* : `Signature=adhoc`, « valid on disk », « satisfies its Designated Requirement » |
| `windows-smoke` sur `main` `e7bd696` | ✅ [41495220316](https://github.com/leandro-dbb/AMV-Tools/actions/runs/41495220316) : installeur NSIS construit et installé en silencieux sur VM Windows (sans GPU), sidecar prêt (CPU), indexation de 2 fichiers à noms accentués, recherche, export de clip, export timeline dans les 6 formats, message hors ligne + repli DirectML/CUDA → CPU, fermeture sans processus restant |
| `e2e-linux` sur `main` (Electron complet, Xvfb, CPU) | ✅ **15/15** du premier coup, [41495826421](https://github.com/leandro-dbb/AMV-Tools/actions/runs/41495826421) (drop, parcours, éditeur de roto) |
| AppImage téléchargée du brouillon, sous Xvfb | ✅ version `0.5.0b3`, se lance, `uv` embarqué exécuté ; sidecar arrêté faute de Python **parce que le test coupe le réseau** (HOME jetable, `UV_OFFLINE=1`, proxy mort) — attendu |

Les runs `ci` / `windows-smoke` déclenchés par le push de la branche
(même arbre que `main`) ont été annulés pour ne pas doubler le travail.

## Installeurs (brouillon)

Page : <https://github.com/leandro-dbb/AMV-Tools/releases> (brouillon
« AMV Tools 0.5.0-beta.3 (bêta de test) » ; l'URL `untagged-…` d'un
brouillon change à chaque modification). `builder-debug.yml` retiré.

| Fichier | Taille |
| --- | --- |
| `AMV.Tools.Setup.0.5.0-beta.3.exe` (Windows x64, NSIS) | 90,9 Mo |
| `AMV.Tools-0.5.0-beta.3-arm64.dmg` (macOS Apple Silicon) | 112,7 Mo |
| `AMV.Tools-0.5.0-beta.3-arm64-mac.zip` (macOS Apple Silicon) | 107,8 Mo |
| `AMV.Tools-0.5.0-beta.3.AppImage` (Linux x64) | 124,0 Mo |
| `amv-tools_0.5.0-beta.3_amd64.deb` (Linux x64) | 85,2 Mo |

(+ les `.blockmap` d'electron-builder.) Les brouillons beta.1 et beta.2 sont
laissés tels quels.

Notes du brouillon (FR + EN) : nouveautés, installation par système
(SmartScreen + invite Visual C++, clic droit → Ouvrir / macOS 15+ « Terminé »
puis « Ouvrir quand même » / `xattr`, `chmod +x` / `.deb`, PyTorch CUDA 13
pour les RTX 50), configuration requise, problèmes connus, checklist de
30 min.

## Ce qui n'a pas été vérifié

- **Ces installeurs beta.3 exacts sur vrai matériel** : le code a été testé
  sur un vrai Windows 11 + RTX 3090 (installeur CI de la branche + builds
  des correctifs suivants) et sur le MacBook M1 Pro (`.dmg` de la branche
  `jarvis/mac-install-fixes`) ; les fichiers du brouillon n'ont été
  installés que sur la VM Windows sans GPU de la CI.
- Le `.deb` et l'AppImage sur un vrai PC Linux de bureau.
- RTX 50, AMD / Intel (DirectML) via l'installeur.
- Import réel des timelines dans Final Cut Pro, Premiere, After Effects,
  Vegas ; l'éditeur de détourage sous Windows.
- Linux après le graphe CUDA / fil NVENC de la passe NVIDIA (activés
  seulement sous Windows ; Linux non remesuré).

## Ce que Léandro doit vérifier

La checklist de 30 min est dans les notes du brouillon. En priorité :

1. Installer sur Windows, Mac et Linux : l'appli s'ouvre, le moteur proposé
   correspond à la carte, Réglages › Diagnostic affiche **0.5.0b3**.
2. Sur le PC NVIDIA : télécharger TransNetV2, indexer un épisode (aperçus
   au survol rapides, pas de bandeau « GPU indisponible »), rechercher
   pendant l'indexation.
3. Sur le Mac : relancer l'appli et chercher tout de suite (plus de
   « Failed to fetch »), un upscale.
4. Export d'un clip (épisode une seule fois dans le nom) et export
   timeline vers son logiciel.
5. Décider de publier ou non le brouillon.
