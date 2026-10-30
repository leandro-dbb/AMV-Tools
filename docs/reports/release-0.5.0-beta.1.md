# Release 0.5.0-beta.1 — bêta testable Windows / macOS / Linux

Nuit du 29 au 30 octobre 2026. Branche `jarvis/v06` (81 commits : main 0.4
\+ Windows / timeline (v06 A) + parcours UI (v06 B) + synchro musique (v06 C)
\+ modèles AnimeJaNai / VDA Large (chantier 4b) + corrections de la nuit
(v06 fix)) fusionnée dans `main`, tag `v0.5.0-beta.1`, release GitHub en
**brouillon**. L'éditeur de roto (`jarvis/v06-roto-editor`) n'est **pas**
fusionné.

## Relecture avant fusion

Diff `origin/main...jarvis/v06` (172 fichiers, +17,5 k lignes) passé au
crible : secrets / jetons, chemins absolus personnels, IP privées, noms de
fichiers d'épisodes dans la doc, code de debug, `.only` dans les tests,
fichiers volumineux, binaires, poids de modèles.

| Constat | Correction |
| --- | --- |
| `app/e2e/explore/*.mjs` : 8 scripts d'exploration jetables (`dbg-keep.mjs`…) pleins de `console.log`, liés à des médias locaux (« Black Clover ») | supprimés (`087f5bf`), commentaire de `helpers.mjs` et rapport v06 B ajustés |
| Rapports v06 : chemins `/home/leandro/…`, `/srv/data/…`, `~/dev/amv-testdata/…`, noms de fichiers d'extraits d'épisodes (`… - extrait.mkv`, `black_clover_s02e04_extract_h264.mp4`) | remplacés par des descriptions neutres (`ccc781b`) |
| macOS : sans certificat, `CSC_IDENTITY_AUTO_DISCOVERY=false` désactivait **toute** signature ; Apple Silicon refuse le code non signé | hook `afterPack` `app/scripts/adhoc-sign.cjs` : signature ad hoc du bundle entier (sans hardened runtime) ; étape *Check macOS signature* dans `build.yml` qui vérifie le `.app` et le contenu du `.zip` (`22bf1bd`) |

Rien trouvé : aucun secret, aucune IP privée, aucun `.only` / `debugger` /
`print` de debug dans le code de l'appli, aucun binaire (le plus gros
fichier ajouté est `uv.lock`), aucun poids de modèle (Beat This! est du code
vendorisé sous MIT, ses poids se téléchargent). Gardé volontairement : les noms de
séries dans la prose des rapports (déjà le cas sur main), les vérités terrain
`tests/data/cuts_gt/*.json` (le champ `video` sert à retrouver l'extrait local,
test sauté sans lui), les crochets de test Electron (`AMV_SMOKE_*`,
`AMV_E2E_SKIP_SYNC`, inactifs sans variable d'environnement).

Vérifié : **Mask Mode auto reste désactivé par défaut** (`enable_mask: False`
dans `backend/config.py`, inchangé par la branche). Modèles non commerciaux
(AnimeJaNai CC-BY-NC-SA-4.0, VDA Large CC-BY-NC-4.0) : téléchargés à la
demande, marqués « usage non commercial » dans Settings > Models et le README,
conformément à la décision du chantier 4b.

## Version

`0.5.0-beta.1` dans `app/package.json` / `package-lock.json`, `0.5.0b1`
(PEP 440) dans `pyproject.toml`, `uv.lock`, `backend/__init__.py` et le titre
FastAPI ; exemple de `docs/RELEASING.md`. La version affichée dans
Settings > Diagnostic vient de `pyproject.toml` (0.5.0b1), celle des
installeurs de `package.json`. CHANGELOG : section `0.5.0-beta.1 — 2026-10-30`
avec un résumé monteur en français et en anglais, au-dessus du détail par
chantier. README : installation macOS / Linux.

## Tests

| Où | Résultat |
| --- | --- |
| VPS, `uv run --no-sync pytest -m "not slow"` | 447 passés, 9 ignorés, 5 désélectionnés (4 min 45) |
| VPS, `npx tsc --noEmit`, `npx vitest run`, `npm run build` | OK, 66 tests vitest |
| CI `ci` sur `jarvis/v06` (version 0.5.0b1) | ✅ [40983871498](https://github.com/leandro-dbb/AMV-Tools/actions/runs/40983871498) |
| `windows-smoke` sur `jarvis/v06` (installation NSIS réelle sur VM Windows) | ✅ [40983871586](https://github.com/leandro-dbb/AMV-Tools/actions/runs/40983871586) |
| `build` de validation (dispatch sur `jarvis/v06`, avant tag) | ✅ [40983868966](https://github.com/leandro-dbb/AMV-Tools/actions/runs/40983868966) — macOS : `Signature=adhoc`, `valid on disk`, `satisfies its Designated Requirement`, aussi après extraction du `.zip` |

Aucun test désactivé.

## Fusion, tag, build

- `main` : merge `--no-ff` de `jarvis/v06` → `5cf4de7` (arbre identique à la
  branche testée), poussé.
- Tag annoté `v0.5.0-beta.1` sur `5cf4de7`, poussé.

| Run | Résultat |
| --- | --- |
| `ci` sur main | ✅ [40984453806](https://github.com/leandro-dbb/AMV-Tools/actions/runs/40984453806) |
| `build` sur le tag (3 OS + release) | ✅ [40984695256](https://github.com/leandro-dbb/AMV-Tools/actions/runs/40984695256) |
| `windows-smoke` sur main (dispatch) | ✅ [40984473637](https://github.com/leandro-dbb/AMV-Tools/actions/runs/40984473637) |

Release **brouillon**, marquée pré-version, titre « AMV Tools 0.5.0-beta.1
(bêta de test) », notes FR + EN (nouveautés, installation par système,
RTX 50, configuration, problèmes connus, checklist de test de 30 min). Page :
<https://github.com/leandro-dbb/AMV-Tools/releases> (l'URL d'un brouillon
change à chaque modification). `builder-debug.yml` (fichier interne
d'electron-builder) retiré des fichiers joints.

| Fichier | Taille |
| --- | --- |
| `AMV.Tools.Setup.0.5.0-beta.1.exe` (Windows x64, NSIS) | 90,7 Mo |
| `AMV.Tools-0.5.0-beta.1-arm64.dmg` (macOS Apple Silicon) | 112,6 Mo |
| `AMV.Tools-0.5.0-beta.1-arm64-mac.zip` (macOS Apple Silicon) | 107,7 Mo |
| `AMV.Tools-0.5.0-beta.1.AppImage` (Linux x64) | 123,9 Mo |
| `amv-tools_0.5.0-beta.1_amd64.deb` (Linux x64) | 85,1 Mo |

(+ les `.blockmap` des mises à jour différentielles.)

Contrôle de l'AppImage publiée sur le VPS (Debian 12, Xvfb, profil jetable,
réseau coupé exprès) : extraction OK (`resources/backend`, `pyproject.toml`,
`uv.lock`, `.uv/uv` présents), l'appli Electron démarre et tient 40 s ; le
sidecar s'arrête au provisionnement de Python, comme attendu sans réseau.

## Ce qui n'a pas été vérifié sur vrai matériel

- L'installeur macOS (seulement construit et vérifié par `codesign` sur le
  runner) : ouverture après quarantaine, Gatekeeper sur macOS 15.
- Les paquets Linux (seulement construits) : AppImage (FUSE, bac à sable
  Chromium sous AppArmor d'Ubuntu 24.04), `.deb`.
- Les moteurs GPU installés par l'appli empaquetée (CUDA 12.6 / 13, DirectML,
  MPS) : la VM Windows n'a pas de GPU, elle couvre le CPU et les chemins
  dégradés.
- Une carte RTX 50 : le moteur proposé est CUDA 13 si le pilote est ≥ 581
  (Windows) / 580 (Linux) ; avec un pilote plus ancien l'appli propose
  CUDA 12.6, qui ne gère pas ces cartes → mettre à jour le pilote d'abord.
