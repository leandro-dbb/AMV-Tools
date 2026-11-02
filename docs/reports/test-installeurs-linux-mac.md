# Test des installeurs Linux (AppImage et .deb) sur la RTX 3090

Branche `jarvis/install-fixes` (depuis `main` = 0.5.0-beta.3), lundi
2 novembre 2026, 20 h 20 → 23 h 35. Installeurs de départ : l'AppImage et le
`.deb` de la release brouillon **0.5.0-beta.3** (sha256 identiques aux
empreintes de la release), puis les mêmes installeurs reconstruits par la CI
avec les corrections de cette branche.

**Mac : pas refait ici.** Le parcours complet du `.dmg` a déjà été mené sur
le MacBook M1 Pro par une autre session, voir
[test-installeur-mac.md](test-installeur-mac.md).

## En bref

- **Les installeurs Linux de la 0.5.0-beta.3 ne marchent pas** : AppImage
  comme `.deb` s'arrêtent **au bout de 10 s au premier lancement** sur
  « Échec du backend — torch==2.4.1 can't be installed » (§ 1). En cause
  depuis la 0.4.0-alpha : sans option, `uv.lock` choisit sous Linux un
  PyTorch 2.4.1 qui n'existe pas pour Python 3.13, la version que l'appli
  télécharge quand le système n'a pas de Python 3.12 (Debian 12, Fedora,
  Ubuntu récents). La CI ne pouvait pas le voir : son test Linux tourne en
  mode développeur avec Python 3.12. **Corrigé**.
- **Après correction, tout le parcours marche, avec l'AppImage comme avec le
  `.deb`** : installation, moteur CUDA 13, indexation de DBS Beerus 01
  (539 scènes en **3 min 36**), recherche FR/EN, personnage, exports de
  clips et de timelines pour les 6 logiciels + OTIO + EDL, synchro musique,
  finition (upscale, ralenti, profondeur, modèles téléchargés depuis le
  dialogue), détourage exporté en ProRes 4444 avec alpha, relance avec
  recherche immédiate, fermeture propre, désinstallation.
- **9 problèmes trouvés, 8 corrigés** (§ 8) dont deux bloquants ou presque
  pour un utilisateur Linux : le premier lancement (ci-dessus) et **le
  repli sur le processeur juste après l'installation du moteur NVIDIA**
  (l'appli restait sur le CPU jusqu'au redémarrage, § 3). Le premier
  lancement est aussi passé de **12 min à 2 min** (à 5 Mo/s) : 3 Go de
  PyTorch CUDA étaient téléchargés uniquement pour compiler SAM 2.
- Temps (à 5 Mo/s) : premier lancement **2 min 03** jusqu'à l'écran
  d'accueil ; moteur CUDA 13 **16 min** (≈ 4,8 Go, l'écran annonçait
  2,5 Go — corrigé) ; modèles de base au premier usage **17 min** (≈ 5 Go).
- Disque après installation : **≈ 8,2 Go en AppImage, 8,4 Go en `.deb`**
  (appli 125 ou 330 Mo, environnement Python 8,0 Go, Python 112 Mo ; cache
  uv bien purgé : 2 à 4 Mo restent) ; **≈ 14,7 / 14,9 Go** avec tous les
  modèles du parcours (6,3 Go Hugging Face + 137 Mo de finition).

## 0. Comment le test a été mené

| Élément | Ce qui a été fait |
|---|---|
| Machine | debian-nas : Debian 12.15, noyau 6.1, 6 vCPU, 21,5 Go de RAM, RTX 3090 24 Go, pilote 580.196.07. Services GPU du serveur coupés (`amv-gpu-take`) pendant tout le test. |
| Session graphique | Notre propre Xvfb (1600×1000), aucune autre appli. Paquets ajoutés pour le test (avec `apt` bridé) : `xvfb`, `libnss3` (+ `libnspr4`), `fuse3` (+ `libfuse3-3`) ; retirés à la fin (§ 7). |
| Téléchargements | Installeurs récupérés par l'API GitHub (`curl --limit-rate 5M`), empreintes vérifiées. **Tous les téléchargements de l'appli** (Python, paquets, modèles) passaient par un petit proxy HTTP(S) local plafonné à **5 Mo/s** (variables `HTTPS_PROXY`), qui compte aussi les volumes. |
| Profil | Un dossier personnel neuf par essai (`HOME` et dossier temporaire dédiés, supprimés à la fin) : l'appli démarre comme chez un nouvel utilisateur. Épisode DBS Beerus 01 (1,4 Go, H.264) copié dans `~/Vidéos/Épisodes test/`, exports dans `~/Vidéos/Exports AMV été/` (accents et espaces exprès). Musique libre : *Double Drift* (Kevin MacLeod, CC-BY). |
| Pilotage | Playwright `_electron` dans la session Xvfb, une étape à la fois (le pilote de `app/e2e/windows/driver.mjs`, adapté) : vrais clics dans l'interface, une capture par étape. Les deux boîtes natives (« Ajouter un dossier », « Enregistrer sous ») sont remplacées par le chemin qu'elles renverraient. |
| Essais | **A** : AppImage beta.3 (échec, § 1). **B** : AppImage corrigée (1er correctif) — parcours complet, modèles téléchargés. **C0** : `.deb` beta.3 (échec) puis mise à jour vers le `.deb` corrigé sur le même profil. **C** : `.deb` corrigé, parcours complet, modèles Hugging Face repris du cache de l'essai B (même contenu, téléchargement déjà mesuré). **D** : `.deb` avec toutes les corrections, premier lancement → moteur CUDA → Diagnostic sans redémarrage (§ 3). |
| Captures | Hors dépôt (dossier des captures d'installation, sous-dossier `linux/`) : `appimage-beta3-*`, `appimage-*` (B), `deb-beta3-puis-maj-*` (C0), `deb-*` (C et D), nommées par étape (`a01` lancement → `a14` fermeture). |

Limite : debian-nas a le `ffmpeg` 5.1 de Debian dans `/usr/bin`, que l'appli
préfère au sien (il a NVDEC/NVENC). Un bureau sans `ffmpeg` installé
prendrait celui fourni avec l'appli (sans NVDEC) : non testé.

## 1. Premier lancement de la 0.5.0-beta.3 — échoue (AppImage et .deb)

**AppImage.** `chmod +x`, lancement : sur cette Debian 12 qui a `libfuse2`
mais pas `fuse3`, rien ne s'ouvre ; en terminal : `fuse: failed to exec
fusermount`. Un bureau Debian/Ubuntu a d'ordinaire `fuse3` ; le README ne le
disait pas (complété). Avec `fuse3` :

1. 0 s : fenêtre « Récupération du sidecar Python », puis téléchargement de
   Python 3.13.18 (34 Mo).
2. **9,6 s : « ÉCHEC DU BACKEND »** (capture `appimage-beta3-a01-ecran-x`) :

   > Sidecar exited with code 2 before opening port …
   > error: Distribution `torch==2.4.1 @ registry+https://pypi.org/simple`
   > can't be installed because it doesn't have a source distribution or
   > wheel for the current platform
   > hint: You're using CPython 3.13 (`cp313`), but `torch` (v2.4.1) only
   > has wheels with the following Python ABI tag: `cp312`
   > Fermez puis relancez AMV Tools pour réessayer…

   Relancer ne change rien. **Un testeur Linux ne peut pas utiliser la
   beta.3.**

**`.deb`.** `sudo apt install ./amv-tools_0.5.0-beta.3_amd64.deb` : 114 s
(330 Mo dans `/opt/AMV Tools`, une soixantaine de dépendances tirées par
`xdg-utils`, `libnotify4`, `libsecret-1-0` sur ce serveur sans bureau),
entrée de menu et commande `amv-tools` créées, `chrome-sandbox` laissé sans
setuid (espaces de noms utilisateur disponibles). Premier lancement :
**même échec, en 8,9 s**.

**Cause.** L'appli installe son environnement Python au premier lancement
avec `uv run` *sans option*. Sous Linux, `uv.lock` résout alors PyTorch
**2.4.1** (sans roue pour Python 3.13), alors que macOS et Windows ont
2.11.0. `uv` prend un Python du système s'il convient (≥ 3.12), sinon il
télécharge le plus récent autorisé : 3.13. Ça passait donc avec un système
en Python 3.12 (Ubuntu 24.04, le runner de la CI), pas ailleurs. Même lock
depuis la 0.4.0-alpha.

**Correction** (`ab29113`) : sous Linux, ce premier lancement installe
l'option `cpu` (PyTorch 2.11.0 CPU, 182 Mo, roues 3.12 et 3.13) ; l'écran
d'accueil installe ensuite le moteur choisi, comme avant. Tests : vitest
(`app/electron/syncExtra.test.ts`) et un test du lock
(`tests/test_install_lock.py` : pour chaque système, le PyTorch du premier
lancement a des roues Python 3.12 **et** 3.13 — il échoue sur la beta.3).

**Mise à jour d'un profil cassé par la beta.3** (essai C0, vérifié) : après
`sudo apt install ./<nouveau>.deb`, la relance tombe sur l'environnement à
moitié créé, l'appli le répare d'elle-même (« venv is incomplete —
repairing ») et l'écran d'accueil arrive en **2 min 03**.

## 2. Premier lancement corrigé : temps, téléchargements, disque

| Étape (5 Mo/s) | Avant le 2ᵉ correctif | Après |
|---|---|---|
| Lancement → écran d'accueil | **12 min 13** (3,55 Go téléchargés) | **2 min 05** sur un profil neuf (D), 2 min 03 sur le profil réparé (C0), ≈ 0,6 Go |

Les 3 Go en trop : SAM 2 (`sam2`) n'existe qu'en sources ; pour le
compiler, `uv` installait dans un environnement isolé **le PyTorch de
PyPI**, c'est-à-dire sous Linux la version CUDA et ses bibliothèques
`nvidia-*`, puis les jetait. Correction (`0d7f902`) : SAM 2 est compilé
avec le PyTorch de l'appli (`match-runtime` de uv), même roue, déjà
téléchargée ; `uv.lock` ne change pas. Mesuré à part sur la même machine :
731 s / 3,6 Go → 193 s / ≈ 0,5 Go (à 3 Mo/s). Windows en profite aussi à l'installation du moteur
(le PyTorch CPU de PyPI n'y est plus téléchargé pour compiler SAM 2).

**Écran d'accueil** [`appimage-a02-detection`] : « Détecté sur cet
ordinateur : NVIDIA GeForce RTX 3090 », CUDA 13 recommandé et présélectionné
(pilote 580, 24 Go, CC 8.6), profil « Précis ». Deux textes faux, corrigés
(`751ba25`, `59de829`) :

- sous CUDA 12.6 : « CUDA 13 exige 581+ » — c'est le chiffre Windows ; sous
  Linux 580 suffit, et l'appli recommandait justement CUDA 13 avec ce
  pilote 580 ;
- « Environ 2,5 Go à télécharger » pour CUDA 13 (et « 1 à 3 Go » en tête,
  « 3,5 Go » avec TensorRT) : **mesuré 4,8 Go sous Linux** (CUDA 13 pour
  PyTorch **et** CUDA 12 pour onnxruntime-gpu), ≈ 4,3 Go sous Windows
  (rapport Windows). Textes et README : « environ 4,5 à 5 Go ».

**Disque** (mesuré après chaque étape) :

| | AppImage (B) | `.deb` (C) |
|---|---|---|
| Appli | 125 Mo (le fichier) | 330 Mo (`/opt/AMV Tools`) |
| Après le premier lancement | `.venv` 1,7 Go + cache uv 5,2 Go (avant le correctif sam2) | — |
| Après le moteur CUDA 13 | `.venv` 7,9 Go, Python 111 Mo, **cache uv purgé : 2,2 Mo** (« cache 20.47 -> 0.00 GB, disk freed 6.32 GB ») | `.venv` 8,0 Go, Python 112 Mo, cache uv 4,2 Mo (« 10.41 -> 0.00 GB ») |
| Modèles (fin du parcours) | Hugging Face 6,3 Go (SigLIP 2 so400m 4,3 Go, SAM 2.1 large 857 Mo, wd-tagger 723 Mo, BiRefNet 425 Mo) + finition 137 Mo | idem |
| **Total** | **≈ 14,7 Go** | **≈ 14,9 Go** |

Données de l'appli dans `~/.config/amv-tools`, modèles dans
`~/.cache/huggingface` (déjà signalé dans le README pour la désinstallation).

## 3. Moteur CUDA 13 et Diagnostic — un bug corrigé

| | AppImage (B) | `.deb` (C) | `.deb` corrigé (D) |
|---|---|---|---|
| « Installer et continuer » → appli | **16 min 05** (32 paquets, ≈ 4,8 Go) | 16 min 21 | 16 min 07 |
| Diagnostic juste après | **« CPU (forced fallback) »** — bandeau « GPU indisponible — exécution sur cpu » | **idem** | **`cuda — NVIDIA GeForce RTX 3090`, PyTorch 2.11.0+cu130, aucun repli, sans redémarrer** |
| Après fermeture + relance | `cuda — NVIDIA GeForce RTX 3090`, 23,6 Go, PyTorch 2.11.0+cu130, CUDA 13.0, aucun repli [`appimage-a04-diagnostic`] | idem | — |

**Bug** : juste après l'installation du moteur NVIDIA, l'appli tournait sur
le processeur jusqu'au redémarrage (indexation ~20× plus lente pour qui ne
relance pas). Le journal le dit : « autotune postponed: torch 2.11.0+cpu
loaded, 2.11.0+cu130 installed ». Cause : au démarrage, le préchauffage de
la recherche lisait le périphérique, ce qui charge PyTorch — la version CPU
du premier lancement, que l'installation remplace ensuite sur le disque
mais pas dans le moteur déjà lancé. Sous Windows le même chargement échouait
(Visual C++ absent), ce qui avait masqué le problème. Correction
(`e7ebee1`) : le préchauffage attend que le moteur soit installé, comme le
réglage automatique ; un test démarre le moteur sur un dossier de données
neuf, appelle ce que l'écran d'accueil appelle et vérifie que PyTorch n'est
pas chargé (il échoue sans la correction).

(Pendant l'essai B j'avais d'abord attribué ce repli à un appel de
diagnostic de mon script ; l'essai C, sans cet appel, a montré le vrai
coupable.)

## 4. Indexation de DBS Beerus 01 — vérifiée en réel

| | AppImage (B) | `.deb` (C) |
|---|---|---|
| Téléchargement des modèles de base (SigLIP 2 so400m 4,55 Go + wd-tagger) | 16 min 54, progression dans la barre d'état (« SigLIP 2 so400m ↓ … 54 % ») [`appimage-a05-progress-30`] | (cache) |
| **Indexation** (profil Précis, 399 coupes → 539 scènes) | **3 min 36** | 4 min 41 |
| dont détection des coupes | 73 s | 107 s |
| dont découpage des longues scènes (928 images) | 69 s | 68 s |
| dont 539 images clés, miniatures, tags, index | 74 s | 105 s |

Écart B/C : en C l'indexation est partie 13 s après une relance, à froid
(bibliothèques CUDA, tagger et épisode pas encore en cache disque) ; en B
tout était chaud après le téléchargement des modèles. Windows sur la même
carte : 6 min (rapport Windows).

## 5. Recherche, personnage, exports, musique, finition, roto — vérifiés en réel

| Fonction | AppImage (B) | `.deb` (C) |
|---|---|---|
| Recherche FR « Beerus endormi » (1ʳᵉ, modèle à froid) | 123 résultats en 7,8 s, Beerus endormi en tête [`appimage-a06-1`] | 6,8 s, mêmes résultats |
| « gros plan visage », « explosion », « Goku flying in the sky », « a character eating » | 8 / 44 / 164 / 26 résultats en 0,8–0,9 s, pertinents ; aperçu au survol ; plans similaires | identique |
| Personnage « Beerus » (3 exemples ajoutés depuis la recherche intégrée) | 57 scènes, seuil auto 83 %, Beerus partout en tête [`appimage-a07-2`] | identique |
| Export de clip (éditeur) | `Dragon_Ball_Super_-_Beerus_01_scene_468.mp4` en 3,2 s : nom correct, **sans doublon** | idem, 3,8 s |
| Export du panier de dérushage (2 clips) | 3,4 s, mais noms bruts `<groupe>_ … _1080p_ _<CRC>__scene0000.mp4` → **corrigé** (`9498678`) | **vérifié** : `Dragon Ball Super - Beerus 01_scene0000.mp4` |
| Timelines Premiere (`.xml`), Resolve (`.fcpxml`), After Effects (`.jsx`), Vegas (`.txt`), Final Cut (`.fcpxml`), Autre | toutes écrites ; XML bien formés ; chemins `file://` avec accents, espaces et crochets correctement encodés ; JSX en `é` ; Vegas en chemin brut | identique |
| Autre → OTIO et EDL (44 clips d'une recherche) | `.otio` (2 pistes de 44) et `.edl` CMX3600 valides | identique |
| Synchro musique (*Double Drift*) | analyse 21,5 s, montage de 180 plans en 0,4 s, FCPXML (447 marqueurs) | analyse **99,7 s** (voir § 9), reste identique |
| Finition : upscale ×2, ralenti ×2, profondeur | bouton « Télécharger » dans le dialogue pour les trois (2,3 s, 5,6 s, 24 s) ; les 3 tâches finies en 21 s | idem (18,8 s) |
| Noms des sorties de finition | `<groupe>_…_<CRC>__scene10_x2.mp4` → **corrigé** (`942125c`) | **vérifié** : `Dragon_Ball_Super_-_Beerus_01_scene10_x2.mp4` |
| Roto : passe IA | 5 min 12 dont téléchargement BiRefNet + SAM 2.1 large (1,3 Go) ; SAM 3 inaccessible (licence Hugging Face) → SAM 2.1, comme ailleurs | 66 s (modèles en cache) |
| Roto : 2 corrections + propagation | pinceau 0,39 s, clic 0,48 s (« image corrigée en 343 ms, IA 123 ms »), propagation 3,9 s | 2,0 s / 1,7 s / 7,2 s |
| Roto : export | **ProRes 4444 avec alpha** (`yuva444p12le`, 23 images) + matte + 23 PNG en 28 s, plan ajouté à la sélection [`appimage-a11-6`] | 34 s ; noms corrigés (`724b0fa`) **vérifiés** : `Dragon_Ball_Super_-_Beerus_01_scene0009_alpha.mov` |

Bug trouvé en route et corrigé (`89fb57f`) : **un « Plans similaires »
se relançait tout seul à chaque retour sur l'onglet Recherche**. En
pratique (essai B) : retour sur Recherche, on tape « explosion » + Entrée,
puis le « Plans similaires » d'un quart d'heure plus tôt revient et
remplace les 44 résultats par « Aucun plan similaire trouvé »
[`appimage-a08b-otio-edl-ECHEC`]. L'onglet est recréé à chaque visite et
oubliait qu'il avait déjà traité la demande. Vérifié corrigé dans le `.deb`
[`deb-a08b-0-avant-saisie`] + test vitest.

## 6. Relance, recherche immédiate, fermeture — vérifiées en réel

| | AppImage (B) | `.deb` (C) |
|---|---|---|
| Fermeture (bouton de la fenêtre) | 2,3 s, code 0 ; **aucun processus restant** (Electron, uv, Python), carte libérée | 3,3 s, idem |
| Fermeture pendant le réglage automatique | 1,25 s, code 0, rien ne reste | — |
| Relance → interface | 17,7 s (moteur prêt ~15 s après le lancement) | 12,4 s |
| Recherche tapée aussitôt | résultats **15,6 s** après la frappe (SigLIP chargé à froid), 40 s après le lancement ; réglage automatique « en attente » pendant la requête : **la recherche ne l'attend pas** | 22,5 s, 39 s après le lancement, idem |
| Dossier temporaire après fermeture | rien de l'appli ; restent des verrous vides de `uv` (0 octet, un par lancement) et un dossier `torchinductor_<utilisateur>` vide créé par PyTorch | idem |

## 7. Désinstallation

`.deb` (vérifié) : `sudo apt remove amv-tools` en 2 s ; `/opt/AMV Tools`,
la commande `amv-tools`, l'entrée de menu et les icônes disparaissent ;
`dpkg --purge` ne laisse rien (seul avertissement : `/opt` non vide, il
contient d'autres logiciels). **Restent, comme pour toute appli** : les
données de l'utilisateur dans `~/.config/amv-tools` (8,0 Go dont 7,9 Go
d'environnement Python, plus la bibliothèque et les exports) et les modèles
dans `~/.cache/huggingface` (6,3 Go) : à supprimer à la main (README).
AppImage : supprimer le fichier, plus les deux mêmes dossiers.

Mise à jour d'un `.deb` par un autre de **même numéro de version** (une
reconstruction de bêta) : `sudo apt install ./….deb` répond « 0 mis à jour »
et ne fait rien ; il faut `--reinstall`.

Fin du test sur debian-nas : profils de test supprimés, les 61 paquets
installés pour le test (dépendances du `.deb`, `xvfb`, `libnss3`, `fuse3`)
retirés et purgés après une simulation vérifiant que rien d'autre ne
partait ; modèles gardés dans un cache de test hors profil (6,3 Go) pour un
prochain essai ; carte rendue (`amv-gpu-release`), HyperQwen relancé.

## 8. Bugs trouvés

| # | Bug | Gravité | Correction | Test |
|---|---|---|---|---|
| 1 | Premier lancement Linux impossible (PyTorch 2.4.1 sans roue Python 3.13), AppImage et `.deb` | **bloquant** | `ab29113` | vitest + pytest (lock) ; vérifié en réel (B, C, C0, D) |
| 2 | Juste après l'installation du moteur NVIDIA : repli CPU jusqu'au redémarrage | **haute** | `e7ebee1` | pytest (moteur démarré pour de vrai) ; **vérifié en réel** (D : CUDA actif juste après l’installation, sans redémarrage) |
| 3 | Premier lancement : 3,2 Go de PyTorch CUDA téléchargés pour compiler SAM 2 (12 min au lieu de 2 à 5 Mo/s) | haute (temps, disque, forfait) | `0d7f902` | pytest ; vérifié en réel (C0 : 2 min 03) |
| 4 | « Plans similaires » rejoué à chaque retour sur Recherche, écrase la recherche tapée | moyenne | `89fb57f` | vitest ; vérifié en réel |
| 5 | Noms des clips du panier de dérushage : groupe, qualité et CRC du fichier | basse | `9498678` | pytest ; vérifié en réel |
| 6 | Noms des sorties de finition : idem, et lettres accentuées perdues | basse | `942125c` | pytest ; vérifié en réel |
| 7 | Noms des exports du détourage : idem | basse | `724b0fa` | pytest ; vérifié en réel |
| 8 | Écran d'accueil : tailles de téléchargement fausses (2,5 Go annoncés, 4,8 mesurés) et « CUDA 13 exige 581+ » faux sous Linux | basse | `59de829`, `751ba25` | vitest (i18n) ; vérifié en réel |
| 9 | AppImage sans `fusermount` (paquet `fuse3`) : rien ne s'ouvre, message cryptique en terminal | basse (bureaux : présent) | README (`59de829`) | — |

## 9. Ce qui reste / non vérifié

- **Synchro musique 5× plus lente dans l'essai C** (99,7 s contre 21,5 s),
  ainsi que les retouches du roto (2 s au lieu de 0,4 s). Pas le réglage
  automatique (il était en attente, journal à l'appui). La machine était à
  ce moment-là en manque de mémoire (21,5 Go, tous les modèles chargés,
  `kswapd` actif, ~105 s de blocage mémoire cumulé depuis le démarrage) :
  hypothèse la plus probable, non démontrée ; à remesurer sur un PC avec
  plus de RAM.
- Numérotation des scènes dans les noms de fichiers toujours mélangée :
  numéro de scène en base (`scene_468`, finition `scene10`) contre position
  dans l'épisode (`scene0000` dérushage, `scene0009` roto) — déjà noté par
  le test Mac (bug 14), non corrigé.
- Une reconstruction de même version (`0.5.0~beta.3`) ne s'installe pas par
  `sudo apt install ./….deb` (« 0 mis à jour ») : il faut `--reinstall`.
  Sans effet pour des versions numérotées normalement.
- Réglage automatique : il ne cède la place qu'entre deux mesures et aux
  requêtes de l'interface ; rien de gênant vu ici.
- Non testé : un vrai bureau (GNOME/KDE, Wayland), double-clic dans le
  gestionnaire de fichiers et lancement par le menu (l'entrée `.desktop`
  existe et pointe sur le bon binaire), Ubuntu et Fedora, un PC **sans**
  `ffmpeg` système, le glisser-déposer, les boîtes natives (remplacées par
  leur résultat), l'import réel des timelines dans les logiciels de
  montage, la mise à jour automatique.
- La CI ne teste toujours pas le premier lancement de l'appli **empaquetée**
  sous Linux (son E2E tourne depuis les sources avec Python 3.12) : le
  test du lock couvre la cause, pas l'ensemble.
- Le job macOS du build de la branche a échoué une fois sur une erreur du
  trousseau de la machine GitHub (`security set-key-partition-list`), sans
  rapport avec ces changements : le build de la branche réussissait (3 builds), puis échoue depuis que des
secrets de signature Apple (`CSC_LINK`, `CSC_KEY_PASSWORD`, `APPLE_*`) ont
été ajoutés au dépôt ce soir (22 h 36) : electron-builder importe le
certificat et le trousseau refuse le mot de passe (« SecKeychainUnlock: The
user name or passphrase you entered is not correct »). Relancé une fois,
même erreur. À vérifier par qui a ajouté ces secrets (mot de passe du
`.p12`) ; les builds Windows et Linux passent.

## 10. Vérifications

- VPS, sur le dernier commit de code (`e7ebee1`) : `uv run pytest -m "not
  slow"` → **615 passés, 19 ignorés, 0 échec** ; `cd app && npx tsc
  --noEmit && npx vitest run && npm run build` → **OK** (92 tests vitest).
  Un test de `tests/test_finish.py` a échoué une fois au milieu de la suite
  puis passé (seul, et la suite entière deux fois, avec et sans les
  changements) : intermittent, sans lien.
- CI de la branche : `ci` et `windows-smoke` verts à chaque poussée ;
  `build` Linux et Windows verts (macOS : voir § 9).
- En réel sur debian-nas : chaque correction vérifiée dans l'appli
  installée depuis les installeurs reconstruits par la CI (essais B, C0, C,
  D) ; voir les colonnes « vérifié » du § 8.
- Nouveaux tests : `tests/test_install_lock.py` (lock : PyTorch du premier
  lancement installable en Python 3.12 et 3.13 sur chaque système ; SAM 2
  compilé avec le PyTorch de l'appli), `tests/test_install_fixes.py` (noms
  des fichiers du dérushage, de la finition et du détourage ; moteur démarré
  sur un dossier neuf sans charger PyTorch), `app/electron/syncExtra.test.ts`,
  `app/src/searchplus.test.ts`.
