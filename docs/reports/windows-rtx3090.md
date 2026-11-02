# AMV Tools sur un vrai Windows 11 + RTX 3090

Branche `jarvis/windows-test` (depuis `jarvis/cuts` = 0.5.0-beta.2 + coupes
TransNetV2 optimisées), nuit du 2 novembre 2026. Première fois qu'AMV Tools
tourne sur un vrai Windows avec une carte NVIDIA : la RTX 3090 de debian-nas,
prêtée à une VM Windows le temps des tests.

**En bref (mise à jour du 2 novembre au soir, voir § 8 « Corrections »).** Proxys de survol 39 → 3 s, indexation -25 % à conditions égales, surcoût du réseau +10,2 → +5,9 %, coupes identiques ; cache uv purgé (gain réel 1 Go : le reste était compté en double). **Constat initial :** Tout le parcours marche sur Windows 11 + RTX 3090 **après 4
corrections** (dont 2 bloquantes pour un utilisateur sur un PC neuf) ; la
précision des coupes est **identique à Linux, image par image**. L'indexation
est en revanche **~1,7× plus lente que sous Linux sur la même carte**
(119 s contre 71 s pour un épisode), surtout à cause des proxys de survol
(×6 à ×8) ; le réseau de coupes coûte **+10 %** sous Windows (+4,8 % sous
Linux).

Captures : `/home/leandro/dev/amv-win-captures/` (83 images, nommées par
étape : `a01-…` premier lancement → `a11-…` roto).

## 1. La VM

VM 501 « Windows-AMV-Test » sur le Proxmox :

| Réglage | Valeur |
|---|---|
| Type | `ostype win11`, `machine q35`, `bios ovmf`, `efidisk0` (clés Microsoft pré-enrôlées), `tpmstate0` v2.0 |
| CPU / RAM | `cpu host`, 8 cœurs, 16 Go (`balloon 0`) |
| Disque | 100 Go virtio-scsi sur `local-lvm` (discard, iothread, ssd) |
| Réseau | virtio sur `vmbr0`, **`rate=8`** (8 Mo/s max, dans les deux sens) |
| GPU | `hostpci0: 0000:0a:00,pcie=1` (affichage virtuel gardé pour la console) |
| Démarrage | `onboot 0`, `agent 1` |
| Système | Windows 11 Enterprise **Évaluation** 26H2 (build 26300.9457), **français**, 90 jours sans clé |

- ISO officielles (Microsoft Evaluation Center, fedorapeople virtio-win
  stable) téléchargées sur l'hôte avec `curl --limit-rate 8M`, l'une après
  l'autre ; taille de l'ISO Windows vérifiée à l'octet près (8 234 248 192 o ;
  le serveur a coupé à 4,7 Go, reprise `-C -`). Installeur AMV : sha256 =
  digest de la release brouillon GitHub ; Node 22.23.1 et uv 0.13.6 : sha256
  officiels vérifiés.
- Installation **sans intervention** (ISO `autounattend` construite avec
  genisoimage) : partitionnement GPT automatique, pilote `vioscsi` injecté,
  compte local `amv` (mot de passe aléatoire, uniquement dans
  `~/.hermes/amv-win.secret` sur le VPS) en ouverture de session
  automatique ; au premier logon : pilotes VirtIO réseau/balloon/série,
  qemu-guest-agent, OpenSSH Server (clé du VPS, mot de passe refusé, shell
  PowerShell), réseau privé, **Windows Update, Delivery Optimization et
  UsoSvc désactivés**, veille et écran de veille coupés. Installation lancée
  à 02:28, SSH opérationnel à 02:48 (20 min).
- Pilote **NVIDIA 621.01 Game Ready** (21 oct. 2026). Le paquet lancé
  directement avec `-s` sort en code 9 sans rien installer ; extrait avec le
  `tar` de Windows, son `setup.exe Display.Driver -s -noreboot -noeula -clean`
  passe (5,5 min). `nvidia-smi` voit la 3090 (WDDM, CUDA 13.5).
- **Sécurité alimentation, appliquée avant toute charge GPU** :
  `nvidia-smi -pl 250` et `-lgc 210,1500` acceptés par le pilote Windows,
  vérifiés (`-q -d CLOCK,POWER` : limite 250 W, horloges 210–1500 MHz), et
  rendus persistants par une tâche planifiée au démarrage en SYSTEM
  (`AMV-GPU-Cap`, journal `C:\amv\gpu-cap.log`) — réappliqués tout seuls au
  redémarrage de 04:49. Sous charge : SM à 1500 MHz, 111 W max observé. Le
  mode persistance n'existe pas sous Windows (avertissement sans effet).

### Coupures de debian-nas

| | Heure (Paris) |
|---|---|
| Coupure 1 : `qm shutdown 401` (propre, 37 s) | 02:26:09 → arrêtée 02:26:47 |
| Retour 1 (restauration automatique : ma session a été coupée à 04:33 ; 501 arrêtée proprement 04:33:07–04:33:27, puis `qm start 401`) | 04:33:39 ; vérifié à 04:48 : `gpu-clock-cap` actif, 250 W, Nextcloud 200, Jellyfin et HyperQwen healthy |
| Coupure 2 (courte, pour finir le bench : la carte était nécessaire) | 04:48:51 → 501 démarrée 04:49:13 |
| Retour 2 (restauration par mes soins) | 05:35:17 (détail ci-dessous) |

Restauration finale : `qm shutdown 501` lancé à 05:34:13 (arrêt propre),
`qm start 401` à 05:34:44, SSH de debian-nas à 05:35:17. Vérifié à 05:39:48 :
`systemctl is-active gpu-clock-cap` → `active` ; `nvidia-smi` : RTX 3090
revenue, pilote 580.196.07, limite 250 W, SM à 1500 MHz sous charge
(bridée) ; conteneurs `hyperqwen-single-1` (healthy), `jellyfin` (healthy),
`nextcloud`, `nextcloud-db`, `nextcloud-redis` Up ; Nextcloud répond 200.
Coupure totale : 2 h 07 puis 46 min.

## 2. Résultats des tests (parcours réel, installeur 0.5.0-beta.2)

L'appli tourne dans la session interactive (ouverture de session
automatique), lancée par une tâche planifiée « interactive » et pilotée par
Playwright `_electron` dans cette session (`app/e2e/windows/driver.mjs` :
étapes envoyées par SSH, une capture à chaque étape). Le dossier d'épisodes
est volontairement piégé : `C:\Users\amv\Vidéos\Épisodes test\[SubsPlease]
Dragon Ball Super - Beerus - 01 (1080p) [B3FEF9B7].mkv`, exports dans
`…\Vidéos\Exports AMV été`.

| Test | Résultat | Captures |
|---|---|---|
| Installation silencieuse (`/S`) | **OK** — 330 Mo dans `%LOCALAPPDATA%\Programs\amv-tools` | — |
| Premier lancement → écran d'accueil | **OK, 297 s** (Python 3.13.18 62 Mo + paquets de base) | `a01-*` |
| Détection matérielle | **OK** — RTX 3090, 24 Go, pilote 621.01, CC 8.6 ; **CUDA 13** et profil « Précis » présélectionnés | `a02-detection` |
| Installation du moteur CUDA 13 | **ÉCHEC sur Windows neuf** (VC++ manquant, bug 1) → **OK après correctif** : runtime installé, `torch 2.11.0+cu130, CUDA 13.0, RTX 3090` | `a03-echec`, `a03c-*` |
| Diagnostic | **ÉCHEC** : « GPU indisponible, repli CPU » (bug 2) → **OK après correctif** : `cuda — NVIDIA GeForce RTX 3090`, tagger PyTorch CUDA fp16, aucun repli | `a04-diagnostic`, `a04b-diagnostic` |
| Indexation DBS 01 complet (profil Précis, sous-segmentation) | **OK** — 539 scènes en **360 s** (voir § 3) | `a05-*` |
| Recherche FR / EN | **OK** — « Beerus endormi », « gros plan visage », « explosion », « Goku flying in the sky », « a character eating » : résultats pertinents ; 1ʳᵉ requête 14,8 s (modèle à froid), puis < 1 s ; aperçu au survol, plans similaires | `a06-*` |
| Personnage | **OK** — « Beerus », 3 exemples ajoutés depuis la recherche du personnage → 57 scènes, toutes pertinentes en tête | `a07-*` |
| Export de clips | **OK** — clip de l'éditeur (5 s), panier de dérush de 2 clips (4 s) | `a08-1..3` |
| Export timeline, chaque logiciel | **OK** — Premiere (`.xml`), Resolve (`.fcpxml`), After Effects (`.jsx`), Vegas (`.txt`), Final Cut (`.fcpxml`), Autre ; chemins avec accents, espaces et crochets correctement encodés dans chaque format (URL `file://` échappée, `\u00e9` en JSX, chemin brut Vegas) | `a08-4-*` |
| Synchro musique (Double Drift, Kevin MacLeod, libre) | **OK** — analyse 27,7 s (108 BPM, 3 drops, 8 passages), montage de 180 plans en 0,9 s, export FCPXML (447 marqueurs) | `a09-*` |
| Finition upscale | **OK** | `a10-dialog-1` |
| Finition ralenti (RIFE 4.26) | **OK** (8 coupes gardées nettes) — mais le modèle n'est pas téléchargeable depuis le dialogue (message « Download it from Settings > Models »), contrairement à l'upscale et à la profondeur | `a10-dialog-2`, `a10e-*` |
| Finition profondeur | **ÉCHEC** « could not write depth frame 0 » (bug 3) → **OK après correctif** (23 images PNG 16 bits en 23 s) | `a10f-*`, `a10g-profondeur` |
| Éditeur de roto | **OK** — passe IA 255 s (téléchargement des modèles de masque compris, bridé), coup de pinceau 1,0 s, clic 0,66 s (« image corrigée en 585 ms, IA 402 ms »), propagation 9 s, export **ProRes 4444 + matte + PNG** 38 s, plan ajouté à la sélection. SAM 3 indisponible (accès Hugging Face) → SAM 2.1 large, comme sous Linux. Le plan trouvé par la requête était un carton-titre : la chaîne est validée, pas la qualité du détourage | `a11-*` |
| Fermeture propre | **OK, 2 fois** — fenêtre fermée → sortie en 2,3–3,5 s, code 0, **aucun `python.exe`, `uv.exe`, ffmpeg ni `AMV Tools.exe` restant** | — |

### Premier lancement : temps et tailles (connexion bridée à 8 Mo/s)

| Étape | Temps | Téléchargé / disque |
|---|---|---|
| Sidecar Python (uv) | 297 s | Python 62 Mo + paquets de base (dont torch CPU) |
| Moteur CUDA 13 | 9 min 03 de téléchargement (22 paquets : torch 1,8 Go, cuDNN 661 Mo, cuBLAS 414 Mo, onnxruntime-gpu 216 Mo…) | — |
| Visual C++ Redistributable (correctif) | quelques s | 24 Mo |
| Modèles de base au premier usage | ~10 min | SigLIP 2 so400m **4,37 Go** + wd-tagger 361 Mo |
| **Disque après installation** | | venv **6,9 Go** + cache uv **7,7 Go** + modèles 4,7 Go ≈ **19 Go** |

Le cache uv (7,7 Go) double le venv et n'est jamais purgé : 7,7 Go à gagner
après l'installation (non corrigé, à décider).

## 3. Temps d'indexation : Windows contre Linux, même RTX 3090

Bench du pipeline complet, sources de la branche (`tests/bench_pipeline.py
--season DBS01 --sections season`, sidecar neuf à chaque passage, mêmes
réglages que les mesures Linux de `docs/reports/coupes.md` § 8), même
carte bridée de la même façon (1500 MHz, 250 W). Windows : Python 3.13.18,
torch 2.11.0+**cu130** (le moteur que l'appli installe), pilote 621.01, 8
vCPU, Windows Defender actif. Linux (debian-nas) : Python 3.13.17, torch
2.11.0+cu126, pilote 580, 6 vCPU.

| | Linux | Windows | Écart |
|---|---|---|---|
| Épisode sans réseau de coupes (399 plans) | **71 s** (série de 10 : 70,6–79,1) | **119 s** (médiane de 5 passages chauds : 112,5 / 119,3 / 122,4 / 118,3 / 133,8) | **+68 %** |
| dont décodage + coupes + mouvement | 60,4 s | 76 s (75,7–83,1) | +26 % |
| dont proxys de survol (399 encodages NVENC) | 5,2 s | **30–44 s** | **×6 à ×8** |
| dont tags + embeddings | 4,6 s | 4,6 s | = |
| Épisode avec réseau (497 plans) | ~76 s | **135 s** (131,9 / 135,2 / 145,1) | +78 % |
| 1ᵉʳ passage à froid (cache disque vide) | 108 s | 212 s | |

Le tout premier passage (212 s) inclut le chargement à froid des poids
SigLIP (4,4 Go) : 50 s, bloqué 42 s sur les 4 premiers tenseurs.

**Ce qui n'explique pas l'écart** : Windows Defender — passage refait avec
exclusions (dossiers de l'appli, cache des modèles, `python.exe`, ffmpeg) :
140,3 / 132,6 / 134,4 s, pas mieux (exclusions retirées ensuite) ; le
bridage (identique) ; le GPU (souvent sous-chargé : SM qui redescend à
450–945 MHz en P3, mémoire à 5 GHz). **Piste principale** : les proxys
lancent un processus ffmpeg + une session NVENC par plan ; sous Windows,
créer un processus et ouvrir une session NVENC (WDDM) coûte bien plus cher
(~90 ms par plan contre ~13 ms). Un encodeur de proxys qui garde un seul
ffmpeg pour tout l'épisode gagnerait ~30 s par épisode sous Windows. Non
fait (mesure à approfondir sur la VM).

**Dans l'appli** (profil « Précis » par défaut : sous-segmentation active,
proxys désactivés), le même épisode prend **360 s** (coupes 101 s,
sous-segmentation 80 s, extraction des 539 images clés 78 s, chargement de
l'index de recherche 56 s). Pas de mesure Linux à réglages identiques :
non comparé.

## 4. Coupes sous Windows : surcoût et précision

- **Surcoût du réseau TransNetV2**, paires alternées sans / avec réseau :
  122,4 → 135,2 s (+10,5 %), 118,3 → 131,9 s (+11,6 %), 133,8 → 145,1 s
  (+8,5 %) : **+10,2 % en moyenne** (Linux : +4,8 %). Passe de décodage :
  +6 à +11 s (Linux +2,7 s) ; le reste vient des 98 plans de plus (497 au
  lieu de 399, comme sous Linux) que les proxys, lents sous Windows, font
  payer davantage. Au-dessus de l'objectif de 5 % sous Windows.
- **Précision : identique à Linux.** `tests/eval/cuts_eval.py` sur les 7
  extraits annotés, réseau actif : listes de coupes ratées et de fausses
  coupes **identiques image par image** à celles du NAS sur les **7/7
  extraits**, chemin NVDEC `nvc` comme chemin ffmpeg `sw` (rappel 77,8 à
  97,6 %, précision 88,1 à 100 %, mêmes valeurs).
- Piège rencontré : sans les poids TransNetV2 (optionnels, 30 Mo),
  `AMV_CUTNET=1` ne fait rien — la première série « avec réseau » donnait
  399 plans ; poids copiés dans le cache, série refaite.

## 5. Bugs trouvés et corrigés (branche `jarvis/windows-test`)

| # | Bug (Windows réel) | Correctif | Vérifié |
|---|---|---|---|
| 1 | **Windows neuf : l'installation CUDA échoue** (« Échec de l'installation », `c10.dll` WinError 126) : PyTorch a besoin de `msvcp140.dll` (Visual C++ Redistributable), absent de Windows et du Python d'uv | `backend/winruntime.py` : après `uv sync`, vérifie les DLL du runtime ; s'il manque, télécharge le `vc_redist.x64` officiel (aka.ms) et le lance via l'invite d'autorisation Windows ; refus / hors ligne → message clair avec le lien | **En réel** (VM : runtime installé, CUDA validé) + tests |
| 1b | Message d'erreur illisible sur Windows français (« spÃ©cifiÃ© ») : sortie UTF-8 décodée en cp1252 | décodage UTF-8 explicite | **En réel** + test |
| 2 | **« GPU indisponible — repli CPU » juste après l'installation** : l'autotune démarrait 20 s après le lancement, pendant l'écran d'accueil, et importait torch (ici cassé faute de VC++, ailleurs la version CPU que l'installation remplace) ; le torch à moitié chargé restait dans le sidecar jusqu'au redémarrage | l'autotune attend la fin de la configuration initiale ; l'installation le démarre ensuite | **En réel** (premier lancement rejoué : CUDA actif sans redémarrage) + tests |
| 3 | **Profondeur : « could not write depth frame 0 »** dès que le dossier d'export contient un accent : `cv2.imwrite` n'ouvre pas les chemins non ASCII sous Windows | PNG encodé par `cv2.imencode`, écrit par Python | **En réel** + test |
| 4 | Un dossier `amv-autotune-*` (63 Mo) laissé dans `%TEMP%` à chaque sidecar arrêté pendant l'autotune (13 après les benchs) | les dossiers de plus d'une heure sont supprimés au lancement suivant ; nettoyage tolérant aux fichiers verrouillés | Tests seulement (VM déjà rendue) |
| 5 | Outils : `tests/bench_pipeline.py` inutilisable sous Windows (`/proc`, `ps`, `os.killpg`, `os.sysconf`) ; `tests/eval/cuts_eval.py` exige `ffprobe` (absent) | Toolhelp/GetSystemTimes/GlobalMemoryStatusEx en ctypes, arrêt par `/api/shutdown` + `taskkill /T` ; repli `ffmpeg -c copy -f framecrc` (mêmes PTS que ffprobe) | **En réel** (tous les benchs et l'évaluation ci-dessus) |

Les correctifs 1 à 3 ont été vérifiés en copiant les fichiers Python
corrigés dans l'appli installée (seule différence avec la beta.2 sur ces
fichiers). Tests : `tests/test_windows_runtime.py` (16 tests).
`uv run pytest -m "not slow"` et front (tsc, vitest, build) verts sur le
VPS.

## 6. Ce qui reste / constats non corrigés

- **Proxys de survol 6 à 8× plus lents sous Windows** (§ 3) : le plus gros
  écart avec Linux ; piste : un seul ffmpeg par épisode.
- Décodage + coupes +26 % et réseau de coupes +10 % sous Windows (objectif
  5 % tenu sous Linux et Mac, pas sous Windows).
- L'autotune ne cède pas la place à une recherche : après une relance, la
  première recherche a attendu > 90 s pendant que l'autotune occupait le
  GPU (14,8 s à froid sans autotune). Pas propre à Windows.
- Ralenti : pas de bouton « Télécharger » dans le dialogue quand RIFE
  manque (l'upscale et la profondeur l'ont).
- Cache uv de 7,7 Go jamais purgé après l'installation.
- Nom des clips exportés : le nom de l'épisode apparaît deux fois
  (`…B3FEF9B7_SubsPlease_…B3FEF9B7_scene_468.mp4`), à vérifier (sans doute
  aussi sous Linux).
- Hypothèse écartée : une DLL torch verrouillée pendant la réinstallation
  CUDA (testé : uv la remplace sans erreur).
- Non testé : glisser-déposer réel, mise à jour de l'appli, désinstallation,
  build Windows de la branche (seuls les fichiers Python corrigés ont été
  essayés dans l'appli installée), un 2ᵉ épisode, le Black Clover HEVC.

## 7. Relancer la VM

La VM 501 est arrêtée, `onboot 0`, et garde tout (Windows activé en
évaluation jusqu'à fin janvier 2027, pilote NVIDIA, bridage au démarrage,
AMV Tools installé et indexé, sources dans `C:\amv\src` avec leur venv,
épisode et extraits annotés dans `C:\amv`). Les ISO Windows et virtio
restent dans `local:iso` ; l'ISO de réponse a été supprimée.

1. `qm shutdown 401 --timeout 300` (la carte ne peut servir qu'à une VM ;
   la RAM de l'hôte ne suffit pas aux deux), vérifier `qm status 401`.
2. `qm start 501`. Le bridage 250 W / 1500 MHz s'applique tout seul au
   démarrage (vérifier `C:\amv\gpu-cap.log` avant toute charge).
3. Adresse : `qm guest cmd 501 network-get-interfaces` (DHCP). SSH :
   `ssh amv@<adresse>` avec la clé du VPS (mot de passe : `~/.hermes/amv-win.secret`).
   Téléchargements depuis la VM : `scp -O` (le mode SFTP du serveur SSH
   Windows tronque les fichiers à 200 Ko).
4. Appli pilotée : `C:\amv\in\driver-start.ps1` (pilote Playwright dans la
   session interactive), étapes envoyées dans `C:\amv\driver\cmd` ; bench :
   `C:\amv\in\run-task.ps1 -Name Bench -Cmd "powershell -File
   C:\amv\in\bench.ps1 0-a 1-a"`.
5. Fin : `qm shutdown 501 --timeout 180` puis `qm start 401`, et vérifier
   `gpu-clock-cap`, `nvidia-smi`, les conteneurs et Nextcloud.

Réglage propre au test laissé dans la VM : l'élévation des administrateurs
ne demande plus de confirmation (`ConsentPromptBehaviorAdmin = 0`), pour
que l'installation du runtime VC++ passe sans clic pendant le test pilotée
à distance.

## 8. Corrections (2 novembre 2026, après-midi)

Mesures Linux sur debian-nas (RTX 3090 bridée 250 W / 1500 MHz) le matin,
puis **une seule coupure** de debian-nas l'après-midi (consigne de 15 h 40,
au lieu de la fenêtre de nuit) pour la VM Windows, même carte, même
bridage. Bench : `tests/bench_pipeline.py --season DBS01 --sections season`,
sidecar neuf à chaque passage, paires alternées sans / avec réseau.
Appli testée = installeur **GitHub Actions** de la branche (run
41463521345, commit `5b1afb9`, SHA-256 vérifié dans la VM), installation
propre (`/S`, dossier de données neuf), puis le build Windows
(`electron-builder --win dir` sur le VPS, NSIS impossible sans Wine)
des commits suivants copié par-dessus pour vérifier leurs correctifs.

### Coupure de debian-nas

| | Heure (Paris) |
|---|---|
| `qm shutdown 401` | 15:38:05 → arrêtée 15:38:37 |
| `qm start 501`, SSH, bridage vérifié avant toute charge (tâche de démarrage 15:39:26 : 250 W, 210–1500 MHz) | 15:38:48 → 15:42:09 |
| Restauration (`qm shutdown 501`, `qm start 401`) | `qm shutdown 501` 18:16:27 (arrêt propre), `qm start 401` 18:16:58 |
| debian-nas de retour, vérifications | SSH 18:17:35 ; vérifié 18:22:29 : `gpu-clock-cap` actif, 3090 à 250 W (P8 au repos), `hyperqwen-single-1` et `jellyfin` healthy, `nextcloud`, `nextcloud-db`, `nextcloud-redis` Up, Nextcloud répond 200. Coupure totale : 2 h 20 |

### Tableau avant / après

| # | Point | Cause racine | Correction | Windows avant → après | Linux avant → après |
|---|---|---|---|---|---|
| 1 | Proxys de survol lents | Pas un ffmpeg par plan (déjà un seul par épisode) : la passe **redécodait la copie 240p** (21 000 images, latence NVDEC par image : 4,4 s sous Linux, ~30 s sous Windows — NVDEC/NVENC y font le même travail qu'à Linux, étalé sur 4× plus de temps) puis réencodait | Le décodage d'indexation encode **les proxys eux-mêmes** (une session NVENC par épisode, IDR sur la 1ʳᵉ image de chaque plan, images gardées sur le GPU jusqu'à ce que la coupe ne puisse plus bouger) ; la passe ne fait plus que **découper sans réencoder** (1 ffmpeg en copie) après avoir vérifié chaque borne | proxys veille 30–44 s ; aujourd'hui, mêmes conditions : base 39,0 / 42,9 s → **2,4–4,6 s** (7 passages) | **5,1 → 0,7 s** (3 paires), 399/399, même taille, PSNR 30,0 contre 29,6 dB |
| 2 | Surcoût du réseau de coupes | Sous Windows, le fil qui redimensionne et analyse les images (attendu 57 % du temps par la boucle de décodage, py-spy) lançait ~200 petits noyaux GPU par lot de 32 images, bien plus chers sous WDDM ; le réseau en ajoutait | Windows : redimensionnement + images du réseau rejoués en **graphe CUDA** par tampon (mêmes noyaux, mêmes bits ; histogramme par `scatter_add` au lieu de `bincount`, qui synchronisait l'hôte à chaque lot), NVENC dans son propre fil | veille +10,2 % ; aujourd'hui **+5,9 %** (3 paires : +8,0 / +7,5 / +2,3 %) — objectif de 5 % approché, pas tenu sur toutes les paires ; passe seule (sonde) avec réseau : 92,4 → 80,8 / 81,6 s | +5,0 % → +6,0 % en relatif (75,3/71,7 → 69,4/65,5 s) : surcoût absolu inchangé (~3,5–3,9 s), dénominateur réduit de 6 s par le point 1 |
| 3 | Indexation Windows 1,7× Linux | **Plancher NVDEC identique** (décodage brut PyNvVideoCodec 53,8 s Windows contre 53,5 s Linux) ; l'écart est dans notre traitement autour (passe `nvc` 65,3 contre 55,7 s ; avec mouvement 70,6 contre 57,3 s) et dans les proxys | points 1 et 2 | mêmes conditions (hôte saturé) : base 181,4 s → **135,1 s** (130,4 / 135,1 / 139,8), **-25 %** | 71,2 → **65,5 s** |
| 4 | Cache uv jamais purgé | Jamais nettoyé ; mais uv **lie** le venv au cache (liens physiques, aussi sous Windows : `torch_cuda.dll` avait 3 noms) — les « 7,7 Go » comptaient en double 7,1 Go du venv ; seuls 0,94 Go étaient propres au cache. Et sous Windows la purge depuis le sidecar échoue (« Accès refusé » sur `c10.dll`, DLL chargée) | `uv cache prune --ci --force` après installation et après chaque mise à jour ; **sous Windows par Electron au démarrage suivant, avant Python** | cache 8,07 Go apparents (0,94 Go réels) → **0** ; 1,0 Go libéré ; empreinte réelle ≈ 12 Go (venv 7,2 + modèles 4,7 + Python 0,06), pas 19 | testé (vrai uv, venv lié, `uv sync --offline` après purge) |
| 5 | 1ʳᵉ recherche > 90 s pendant l'autotune | L'autotune ne cédait qu'à l'indexation | Middleware : toute requête utilisateur interrompt le bench, qui reprend là où il en était après 15 s de calme, fil en priorité basse ; arrêt à chaque lot | appli installée : 3 recherches **95 / 87 / 126 ms** pendant l'autotune (en pause), reprise et fin en 188 s | décodage / réseau : 0,1–0,2 s ; étape des lots : 3,2–5,6 s = rechargement de SigLIP (déchargé après 30 s), pas l'autotune |
| 6 | Ralenti sans bouton « Télécharger » | Le dialogue vérifiait RIFE 4.25 en dur, le profil fait tourner RIFE 4.26 | `capabilities.defaults` : le dialogue suit le modèle du profil | appli : bouton présent, RIFE 4.26 téléchargé en 14,5 s, installé | tests |
| 7 | Nom d'épisode en double | `{anime}` et `{episode}` recevaient tous deux le nom complet | titre / numéro d'épisode extraits du nom | appli : `Dragon_Ball_Super_-_Beerus_01_scene_10.mp4` | tests |
| 8 | Fuite `%TEMP%` de l'autotune | (corrigé la veille) | — | en vrai : 2 dossiers laissés par des sidecars de bench arrêtés en plein autotune ; l'un vieilli de 2 h a été **supprimé** au démarrage suivant, les récents (< 1 h) gardés comme prévu | — |

**Précision des coupes : identique.** `tests/eval/cuts_eval.py`, 7 extraits,
réseau actif : Windows avec tout le nouveau code (graphe CUDA compris),
chemins `nvc` et `sw` : **14/14 identiques image par image** à la
référence Windows de la veille et à Linux ; Linux (nouveau code, proxys
directs) : 14/14 identiques au code de base. Variantes essayées pour
Windows (lots de 2 fenêtres, réseau synchrone, 4 tampons, latence basse du
décodeur) : 7/7 identiques sous Linux, sans gain mesurable sous Windows.

### Sondes Windows (épisode DBS 01, `tests/eval/decode_probe.py`)

| | Windows | Linux |
|---|---|---|
| `ffmpeg -hwaccel cuda -f null` | 58,6 s | 54,4 s |
| PyNvVideoCodec seul (plancher NVDEC) | **53,8 s** | 53,5 s |
| passe `nvc` (coupes, médianes, copie proxys) | 65,3 s | 55,7 s |
| + mouvement (la passe de l'appli) | 70,6 s | 57,3 s |
| + réseau de coupes | 91–106 s → **80,8 / 81,6 s** (graphe CUDA) | 58,5 s |
| onepass (ffmpeg) | 150,6 s | 67,7 s |

py-spy (Windows, réseau actif) : le fil principal attend le fil de
traitement 57 % du temps ; ce fil passe ~27 % dans les lancements de
noyaux de redimensionnement, ~20 % dans les appels NVENC image par image,
~23 % dans le détecteur. GPU surtout en P3 (~1 070 MHz en moyenne) : charge
légère, goulot côté CPU/lancements.

### Parcours de l'appli installée (installeur CI)

| Étape | Résultat |
|---|---|
| Installation silencieuse | OK, 55 s, 330 Mo |
| Premier lancement (Python + paquets de base) | OK, 504 s (297 s la veille : réseau et hôte plus chargés) |
| Détection matérielle | OK — RTX 3090, CUDA 13 recommandé |
| Moteur CUDA 13 | OK, 11 min 08 jusqu'à l'appli |
| Purge du cache uv | sidecar : échec partiel (« Accès refusé », corrigé) ; build corrigé : purge par Electron au démarrage, cache 0 Go |
| Indexation DBS 01 (profil Précis, proxys activés) | OK, 816 s, 539 plans (sans réseau à cause du défaut d'autotune, corrigé ; hôte saturé) |
| Recherche pendant l'autotune | 95 / 87 / 126 ms, autotune en pause puis terminé |
| Ralenti, modèle manquant | bouton « Télécharger » présent, RIFE 4.26 en 14,5 s |
| Export de clip | `Dragon_Ball_Super_-_Beerus_01_scene_10.mp4` |
| Mise à jour avec cache purgé | resynchronisation au lancement, sidecar prêt en 16,5 s, CUDA |
| Fermeture | propre, 1–3 s |

Non refait aujourd'hui (faute de temps dans la fenêtre) : personnage,
dérushage, export timeline, synchro musique, roto — inchangés par ces
corrections et validés la veille.

### Trouvé en vrai pendant ces tests (corrigé)

- **Autotune après l'installation du moteur** : il tournait dans le sidecar
  qui avait encore torch CPU chargé, mais l'empreinte nommait torch CUDA :
  réseau de coupes mesuré sur CPU (+185 %) et rangé « manual » pour de bon
  → l'appli indexait sans réseau (399 plans au lieu de 497). L'autotune
  attend maintenant le lancement suivant quand le torch chargé n'est pas
  celui installé (`a98a984`). Relancé dans la VM : réseau « auto », 0 %.
- Purge du cache refusée par Windows (ci-dessus, `846b740`).
- `_ProxyWriter` sans dossier de proxys (proxys désactivés, profil
  Précis) : la passe NVDEC échouait (attrapé sur Linux avant commit).

### Ce qui reste

- **Proxys avec sous-segmentation** (profil Précis + proxys activés) : les
  bornes des sous-plans viennent après le décodage, le mode direct ne
  s'applique pas ; l'ancien chemin coûte toujours cher sous Windows
  (appli : 520 proxys en une passe 121,7 s + 19 un par un 79,3 s, hôte
  chargé). Les proxys sont désactivés par défaut dans ce profil.
- Mac : chemin des proxys inchangé (VideoToolbox + libx264), non testé ici
  (le Mac est testé par une autre session).
- Linux après le graphe CUDA / fil NVENC : non remesuré (activés seulement
  sous Windows, carte rendue à debian-nas ensuite).
- Bruit : l'hôte Proxmox était à 56/62 Go de RAM (4,3 Go en swap, KSM
  actif) pendant la coupure ; les passages Windows varient de ±15 %, le
  premier de chaque série est toujours lent ; d'où les paires alternées.
