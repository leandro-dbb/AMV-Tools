# v06 — Chantier A : « ça marche du premier coup chez eux » (Windows + export timeline)

Branche `jarvis/v06-windows-timeline`. Session reprise à 23 h 45 après ~30 min
d'une session DeepSeek : son travail a été relu, gardé quand il était juste,
corrigé sinon (détails en fin de rapport).

**Dernier run Windows vert : run n° 40422827708 (9ᵉ run du workflow),
commit `e7efa77`** —
https://github.com/leandro-dbb/AMV-Tools/actions/runs/40422827708
(runs verts précédents : 40419245822, 40421736839, 40422282429). Les commits
suivants ne touchent que la documentation (le workflow ignore les commits
docs).

Chiffres du run vert (VM GitHub `windows-latest`, CPU, sans GPU) : premier
démarrage de l'app installée 16 s, détection GPU « cpu », indexation de deux
clips (téléchargement des modèles compris) 61 s, recherche 6 résultats,
clip exporté, 6 timelines écrites, fermeture en 1 s sans processus restant.

## En bref

- Un **vrai Windows** (machine virtuelle GitHub `windows-latest`) construit
  l'installeur comme la release, l'installe en silencieux, lance l'application
  installée et fait tout le parcours : installation du moteur CPU, indexation,
  recherche, export d'un clip, export timeline dans les 6 formats, coupure
  réseau simulée, fermeture et vérification qu'aucun processus ne traîne.
- Ce test a trouvé **5 vrais problèmes Windows/premier lancement**, tous
  corrigés (dont un bloquant : l'export de clip plantait sur tout PC sans
  carte NVIDIA, et un très visible : 16 min d'indexation « figée » à 0 %
  pendant le téléchargement des modèles).
- L'export timeline se choisit maintenant **par logiciel** (Premiere Pro,
  DaVinci Resolve, After Effects, Vegas Pro, Final Cut Pro, Autre), avec la
  marche à suivre pour l'import. Deux nouveaux exports : **script After
  Effects** et **EDL texte Vegas** (le vrai format que Vegas sait importer).
- Chaque format est relu par des lecteurs qui ne sont pas les nôtres puis
  **rendu image par image par MLT** (le moteur de Kdenlive/Shotcut) : chaque
  clip tombe sur la bonne image source, sur des clips de synthèse **et sur
  deux vrais extraits d'épisodes** (DBS Beerus en H.264, Black Clover en HEVC).
- Ce qui reste à faire par Léandro : installer sur son PC, et ouvrir une
  timeline dans Premiere, Resolve, After Effects et Vegas avec les
  checklists de [docs/import-timeline.md](../import-timeline.md).

## 1. Windows sur une vraie machine

### Le test automatique

Fichiers : `.github/workflows/windows-smoke.yml` (déclenché à la main ou à
chaque push sur une branche `jarvis/**`) et `.github/scripts/windows-smoke.ps1`.

Ce qu'il fait, dans l'ordre :

1. construit l'installeur NSIS exactement comme la release (non signé) et
   l'installe avec `/S` ;
2. lance **l'application installée** (pas un backend lancé à part) avec un
   dossier de données dont le chemin contient des espaces et des accents
   (`…\Données utilisateur Jérôme`) — c'est là que vivent le Python, l'env
   virtuel, la base SQLite et les logs ;
3. attend que le sidecar réponde sur `/health` ;
4. interroge la détection GPU d'avant installation (doit conseiller « CPU »
   sur cette VM sans GPU) ;
5. installe le moteur CPU par le même appel que l'écran d'accueil ;
6. recharge la fenêtre et prend des captures d'écran (accueil, interface) ;
7. génère deux clips de test avec **le ffmpeg fourni par l'application**
   (aucun ffmpeg système dans le PATH de l'app), noms avec tiret long,
   accents et apostrophe, 23,976 i/s avec son et 25 i/s sans son ;
8. les indexe sur CPU (vrai téléchargement des modèles SigLIP 2 et
   wd-tagger), fait une recherche texte, exporte un clip, puis une timeline
   dans chaque format vers un dossier nommé `exports é 日本` ;
9. vérifie dans l'environnement Python de l'app : demande CUDA ou DirectML
   sans GPU → repli CPU propre ; téléchargement de modèle sans internet →
   message clair (« Check your internet connection… ») ;
10. ferme l'app par le chemin normal et vérifie qu'aucun processus
    (Python, uv, ffmpeg) ne reste ; vérifie que le log du backend ne contient
    ni traceback ni caractère cassé.

Artefacts de chaque run : captures d'écran, logs Electron et backend,
fichiers de timeline produits, `summary.json` (temps de premier démarrage,
temps d'indexation, nombre de résultats…).

Pour que l'app installée soit pilotable, l'app accepte des réglages de test
**inertes pour les utilisateurs** (variables d'environnement) : port et jeton
fixes (jeton ignoré s'il fait moins de 32 caractères), dossier de commandes
(`reload`, `shot-…`, `quit` — seuls ces noms de fichiers sont lus et
supprimés). Une route `/health` (protégée par le jeton comme le reste) a été
ajoutée.

### Bugs Windows trouvés par ce test (tous corrigés)

| # | Symptôme sur Windows | Cause | Correction |
|---|---|---|---|
| 1 | **Export de clip : erreur 500** sur tout PC sans NVIDIA (« Cannot load nvcuda.dll ») | le ffmpeg Windows fourni contient NVENC/CUDA, l'app concluait « encodeur matériel disponible » sans regarder s'il y a une carte | NVENC exige une carte NVIDIA détectée (`nvidia-smi -L`) ; si l'encodage matériel échoue quand même (pilote trop vieux…), repli automatique sur libx264 |
| 2 | Découpage des scènes et réglage auto tentaient le décodage CUDA sur PC sans NVIDIA (erreurs puis repli lent) | même cause, côté décodage (ffmpeg et PyAV) | même garde « carte NVIDIA présente » |
| 3 | Lignes de log affichées avec `�` à la place des accents et tirets (écran de démarrage) | la sortie de Python passait en cp1252 | `PYTHONIOENCODING=utf-8` pour le sidecar (vérifié : plus aucun `�` dans les logs du run) |
| 4 | L'écran d'accueil conseillait **toujours CUDA 13**, même sur un PC AMD ou sans GPU | choix codé en dur | sonde matérielle sans torch (`nvidia-smi`, cartes vidéo Windows) : CUDA 13, CUDA 12.6 (carte avant RTX 20xx ou pilote < 580), DirectML (AMD/Intel), MPS (Mac), CPU — présélectionnée avec la ligne « Détecté sur ce PC : … » |

Corrigés aussi en passant (repérés en lisant le code pendant l'enquête) :
la sortie d'erreur de ffmpeg pouvait faire planter l'export avec un nom
d'épisode japonais (décodage cp1252) ; les timelines étaient écrites avec des
fins de ligne différentes selon l'OS.

Points vérifiés qui **marchaient déjà** : chemins avec espaces/accents dans le
dossier de données (env virtuel, Python, base), sources et dossier de sortie
avec accents, apostrophes et japonais, ffmpeg fourni trouvé sans ffmpeg
système, fermeture propre sans processus orphelin, premier démarrage rapide.

### GPU

- Le runner n'a pas de GPU (carte « Microsoft Hyper-V Video ») : la sonde
  conseille bien CPU, l'app tourne en CPU sans erreur, une demande CUDA ou
  DirectML retombe sur CPU (« CPU (forced fallback) ») sans planter.
- La logique de choix NVIDIA/AMD/Intel est testée par tests unitaires
  (`tests/test_gpu_probe.py`). **Pas testé sur un vrai PC NVIDIA/AMD
  Windows** : à faire par Léandro (voir plus bas).
- Non fait : installer le moteur DirectML dans la VM (ce serait possible via
  le rendu logiciel WARP de Windows, mais long et peu représentatif).

### Premier lancement : téléchargement des modèles

- Message clair sans internet : **vérifié sur Windows** (étape 9).
- Reprise après coupure : `huggingface_hub` reprend les téléchargements
  interrompus (requêtes partielles), la reprise est conservée entre les
  tentatives — couvert par `tests/test_model_download.py` (simulé, pas de
  coupure réelle).
- Taille annoncée : affichée dans Réglages > Modèles (`size_mb` par modèle).
- **Progression : bug trouvé et corrigé.** Au run 6, un jour de lenteur
  côté Hugging Face, la première indexation est restée **16 minutes** sur
  « Indexation 0 % » : SigLIP 2 et wd-tagger (~1,9 Go) se téléchargeaient en
  silence au milieu du premier lot. Désormais l'indexation commence par
  télécharger les modèles manquants (mêmes garanties : reprises, contrôle de
  taille/empreinte, message clair hors-ligne) et la barre d'état affiche
  « SigLIP 2 base (NaFlex) ↓ 512 / 1500 MB » avec le pourcentage (taille
  mesurée chaque seconde dans le cache). Aucun changement d'interface
  nécessaire. Le test Windows exige maintenant ce pré-téléchargement sur la
  machine neuve (`tests/test_model_prefetch.py` pour la logique).

## 2. Export timeline

### Choix par logiciel

La fenêtre d'export demande « Logiciel de montage » et affiche sous le choix
la marche à suivre (FR/EN). Le dernier logiciel choisi est retenu.

| Choix | Fichier | Pourquoi ce format |
|---|---|---|
| Premiere Pro | FCP 7 XML `.xml` | format d'échange natif de Premiere ; chemins écrits comme Premiere les écrit (`file://localhost/C%3a/…`) |
| DaVinci Resolve | FCPXML 1.9 | import Timeline natif, cadences mélangées gérées |
| After Effects | script `.jsx` | AE n'importe aucune timeline : le script crée la composition |
| Vegas Pro | EDL texte Vegas `.txt` | le seul import de timeline présent dans toutes les versions de Vegas |
| Final Cut Pro | FCPXML 1.9 | natif |
| Autre | OTIO ou EDL CMX3600 (ou les autres) | Kdenlive, Avid, outils |

**Vegas — choix et justification.** DeepSeek avait mis du CMX3600 renommé en
`.txt` : Vegas ne lit pas ce format. `Fichier > Importer > Fichier texte EDL`
lit la liste d'événements propre à Vegas (en-tête de 28 champs entre
guillemets, point-virgule, temps en millisecondes), format confirmé par un
export Vegas publié sur Creative Cow et par la documentation Vegas (Boris FX).
On l'a préféré à un script `.cs` Vegas, dont l'API change de nom selon les
versions. Vegas 14+ peut aussi tenter le FCP 7 XML (plan B documenté).

**After Effects.** Script ExtendScript ES3 (pas de `let`/`const`/fonctions
fléchées), 100 % ASCII (accents et japonais échappés), un seul « annuler » ;
importe chaque source une fois, crée la composition à la cadence et taille de
la timeline, place chaque clip (startTime/inPoint/outPoint exacts en
fractions), marqueurs, alerte listant les fichiers manquants, et cherche les
fichiers à côté du script si le dossier a bougé.

### Validation par format

| Format | Validation stricte | Relu par | Rendu MLT, synthèse (23,976/25/29,97, clip de 3 images) | Rendu MLT, vrais épisodes | Produit sur Windows |
|---|---|---|---|---|---|
| FCP 7 XML (Premiere) | structure, timebase/ntsc, in/out, pathurl | OpenTimelineIO `fcp_xml` | ✅ image exacte | ✅ | ✅ |
| FCPXML 1.9 (Resolve/FCP) | **DTD officielle Apple 1.9** (xmllint, empreinte sha256 épinglée) ; spine contiguë | lecteur conforme à la spec | ✅ image exacte | ✅ | ✅ + DTD OK |
| EDL CMX3600 | colonnes, bobines ≤ 8 car., FROM CLIP NAME, durées source = enregistrement | OpenTimelineIO `cmx_3600` | ✅ (±1 image entre cadences : limite du format) | ✅ | ✅ |
| OTIO | round-trip, pistes de même longueur, adaptateur xges | OpenTimelineIO | ✅ image exacte | ✅ | ✅ |
| EDL texte Vegas | 28 champs, CRLF, temps exacts en ms | lecteur conforme à la spec | ✅ image exacte | ✅ | ✅ (CRLF simple vérifié) |
| Script After Effects | ES3, ASCII, `node --check` | **exécuté** dans un After Effects simulé (Node) | ✅ image exacte | ✅ | ✅ syntaxe OK |

« Rendu MLT » : la liste de montage relue est donnée à `melt` (moteur de
Kdenlive et Shotcut), qui ouvre les vrais fichiers (noms avec espaces,
accents, apostrophes, japonais) et rend la timeline ; la première et la
dernière image de chaque clip sont comparées aux images de la source décodée.
Tests : `tests/test_timeline_validation.py`, `tests/test_timeline_nle_render.py`
(le test sur vrais épisodes se lance avec
`AMV_NLE_MEDIA=extrait1.mkv:extrait2.mkv`). Kdenlive lui-même n'a pas été
installé (ouverture d'OTIO uniquement via son interface graphique).

### Bugs timeline trouvés et corrigés

- **OTIO, cadences mélangées** : la modification DeepSeek réécrivait le point
  d'entrée à la cadence de la séquence → un clip 23,976 dans une séquence 25
  tombait une image trop tard au rendu MLT. Point d'entrée remis à la cadence
  source (exact), durée à la cadence séquence (placement exact).
- **EDL, noms et chemins longs coupés** : les lignes `* FROM CLIP NAME:` /
  `* FROM FILE:` étaient repliées à 80 colonnes ; les logiciels n'en lisent
  qu'une ligne, donc relinkaient avec un nom tronqué. Elles restent sur une
  ligne (comme dans les exports de Premiere/Resolve).
- **EDL, clips très courts à une autre cadence** : durée source ≠ durée
  enregistrement (lu comme un changement de vitesse) ou événement de 0 image.
  La sortie source = entrée + durée enregistrée (≥ 1 image).
- **FCP 7** : chemins au format Premiere (`file://localhost/C%3a/…`).
- **FCPXML** : marqueurs avec une balise `<note>` interdite par la DTD Apple
  (corrigé par DeepSeek, confirmé par la DTD).

## Ce que DeepSeek avait fait, et ce qu'il en reste

- Gardé : workflow Windows (réécrit en une seule étape : le runner tuait
  l'application entre deux étapes, cause de son run en échec), route
  `/health`, capture d'écran de la fenêtre, script After Effects (complété),
  correctifs FCPXML marqueur / FCP 7 / OTIO piste audio, choix par logiciel
  dans l'UI (textes d'aide Vegas et « Autre » corrigés), tests de validation.
- Corrigé : export Vegas (mauvais format), modification OTIO (décalage d'une
  image), correction EDL incomplète, vérifications du workflow (champs d'API
  et chemins de venv faux), DTD téléchargée sans vérification d'empreinte,
  surveillance d'un fichier « quit » remplacée par un dossier de commandes
  qui ne supprime que ses propres fichiers, jeton de test trop permissif.

## Fichiers partagés modifiés (hors périmètre A)

- `app/src/components/OnboardingScreen.tsx`, `app/src/api/client.ts`,
  `app/src/i18n/strings.onboarding.ts` : présélection du moteur détecté
  (ajouts uniquement).
- `backend/api/setup.py` : route `GET /api/setup/hardware` (ajout).
- `backend/indexing/cuts.py`, `backend/export/ffmpeg.py` : gardes NVIDIA et
  repli libx264.
- `backend/state.py` : 3 lignes pour lancer le pré-téléchargement des modèles
  au début de l'indexation (nouveau module `backend/model_prefetch.py`).
- `backend/server.py` : route `/health` (DeepSeek, version codée en dur retirée).
- `app/src/components/TimelineExportDialog.tsx`, `strings.timeline.ts`,
  `types.ts` : choix par logiciel.
- `README.md`, `CHANGELOG.md`, `pyproject.toml`/`uv.lock` (dépendance de dev
  `opentimelineio-plugins`, Apache-2.0, ajoutée par DeepSeek pour les
  adaptateurs FCP 7 / EDL).

## Tests

- `uv run pytest -m "not slow"` : **380 passés**, 7 ignorés (tests opt-in
  sur vrais épisodes et matériel réel). Lors d'un passage précédent,
  `test_autotune::test_time_parallel_finds_the_engine_width` avait échoué
  une fois : mesure de temps faussée par la charge du VPS (~20, sessions
  parallèles), sans lien avec ce chantier (passe seul 3/3, et dans le
  passage final).
- Tests de rendu MLT sur vrais extraits (DBS Beerus 01 en H.264, Black
  Clover S02E04 en HEVC, déjà présents dans les données de test hors dépôt, aucun
  nouveau transfert depuis le NAS) : 6/6 formats OK.
- `npx tsc --noEmit`, `npx vitest run` (28 passés), `npm run build` : OK.

## Ce que Léandro doit tester lui-même

**Sur son PC Windows (15 min)**
1. Télécharger l'installeur depuis les artefacts d'un run `build` (ou lancer
   le workflow build), l'installer.
2. Premier lancement : vérifier que l'écran d'accueil affiche « Détecté sur
   ce PC : NVIDIA GeForce RTX … » et présélectionne CUDA 13 (ou 12.6 si
   pilote < 580).
3. Indexer un épisode, exporter un clip (doit utiliser NVENC : log
   « h264_nvenc available »).
4. Si possible sur un PC sans NVIDIA (portable Intel/AMD) : DirectML proposé,
   export de clip qui fonctionne.
5. Fermer l'app : plus aucun `python.exe`/`uv.exe` dans le gestionnaire des
   tâches.

**Dans chaque logiciel** : suivre les checklists de 5 min de
[docs/import-timeline.md](../import-timeline.md) — Premiere Pro, DaVinci
Resolve, After Effects, Vegas Pro (jamais testé dans Vegas : vérifier en
particulier les chemins accentués et le séparateur décimal sur un Windows en
français).
