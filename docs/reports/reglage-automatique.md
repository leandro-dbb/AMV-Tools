# Réglage automatique selon la machine

Branche `jarvis/v04-autotune` (depuis `jarvis/v04-models4`), 25 octobre 2026.

## Ce qui est réglé

| Réglage | Avant (constante) | Mesure | Plateforme |
|---|---|---|---|
| `parallel_decodes` | 2 (`AMV_PARALLEL_DECODES`) | 4 décodages single-pass d'un clip de 30 s avec 1 / 2 / 4 en vol | NVIDIA |
| `chunk_workers` | `chunked.WORKERS` = 8 | 2 clips de 10 s par session VideoToolbox, 2 / 4 / 8 / 12 sessions | macOS |
| `tagger_engine` | Core ML | MPS contre Core ML sur 32 images, sortie comparée (écart ≤ 0,02, top-5 identique sur ≥ 95 % des images) | macOS |
| `batch_size` | palier mémoire (CUDA 8/16/32/64, MPS 8) | tagger + SigLIP sur 4× le plus grand lot, candidats ≤ palier | CUDA, MPS |
| TensorRT | — | détecté seulement : aucun chemin TensorRT compatible (SigLIP NaFlex le refuse) | NVIDIA |

Règles de choix (`backend/autotune/choose.py`) : pour un nombre de décodages,
le plus petit dont le plus rapide ne fait pas gagner 5 % ; pour un moteur ou un
lot, la valeur actuelle sauf si une autre gagne au moins 5 % (et, pour un
moteur, donne la même sortie). Sur le clip synthétique (bibliothèque vide), la
règle « référence » s'applique aussi aux décodages et la mesure est refaite au
lancement suivant dès qu'une vidéo est indexée : la 3090 y choisissait 1
décodage (clip trop facile à décoder) contre 2 sur un vrai épisode.

Empreinte (`fingerprint.py`) : GPU ou puce (+ mémoire sur Mac), pilote ou
version de macOS, torch, ffmpeg, version de l'app, version du schéma de mesure.
Cache : `<dossier de données>/autotune.json`, une entrée par empreinte (4
gardées). Le pipeline lit `autotune.value(nom, défaut)` : variables d'env
d'abord (`AMV_PARALLEL_DECODES`, `AMV_CHUNK_WORKERS` nouveau, `AMV_COREML=0`),
puis la mesure, puis l'ancienne constante. Une taille de lot modifiée dans
Settings reste prioritaire. `AMV_AUTOTUNE=0` coupe le lancement automatique.

Tâche de fond : 20 s après le démarrage du sidecar, seulement si aucune
indexation ni ré-embedding ne tourne ; si une indexation démarre pendant la
mesure, elle s'interrompt sans rien écrire et repassera au lancement suivant.

## RTX 3090

Bench seul, extrait de 10 s de `extrait-4min.mkv` (H.264 1080p), bouclé à 30 s :

| Passe | 1 décodage | 2 | 4 | Choix | Lot (8 / 16 / 32 / 64, ms/image) | Durée |
|---|---|---|---|---|---|---|
| 1 | 1,739 s | **1,510** | 1,656 | 2 | 8,31 / 7,12 / 6,86 / **6,61** → 64 | 45 s |
| 2 | 1,631 s | **1,360** | 1,571 | 2 | 7,71 / 7,29 / 6,99 / **6,75** → 64 | 40 s |
| synthétique | 1,306 s | 1,395 | — | 2 (référence) | → 64 | 40 s |

Deux réglages ont été écartés en route : 2 vagues de clips de 60 s (choix 2,
mais 113 s de mesure) et 1 vague de clips de 30 s (1,87 / 1,82 / 1,70 s : trop
bruité, choisissait 4). La charge fixe de 4 décodages par candidat, comme une
petite file, sépare nettement les candidats en 40 s.

`tests/bench_pipeline.py --quick extrait-4min.mkv`, paires alternées (avant =
`76b1011`, après = cette branche avec l'`autotune.json` mesuré) :

| Run (ordre) | Total | Indexation (season) | `pipeline_s` | Recherche | Export |
|---|---|---|---|---|---|
| avant 1 | 87,8 s | 21,7 s | 39,1 s | 8,1 s | 9,3 s |
| après 1 | 84,1 s | 21,3 s | 36,7 s | 6,3 s | 9,2 s |
| avant 2 | 83,6 s | 20,3 s | 36,7 s | 7,9 s | 8,5 s |
| après 2 | 84,8 s | 20,9 s | 36,9 s | 7,3 s | 8,7 s |
| **médiane avant / après** | 85,7 / 84,5 s | 21,0 / 21,1 s | 37,9 / 36,8 s | | |

Pas plus lent : l'écart est dans le bruit entre deux runs d'une même version.
`--quick` n'a qu'un fichier, donc il ne fait pas tourner les décodages
parallèles (la file n'en lance qu'à partir de 2 vidéos) ; le lot mesuré (64)
est celui d'avant. Le gain mesuré pour 2 décodages vient du bench lui-même.

Sidecar réel, dossier de données vide (bibliothèque vide → clip synthétique) :
le réglage démarre seul 20 s après le lancement, `GET /api/autotune` passe de
`idle` à `done` en 44 s (2 décodages gardés, lot 64, TensorRT « non
installé »), et `POST /api/autotune/run` relance bien la mesure. Le premier
essai de ce test, avec l'ancienne règle synthétique, avait choisi 1 décodage
(1,29 contre 1,40 s) : d'où le correctif qui garde la valeur par défaut sur le
clip synthétique.

Tests sur le NAS : `test_autotune`, `test_chunked_decode`, `test_cuda_perf`,
`test_index_queue`, `test_nvc_pass` verts (66 ; même jeu sans `test_autotune`
vert sur la base). `test_cuda_perf5.py`, présent dans la copie NAS mais absent
de cette branche et de sa base (reste d'une autre branche, rsync ne supprime
pas), échoue faute de l'API qu'il teste (`_ProxyWriter.slot_of`) : hors sujet.

Carte prise à 14 h 33, rendue à 14 h 50 (17 min).

## Mac

Le Mac ne répondait pas (« No route to host ») : rien de
validé sur Apple Silicon. Les chemins macOS (sessions VideoToolbox, MPS contre
Core ML pour le tagger, lot MPS) ne sont testés qu'en mock. À vérifier quand il
sera allumé : `uv run python -c "from backend import autotune; print(autotune.run())"`
doit retrouver ~8 sessions sur le M1 Pro.

## Limites

- Finition (upscale, RIFE, profondeur) : le choix MPS / Core ML n'est pas
  mesuré (il faut les modèles de finition et plusieurs minutes) ; les choix
  actuels restent.
- NVIDIA : TensorRT n'est que détecté, l'app n'a pas de chemin TensorRT pour
  les modèles par défaut.
- La mesure des décodages ne voit pas le reste du pipeline (tagging pendant le
  décodage) ; la règle des 5 % et la charge fixe limitent le risque.
- Premier lancement : ~40 s de GPU en tâche de fond, éventuellement pendant
  une recherche.

## Vérifications

- VPS : `uv run pytest -m "not slow"` → 287 passed, 1 skipped ;
  `tests/test_autotune.py` → 27 tests (règles de choix, empreinte, cache et
  invalidation, repli sur les constantes, priorité des variables d'env, tâche
  de fond, API, clip synthétique réel).
- `cd app && npx tsc --noEmit && npx vitest run && npm run build` → OK
  (26 tests vitest, dont la complétude FR/EN des chaînes).
- Sidecar local (CPU) : démarre, `/api/status` répond, le réglage tourne seul
  et enregistre « rien à régler » (plateforme `cpu`).
