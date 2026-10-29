# Ouvrir une timeline AMV Tools dans ton logiciel de montage

AMV Tools n'exporte **pas de vidéo** quand tu fais « Exporter la timeline » :
il écrit un petit fichier qui décrit le montage (quels clips, dans quel ordre,
avec quels points d'entrée/sortie) et qui **pointe vers tes épisodes d'origine**.
Rien n'est réencodé, la qualité reste celle des sources.

Dans la fenêtre d'export, choisis simplement ton logiciel :

| Logiciel choisi | Fichier produit | Comment l'ouvrir |
|---|---|---|
| Premiere Pro | `.xml` (FCP 7 XML) | Fichier > Importer… |
| DaVinci Resolve | `.fcpxml` (FCPXML 1.9) | Fichier > Importer > Timeline… |
| After Effects | `.jsx` (script) | Fichier > Scripts > Exécuter le fichier de script… |
| Vegas Pro | `.txt` (EDL texte Vegas) | Fichier > Importer > Fichier texte EDL (*.txt) |
| Final Cut Pro | `.fcpxml` | Fichier > Importer > XML… |
| Autre (OTIO/EDL) | `.otio` ou `.edl` | selon le logiciel (Kdenlive, Resolve, Avid…) |

> **Règle d'or** : ne déplace pas et ne renomme pas tes épisodes entre l'export
> et l'import. Si tu dois transporter le projet sur un autre PC, coche
> « Rassembler les médias » : les clips sont copiés à côté de la timeline.

---

## Premiere Pro

1. Fichier > Importer… (Ctrl+I), choisis le fichier `.xml`.
2. Premiere crée un chutier au nom de la timeline avec la séquence dedans.
   Double-clique la séquence.
3. Si Premiere demande « Lier le média », clique **Rechercher** et pointe le
   dossier de tes épisodes : il retrouve les fichiers par leur nom.

**Pourquoi FCP 7 XML et pas FCPXML ?** Premiere ne lit pas le FCPXML (le
format de Final Cut Pro X). Le FCP 7 XML est son format d'échange natif ; les
chemins y sont écrits exactement comme Premiere les écrit lui-même
(`file://localhost/C%3a/...`).

### Checklist Premiere (5 min)
- [ ] La séquence s'ouvre sans message « média hors ligne ».
- [ ] Le nombre de clips correspond à celui annoncé par AMV Tools.
- [ ] La cadence de la séquence (clic droit > Paramètres de séquence) est
      celle choisie (23,976 / 24 / 25 / 29,97).
- [ ] Le premier et le dernier clip commencent sur la bonne image (compare
      avec l'aperçu d'AMV Tools).
- [ ] Un épisode avec accents/apostrophe/espaces dans son nom est bien relié.
- [ ] Les marqueurs (si activés) apparaissent sur les clips.
- [ ] Le son est présent pour les épisodes qui en ont.

## DaVinci Resolve

1. Fichier > Importer > Timeline… (Ctrl+Maj+I), choisis le `.fcpxml`.
2. Dans la fenêtre qui s'ouvre, laisse cochés « Importer automatiquement les
   clips sources dans le pool de médias » et décoche « Utiliser les
   paramètres de taille du projet » seulement si ta résolution diffère.
3. Si des clips apparaissent en rouge (hors ligne) : clic droit sur la
   timeline > Relier les clips sélectionnés… et choisis le dossier des épisodes.

Resolve lit aussi le `.xml` (FCP 7) et le `.otio` (Resolve 18.5 et plus) :
utiles si le FCPXML pose un souci.

### Checklist Resolve (5 min)
- [ ] La timeline s'importe sans erreur, les clips ne sont pas rouges.
- [ ] La cadence de la timeline est la bonne (avant d'importer, la cadence du
      **projet** doit être réglée sur la même valeur : Resolve ne la change
      plus après le premier média).
- [ ] Ordre, nombre de clips et trous entre clips identiques à l'aperçu.
- [ ] Un clip très court (moins de 5 images) est présent.
- [ ] Mélange 23,976 / 25 / 29,97 : chaque clip commence sur la bonne image.

## After Effects

After Effects n'importe aucun fichier de timeline : AMV Tools écrit donc un
**script** qui construit la composition à ta place.

1. Fichier > Scripts > Exécuter le fichier de script…, choisis le `.jsx`.
   (Si After Effects refuse : Édition > Préférences > Scripts et expressions >
   coche « Autoriser les scripts à écrire des fichiers et à accéder au réseau ».)
2. Le script importe chaque épisode une seule fois, crée une composition à la
   cadence et à la taille de la timeline, et place chaque clip comme un calque
   avec son point d'entrée/sortie exact.
3. Ctrl+Z annule tout en une fois.

Si un épisode est introuvable, une alerte liste les fichiers manquants (le
reste de la composition est créé). Le script cherche aussi les fichiers à
côté de lui (dossier déplacé, médias rassemblés).

### Checklist After Effects (5 min)
- [ ] Le script s'exécute sans erreur et la composition s'ouvre.
- [ ] Paramètres de composition : cadence et taille attendues.
- [ ] Les calques se suivent dans le bon ordre, sans chevauchement.
- [ ] Premier/dernier calque : la bonne image au début et à la fin.
- [ ] Un nom d'épisode avec accents/japonais est correctement importé.

## Vegas Pro

Vegas Pro ne lit ni le FCP 7 XML ni le FCPXML de base. Son import fiable,
présent dans toutes les versions, est **Fichier > Importer > Fichier texte
EDL (*.txt)** : il lit le format de liste d'événements propre à Vegas (celui
que Vegas écrit lui-même avec Fichier > Exporter > Fichier texte EDL). C'est
ce qu'AMV Tools produit : une ligne par événement, temps en millisecondes,
vidéo sur la piste 1, son sur la piste 2, chemin complet de chaque épisode.

1. Fichier > Importer > Fichier texte EDL (*.txt), choisis le `.txt`.
2. Vegas ouvre un nouveau projet avec les événements placés.
3. Règle les propriétés du projet (cadence/taille) : Fichier > Propriétés >
   « Adapter aux paramètres du média ».

On a choisi l'EDL texte plutôt qu'un script `.cs` Vegas parce que l'API de
script change de nom selon les versions (Sony.Vegas / ScriptPortal.Vegas) et
que les scripts demandent une manipulation de plus. Les versions récentes de
Vegas (14 et plus) savent aussi importer un FCP 7 XML via « Final Cut Pro 7 /
DaVinci Resolve Timeline (*.xml) » : c'est un plan B si l'EDL texte pose
problème.

### Checklist Vegas (5 min) — **jamais testé dans Vegas pour l'instant**
- [ ] L'import est accepté (pas de message « format non reconnu »).
- [ ] Les événements vidéo sont sur la piste 1 et le son sur la piste 2.
- [ ] Positions et durées correctes (compare 2-3 clips avec l'aperçu).
- [ ] Le point d'entrée dans l'épisode est le bon (première image du clip).
- [ ] Un épisode avec accents dans le chemin est bien trouvé. Si ce n'est pas
      le cas, note-le : c'est l'encodage du fichier texte qu'il faudra changer
      (aujourd'hui UTF-8 sans BOM).
- [ ] Sur un Windows en français : les temps (avec un point décimal) sont bien
      lus. Si tout arrive à 0, c'est le séparateur décimal.

## Final Cut Pro (Mac)

1. Fichier > Importer > XML…, choisis le `.fcpxml`.
2. Final Cut crée un événement contenant le projet.

### Checklist Final Cut (5 min)
- [ ] Import sans message d'erreur de DTD.
- [ ] Projet à la bonne cadence, clips dans l'ordre, trous respectés.

## Autre (OTIO / EDL)

- **OTIO** (`.otio`, OpenTimelineIO) : Resolve 18.5+ (Fichier > Importer >
  Timeline), Kdenlive (import OpenTimelineIO), outils de pipeline.
- **EDL CMX3600** (`.edl`) : la liste de coupes universelle (Avid, Resolve,
  Premiere…). Limites du format : une seule cadence pour toute la timeline,
  noms de bobine limités à 8 caractères ; le nom complet du clip et le chemin
  du fichier sont dans les commentaires `* FROM CLIP NAME:` / `* FROM FILE:`,
  que Premiere et Resolve utilisent pour relier les médias.

---

## Ce qui a été vérifié automatiquement (sans ces logiciels)

- FCPXML validé contre la DTD officielle d'Apple (version 1.9).
- Chaque format est relu par un lecteur indépendant (OpenTimelineIO pour
  FCP 7 XML / EDL / OTIO, lecteur conforme à la spécification pour FCPXML et
  EDL texte Vegas, exécution du script After Effects dans un After Effects
  simulé), puis **rendu image par image par MLT** (le moteur de Kdenlive et
  Shotcut) : chaque clip commence et finit sur l'image source demandée, y
  compris en mélangeant 23,976 / 25 / 29,97 i/s, avec un clip de 3 images et
  des noms de fichiers avec espaces, accents et apostrophes.
- Sur un vrai Windows (GitHub Actions) : export de chaque format depuis
  l'application installée, avec un dossier de sortie contenant accents et
  japonais.

Ce qui **n'a pas** pu être vérifié : l'ouverture dans Premiere Pro, Resolve,
After Effects, Vegas Pro et Final Cut eux-mêmes. Les checklists ci-dessus sont
là pour ça.
