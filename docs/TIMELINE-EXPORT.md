# Send to your editor — timeline export

[← Back to the README](../README.md)

## Send To Your Editor

AMV Tools can hand a selection to a real NLE as a **timeline of cuts** that
points at the original media. Nothing is re-encoded: every writer stores the
source path plus frame-accurate in/out points, so the editor sees the exact
frames you picked.

Where the button lives:

- **Derush → Library**: the current filter (all keeps, a folder, favorites or
  unfiled) is exported.
- **Tags**: the selected scenes.
- **Search**: the current result set.

The dialog asks which editor you use (Premiere Pro, DaVinci Resolve, After
Effects, Vegas Pro, Final Cut Pro, Other), shows how to import there, previews
the clip count and total duration, then asks where to save the timeline.
Step-by-step import guide with a 5-minute checklist per editor (FR):
[import-timeline.md](import-timeline.md). Formats:

| Format | Best for | Notes |
| --- | --- | --- |
| **FCP 7 XML** (xmeml v5) | Premiere Pro (also Resolve) | Exact integer-frame in/out, NTSC flag, labels, markers; `pathurl` spelled like Premiere's own exports (`file://localhost/C%3a/...`). |
| **FCPXML 1.9** | Final Cut Pro 10.4+, Resolve | Rational times (`1001/24000s`), mixed source rates. |
| **EDL CMX3600** | Any NLE, as a universal cut list | 8-char reel names; `* FROM CLIP NAME:` / `* FROM FILE:` comments on one line each (what NLEs relink from). |
| **OpenTimelineIO** | Resolve, Kdenlive, OTIO tools | Native OTIO JSON (with markers), written by AMV Tools itself — no extra package needed. |
| **Vegas EDL text** (`.txt`) | Vegas Pro | Vegas' own event list (semicolons, milliseconds) for `File → Import → EDL Text File`; full source paths. |
| **After Effects script** (`.jsx`) | After Effects | ExtendScript that imports the sources, creates the comp and places every clip with exact in/out. |

Options in the dialog:

- **Sequence frame rate / resolution**: first clip or an explicit choice
  (23.976, 24, 25, 29.97, 30, 59.94, 60; 720p → 4K).
- **Back-to-back or gaps**: insert a configurable gap of N frames between
  clips.
- **Handles**: extend each clip by N frames before/after (clamped to the
  source).
- **Drop-frame**: for 29.97/59.94 only. Non-drop is the default.
- **Scene markers / labels / folders**: markers carry the scene name and tags
  in every format; favorites get the Mango label in FCP XML.
- **Include source audio**: omit it for a video-only assembly.
- **Collect media**: copies the originals (byte-for-byte, no transcode) into
  `<timeline>_media/` beside the timeline and points every clip at the
  copies, so you can send the timeline + its `_media` folder to another
  editor. How the paths are written depends on what each format can express:

  | Format | Media path after “Collect media” |
  | --- | --- |
  | FCPXML, OpenTimelineIO | Relative to the timeline file: `timeline_media/clip.mp4`. Moving the timeline and its `_media` folder together keeps the media linked. |
  | EDL | No media path in the format; the `* FROM FILE:` comment carries the same relative path. |
  | FCP 7 XML | **Absolute** `file://` URL of the copy, e.g. `file://localhost/C%3a/Edits/timeline_media/clip.mp4`. |

  Why FCP 7 XML stays absolute: xmeml's `<pathurl>` is a `file://` URL and
  Premiere Pro, Final Cut Pro 7, Resolve and Vegas do not resolve relative
  paths in it — a relative value imports as offline media everywhere. On the
  machine that exported, the copies open directly. On another machine, the
  importer asks to relink: point it at the `_media` folder once (Premiere:
  *Locate* with “Relink others automatically”; Resolve: *Relink Selected
  Clips* on the folder) — `<name>` holds the file name, so every clip is
  found by name in that folder.

VFR sources are approximated using their average frame rate; the preview
shows a warning. Mixed frame-rate sources are converted to the sequence rate
without retiming the media.

### Importing step by step

**Premiere Pro** — `File → Import…`, pick the exported `.xml` (FCP 7 XML).
The sequence appears in the project panel with clips already cut and the
media linked. For AE, import the same XML into Premiere, then use
`File → Adobe Dynamic Link → Import Premiere Pro Sequence` in After Effects
(or open the sequence in Premiere and `Edit → Edit in Adobe After Effects`).
If Premiere asks to relink, point it at the collected `_media` folder.

**DaVinci Resolve** — create/open a project, then
`File → Import → Timeline…` and choose the `.fcpxml` (or `.xml` / `.otio`).
Resolve maps reel names and clip labels; choose “Use source frame rate” if
prompted for mixed-rate sources. For EDL use
`File → Import → Pre-conformed EDL…` and select the sequence frame rate.

**Final Cut Pro** — `File → Import → XML…`, select the `.fcpxml`. The event
and project are created automatically. FCP 7 XML can also be imported by FCP
10 through `File → Import → XML…` in most versions; FCPXML is the recommended
path.

**Vegas Pro** — `File → Import → EDL Text File (*.txt)`, select the exported
`.txt` (choose *Vegas Pro* in the dialog). Video lands on track 1, audio on
track 2. Vegas 14+ can also try the FCP 7 XML through
`File → Import → Final Cut Pro 7 / DaVinci Resolve Timeline (*.xml)`.

**After Effects** — choose *After Effects* in the dialog, then in AE
`File → Scripts → Run Script File…` and pick the `.jsx`: it imports each
source once, creates a composition at the timeline's rate and size and places
every clip as a layer with its exact in/out (one undo step). Missing files
are listed in an alert; files next to the script are found too.

**EDL in any NLE** — import the `.edl` as a pre-conformed/CMX3600 EDL. The
`* FROM CLIP NAME:` and `* FROM FILE:` comment lines identify the source;
EDL itself does not carry media paths, so relink from those comments if your
NLE asks.
