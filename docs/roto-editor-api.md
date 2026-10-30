# Roto editor — sidecar API (v06)

All routes are under the sidecar base URL and need the usual auth header
(`authHeaders()` in `app/src/api/client.ts`). Image routes also accept the
token the way the existing Mask Mode image URLs do.

A *project* is one shot being rotoscoped. It lives on disk
(`<user data>/roto_projects/<id>/`) and is saved automatically after every
change, so the editor can be closed and reopened later.

Frame indices are 0-based, coordinates are in **working-frame pixels**
(`width` x `height` of the project, ~1080 px on the long side).

## Types

```ts
type RotoReason = 'lost' | 'jump' | 'specks' | 'object_lost' | 'hole' | 'rough';

interface RotoSuspect { frame: number; score: number; reasons: RotoReason[] }

interface RotoLayer { id: string; name: string; color: string }   // color: '#rrggbb'

interface RotoEdgeSettings {
  keep_ink: boolean;   // "Keep the black line" (default true)
  softness: number;    // -1 (harder) .. +1 (softer), default 0
  grow_px: number;     // -6 .. +6 px (negative = shrink), default 0
  temporal: number;    // 0 .. 1 smoothing over time, default 0
}

interface RotoJob {
  kind: 'detect' | 'propagate' | 'export';
  percent: number;          // 0..100
  message: string;          // short, already human readable (EN)
  done: boolean;
  cancelled: boolean;
  error: string | null;
  result?: any;             // export: { files: {prores?, png_dir?, matte?}, added_to_selection: boolean }
}

interface RotoModels {
  device: 'cuda' | 'mps' | 'cpu';
  detector: string;          // e.g. "BiRefNet"
  characters: string | null; // "SAM 3" or null when not available
  tracker: string | null;    // e.g. "SAM 2.1 large", null = no video tracking on this machine
  clicks: string;            // model answering clicks, e.g. "SAM 2.1 large" or "BiRefNet (character under the click)"
  sam3_missing: boolean;     // true -> show "SAM 3 not available (no Hugging Face access): SAM 2.1 large used instead"
  slow_machine: boolean;     // true on CPU: live correction can take a few seconds
}

interface RotoProject {
  id: string;
  scene_id: number | null;
  name: string;
  n_frames: number;
  fps: number;
  width: number;
  height: number;
  start_ms: number;
  end_ms: number;
  has_ai_pass: boolean;      // false until the first "Détourer" finished
  layers: RotoLayer[];
  keyframes: Record<string, number[]>;  // layer id -> frames locked by the user (edited or validated)
  suspects: RotoSuspect[];   // sorted by frame
  settings: RotoEdgeSettings;
  models: RotoModels;
  job: RotoJob | null;       // running or last finished job
  version: number;           // bumps on every matte change (use as cache buster)
  frame_versions: Record<string, number>; // frame -> version of its last change (fine-grained cache busting)
  exports: { prores?: string; png_dir?: string; matte?: string } | null;
  can_undo: boolean;
  can_redo: boolean;
}
```

## Routes

| method | path | body | returns |
|---|---|---|---|
| POST | `/api/roto/open` | `{scene_id, start_ms?, end_ms?}` | `RotoProject` (reuses the saved project of the same scene + range) |
| GET | `/api/roto/{id}` | | `RotoProject` |
| GET | `/api/roto/{id}/frame/{i}` | | JPEG of working frame i |
| GET | `/api/roto/{id}/matte/{i}?layer=<id>&v=<n>` | | 8-bit grayscale PNG (white = character) of frame i **with the edge settings applied**. No `layer` = all layers combined. 404 before the first AI pass unless the frame was painted. |
| POST | `/api/roto/{id}/detect` | `{layer?: string, target: 'all' \| 'main' \| 'click', frame?: number, positive?: [x,y][], negative?: [x,y][]}` | `RotoJob` (starts the AI pass in the background; `target:'click'` = only the character(s) clicked on `frame`) |
| GET | `/api/roto/{id}/job` | | `RotoJob \| null` (poll every ~300 ms while running) |
| POST | `/api/roto/{id}/cancel` | | `{ok}` |
| POST | `/api/roto/{id}/brush` | `{frame, layer, strokes: [{points: [x,y][], radius, mode: 'add' \| 'remove' \| 'erase'}]}` | `{ms, version, frame, keyframes, suspects}` — `erase` = put back the AI pass under the stroke. The frame becomes a locked keyframe. |
| POST | `/api/roto/{id}/click` | `{frame, layer, x, y, positive: boolean}` | `{ms, version, frame, keyframes, suspects, model}` — positive: adds the part under the click, negative: removes it. Frame locked. |
| POST | `/api/roto/{id}/lock` | `{frame, layer, locked: boolean}` | `{keyframes, suspects}` (validate a frame as correct without editing it) |
| POST | `/api/roto/{id}/propagate` | `{frame, layer, direction: 'forward' \| 'backward' \| 'both'}` | `RotoJob` — video tracking from the corrected frame(s); locked frames are never overwritten; stops at the next locked frame in each direction. |
| POST | `/api/roto/{id}/undo` / `/redo` | | `{version, frame, can_undo, can_redo, keyframes, suspects}` (`frame` = the frame the undone step touched, to jump there) |
| PUT | `/api/roto/{id}/settings` | `RotoEdgeSettings` | `RotoProject` |
| POST | `/api/roto/{id}/layers` | `{name?}` | `RotoProject` (adds an empty layer = another character) |
| DELETE | `/api/roto/{id}/layers/{layer}` | | `RotoProject` |
| POST | `/api/roto/{id}/export` | `{formats: ('prores' \| 'png' \| 'matte')[], add_to_selection: boolean}` | `RotoJob` (`result.files`) |
| GET | `/api/roto/scene/{scene_id}` | | `{projects: {id, start_ms, end_ms, has_ai_pass, exported}[]}` |

Errors: HTTP 4xx/5xx with `{detail: "<human readable>"}`; 409 when a job is already
running.
