# Music sync — cut on the beat

[← Back to the README](../README.md)

## Music Sync — Cut On The Beat

The **Music** tab turns a song into a first cut made of your own indexed
shots:

1. **Pick your song** (mp3, wav, flac, m4a, ogg, or the audio of a video).
   AMV Tools finds the tempo, every beat and bar, the strong hits, an energy
   curve and the parts of the song — intro, verse, build-up, drop, chorus,
   calm part, outro — with the drops flagged on the waveform. About 1.5-4 s
   for a 4-minute song on a modest CPU, cached by file content (renaming or moving the
   file keeps the analysis).
2. **Propose an edit.** Each passage gets shots whose intensity matches it
   (calm shots on the intro and calm parts, faster cuts through the build-up,
   action shots cut on 1-2 beats on the drop), using the motion / intensity
   analysis of Search+, the tags and the text search. Choose the series and
   characters, the cut rhythm (steady ↔ nervous), avoid reusing a shot, and
   add keywords per passage ("Goku" on the drop).
3. **Tweak it**: click a shot to swap it for one of its alternatives, lock
   the ones you like, regenerate one passage or everything but the locked
   shots, and watch a **low-resolution MP4 preview** with the music. The
   preview also spots shots that hide a cut of the episode (it would fall
   off the beat): they are marked in red and avoided by the next proposal.
4. **Export** the edit as a timeline (FCP 7 XML for Premiere / Vegas /
   Resolve / After Effects via Premiere, FCPXML, EDL, OTIO) with the song on
   its own audio track and markers on the beats and passages — or only the
   **markers** (beats + passages on the song, to cut by hand; also as a
   DaVinci Resolve marker EDL and CSV).

Every cut is placed on the frame nearest to its beat (at 23.976 fps that is
at most ±21 ms, half a frame). Analysis uses librosa (ISC) and plain signal
processing: no model to download, works on any CPU.
Details and measurements: [v06 C report](reports/v06-C-synchro-musique.md).
