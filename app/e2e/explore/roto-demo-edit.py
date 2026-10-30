"""Edit the roto editor demo recording (roto-demo.mjs) into an MP4: the
waits for the AI pass, the propagations and the export are sped up so each
lasts a few seconds, with a "×N" label; everything else plays in real time.
usage: python roto-demo-edit.py <demo.webm> <demo-events.json> <out.mp4>"""
import json
import subprocess
import sys

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MAX_WAIT_S = 4.0

src, ev_path, out = sys.argv[1:4]
events = json.load(open(ev_path))
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src],
                           capture_output=True, text=True).stdout.strip() or 0)
waits, start = [], None
for e in events:
    if e["kind"] == "wait-start":
        start = (e["t"], e["label"])
    elif e["kind"] == "wait-end" and start:
        waits.append((start[0], e["t"], start[1]))
        start = None
segs, t = [], 0.0
for a, b, label in waits:
    if a > t:
        segs.append((t, a, 1.0, None))
    speed = max(1.0, (b - a) / MAX_WAIT_S)
    segs.append((a, b, speed, label))
    t = b
if t < dur:
    segs.append((t, dur, 1.0, None))
names = {"detect": "passe IA", "propagate": "propagation", "export": "export"}
parts, labels = [], []
for i, (a, b, speed, label) in enumerate(segs):
    f = f"[0:v]trim={a:.3f}:{b:.3f},setpts=(PTS-STARTPTS)/{speed:.4f}"
    if label and speed > 1.05:
        txt = f"{names.get(label, label)} — accéléré ×{speed:.0f} (durée réelle {b - a:.0f} s)"
        f += (f",drawtext=fontfile={FONT}:text='{txt}':x=(w-tw)/2:y=24:fontsize=28:fontcolor=white:"
              "box=1:boxcolor=black@0.7:boxborderw=12")
    parts.append(f + f"[v{i}]")
    labels.append(f"[v{i}]")
graph = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(labels)}:v=1:a=0,fps=30[out]"
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-filter_complex", graph,
                "-map", "[out]", "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", out], check=True)
print("ok", out, [(round(a, 1), round(b, 1), round(s, 1)) for a, b, s, _ in segs])
