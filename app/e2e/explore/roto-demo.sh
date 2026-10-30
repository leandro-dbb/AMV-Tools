#!/bin/bash
# Screen recording of the roto editor demo on our own Xvfb display (never the
# user's): Xvfb :97 1600x1000, ffmpeg x11grab, e2e/explore/roto-demo.mjs.
# usage: e2e/explore/roto-demo.sh <out dir> <plan.json>
set -e
OUT=$1; PLAN=$2
mkdir -p "$OUT"
Xvfb :97 -screen 0 1600x1000x24 -nolisten tcp > "$OUT/xvfb.log" 2>&1 &
XPID=$!
sleep 2
date +%s.%N > "$OUT/rec-start.txt"
ffmpeg -hide_banner -loglevel error -y -f x11grab -draw_mouse 0 -framerate 30 -video_size 1600x1000 -i :97 \
  -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p "$OUT/screen.mkv" < /dev/null &
FPID=$!
DEMO_CHROME_PX=100 DISPLAY=:97 node "$(dirname "$0")/roto-demo.mjs" "$OUT" "$PLAN" || echo "demo script failed: $?"
sleep 1
kill -INT $FPID; wait $FPID || true
kill $XPID || true
