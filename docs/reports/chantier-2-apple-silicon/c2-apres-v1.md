## c2-apres — MacBook-Pro-de-Leandro.local (Apple M1 Pro)

Pipeline total: **5 min 34** (season + search + export + finish); run wall time incl. sidecar starts 6 min 16.

| part | wall | GPU util | CPU | GPU mem peak | RAM peak |
|---|---:|---:|---:|---:|---:|
| season | 3 min 22 | 48 % | 60 % | 3729.8 MB | 4621 MB |
| search | 4.14 s | 11 % | 14 % | 1205.4 MB | 2370 MB |
| export | 5.81 s | 14 % | 45 % | 1864.5 MB | 4258 MB |
| finish | 2 min 01 | – | – | – MB | – MB |

| indexing stage | wall | GPU util | NVDEC | NVENC | CPU |
|---|---:|---:|---:|---:|---:|
| probe | 0.09 s | – | – | – | – |
| decode_cuts_motion | 2 min 18 | 66 % | – | – | 78 % |
| median_frames | 0.00 s | – | – | – | – |
| thumbnails_db | 2.51 s | 44 % | – | – | 12 % |
| tags | 0.04 s | 15 % | – | – | 18 % |
| embeddings | 0.06 s | – | – | – | – |
| finalize | 0.13 s | 15 % | – | – | 18 % |
| hover_proxies | 61.1 s | 8 % | – | – | 23 % |
- finish upscale_x2: 70.5 s (done), GPU 98 %
- finish rife_x2: 17.8 s (done), GPU 93 %
- finish depth: 33.0 s (done), GPU 70 %
- search: first text query (cold) 3.678 s, warm text 25.1 ms, image [127.2, 33.8, 36.4] ms, similar [7.6, 7.3, 6.5, 6.4, 7.0] ms
- swap-outs during the run: 0
