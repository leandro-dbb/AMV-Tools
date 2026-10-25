## c2-apres — MacBook-Pro-de-Leandro.local (Apple M1 Pro)

Pipeline total: **4 min 52** (season + search + export + finish); run wall time incl. sidecar starts 5 min 36.

| part | wall | GPU util | CPU | GPU mem peak | RAM peak |
|---|---:|---:|---:|---:|---:|
| season | 2 min 40 | 59 % | 78 % | 3807.9 MB | 4490 MB |
| search | 4.13 s | 12 % | 14 % | 1677.4 MB | 2400 MB |
| export | 6.42 s | 11 % | 72 % | 1790.2 MB | 4167 MB |
| finish | 2 min 02 | – | – | – MB | – MB |

| indexing stage | wall | GPU util | NVDEC | NVENC | CPU |
|---|---:|---:|---:|---:|---:|
| probe | 0.09 s | – | – | – | – |
| decode_cuts_motion | 2 min 18 | 66 % | – | – | 78 % |
| median_frames | 0.00 s | – | – | – | – |
| thumbnails_db | 2.44 s | 33 % | – | – | 15 % |
| tags | 0.04 s | – | – | – | – |
| embeddings | 0.06 s | – | – | – | – |
| finalize | 0.13 s | – | – | – | – |
| hover_proxies | 18.9 s | 8 % | – | – | 84 % |
- finish upscale_x2: 71.0 s (done), GPU 98 %
- finish rife_x2: 17.6 s (done), GPU 91 %
- finish depth: 33.0 s (done), GPU 70 %
- search: first text query (cold) 3.641 s, warm text 26.1 ms, image [137.3, 39.4, 39.4] ms, similar [9.6, 7.5, 6.4, 6.3, 6.6] ms
- swap-outs during the run: 0
