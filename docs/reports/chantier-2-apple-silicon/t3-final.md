## t3-final — MacBook-Pro-de-Leandro.local (Apple M1 Pro)

Pipeline total: **3 min 03** (season + search + export + finish); run wall time incl. sidecar starts 3 min 46.

| part | wall | GPU util | CPU | GPU mem peak | RAM peak |
|---|---:|---:|---:|---:|---:|
| season | 101.2 s | 70 % | 69 % | 2393.1 MB | 4541 MB |
| search | 4.13 s | 11 % | 14 % | 1623.3 MB | 2418 MB |
| export | 6.14 s | 12 % | 76 % | 1806.1 MB | 4244 MB |
| finish | 71.4 s | – | – | – MB | – MB |

| indexing stage | wall | GPU util | NVDEC | NVENC | CPU |
|---|---:|---:|---:|---:|---:|
| probe | 0.13 s | – | – | – | – |
| decode_cuts_motion | 91.8 s | 76 % | – | – | 70 % |
| median_frames | 0.00 s | – | – | – | – |
| thumbnails_db | 1.88 s | 45 % | – | – | 16 % |
| tags | 0.04 s | – | – | – | – |
| embeddings | 0.05 s | – | – | – | – |
| finalize | 0.74 s | 15 % | – | – | 50 % |
| hover_proxies | 6.52 s | 7 % | – | – | 79 % |
- finish upscale_x2: 37.3 s (done), GPU 93 %
- finish rife_x2: 12.9 s (done), GPU 88 %
- finish depth: 21.3 s (done), GPU 79 %
- search: first text query (cold) 3.595 s, warm text 26.9 ms, image [156.8, 47.4, 34.0] ms, similar [8.9, 7.6, 6.7, 7.0, 7.1] ms
- swap-outs during the run: 0
