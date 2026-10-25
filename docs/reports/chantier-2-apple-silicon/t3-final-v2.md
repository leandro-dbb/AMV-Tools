## t3-final — MacBook-Pro-de-Leandro.local (Apple M1 Pro)

Pipeline total: **3 min 04** (season + search + export + finish); run wall time incl. sidecar starts 3 min 47.

| part | wall | GPU util | CPU | GPU mem peak | RAM peak |
|---|---:|---:|---:|---:|---:|
| season | 101.7 s | 69 % | 68 % | 2428.8 MB | 4411 MB |
| search | 4.16 s | 11 % | 14 % | 1622.6 MB | 2411 MB |
| export | 6.22 s | 9 % | 78 % | 1790.6 MB | 4163 MB |
| finish | 71.6 s | – | – | – MB | – MB |

| indexing stage | wall | GPU util | NVDEC | NVENC | CPU |
|---|---:|---:|---:|---:|---:|
| probe | 0.15 s | 0 % | – | – | 7 % |
| decode_cuts_motion | 92.2 s | 75 % | – | – | 69 % |
| median_frames | 0.00 s | – | – | – | – |
| thumbnails_db | 1.87 s | 55 % | – | – | 16 % |
| tags | 0.04 s | – | – | – | – |
| embeddings | 0.06 s | 16 % | – | – | 7 % |
| finalize | 0.75 s | 15 % | – | – | 46 % |
| hover_proxies | 6.47 s | 7 % | – | – | 75 % |
- finish upscale_x2: 37.2 s (done), GPU 94 %
- finish rife_x2: 13.1 s (done), GPU 88 %
- finish depth: 21.4 s (done), GPU 79 %
- search: first text query (cold) 3.636 s, warm text 27.8 ms, image [144.1, 51.4, 35.2] ms, similar [8.7, 7.4, 6.9, 7.4, 7.7] ms
- swap-outs during the run: 0
