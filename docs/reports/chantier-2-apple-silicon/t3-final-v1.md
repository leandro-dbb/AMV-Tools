## t3-final — MacBook-Pro-de-Leandro.local (Apple M1 Pro)

Pipeline total: **3 min 05** (season + search + export + finish); run wall time incl. sidecar starts 3 min 49.

| part | wall | GPU util | CPU | GPU mem peak | RAM peak |
|---|---:|---:|---:|---:|---:|
| season | 102.7 s | 70 % | 68 % | 2403.7 MB | 4431 MB |
| search | 3.99 s | 14 % | 14 % | 1883.3 MB | 2400 MB |
| export | 7.38 s | 9 % | 84 % | 1778.7 MB | 4969 MB |
| finish | 71.3 s | – | – | – MB | – MB |

| indexing stage | wall | GPU util | NVDEC | NVENC | CPU |
|---|---:|---:|---:|---:|---:|
| probe | 0.12 s | – | – | – | – |
| decode_cuts_motion | 92.7 s | 76 % | – | – | 69 % |
| median_frames | 0.00 s | – | – | – | – |
| thumbnails_db | 2.51 s | 55 % | – | – | 12 % |
| tags | 0.04 s | – | – | – | – |
| embeddings | 0.05 s | – | – | – | – |
| finalize | 0.75 s | 15 % | – | – | 48 % |
| hover_proxies | 6.46 s | 8 % | – | – | 78 % |
- finish upscale_x2: 37.2 s (done), GPU 94 %
- finish rife_x2: 12.8 s (done), GPU 87 %
- finish depth: 21.3 s (done), GPU 78 %
- search: first text query (cold) 3.36 s, warm text 29.1 ms, image [237.8, 50.9, 34.5] ms, similar [8.4, 7.6, 7.3, 7.6, 7.6] ms
- swap-outs during the run: 0
