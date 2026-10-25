## c2-avant — MacBook-Pro-de-Leandro.local (Apple M1 Pro)

Pipeline total: **7 min 07** (season + search + export + finish); run wall time incl. sidecar starts 7 min 51.

| part | wall | GPU util | CPU | GPU mem peak | RAM peak |
|---|---:|---:|---:|---:|---:|
| season | 4 min 47 | 37 % | 68 % | 3639.9 MB | 4614 MB |
| search | 4.20 s | 11 % | 15 % | 1637.8 MB | 2393 MB |
| export | 11.6 s | 10 % | 27 % | 1705.7 MB | 2646 MB |
| finish | 2 min 05 | – | – | – MB | – MB |

| indexing stage | wall | GPU util | NVDEC | NVENC | CPU |
|---|---:|---:|---:|---:|---:|
| probe | 0.09 s | 16 % | – | – | 24 % |
| decode_cuts_motion | 2 min 15 | 69 % | – | – | 77 % |
| median_frames | 0.00 s | – | – | – | – |
| thumbnails_db | 2.46 s | 50 % | – | – | 16 % |
| tags | 0.04 s | – | – | – | – |
| embeddings | 0.05 s | – | – | – | – |
| finalize | 0.13 s | 0 % | – | – | 12 % |
| hover_proxies | 2 min 29 | 8 % | – | – | 61 % |
- finish upscale_x2: 70.6 s (done), GPU 98 %
- finish rife_x2: 17.8 s (done), GPU 92 %
- finish depth: 36.2 s (done), GPU 64 %
- search: first text query (cold) 3.718 s, warm text 26.9 ms, image [130.8, 39.8, 37.9] ms, similar [8.1, 6.7, 6.2, 6.9, 6.6] ms
- swap-outs during the run: 0
