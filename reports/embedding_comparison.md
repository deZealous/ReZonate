# Embedding comparison: CLAP vs MERT on FMA

Sample: 440 FMA clips (30 s), 11 top-level genres, stratified (~40/genre). Same three 10 s windows per clip for both models, and the per-track vector is the mean over windows. Metrics are computed on centered + L2-normalised vectors.

## Full sample (all views)

| name            |   probe_macro_f1 |   knn_p@10 |   knn_acc |   subgenre_jaccard@10 |   anisotropy_raw |     dim |
|:----------------|-----------------:|-----------:|----------:|----------------------:|-----------------:|--------:|
| random (chance) |            0.091 |      0.085 |     0.105 |                 0.049 |          nan     | nan     |
| clap            |            0.535 |      0.445 |     0.586 |                 0.305 |            0.198 | 512.000 |
| mert_L0         |            0.350 |      0.229 |     0.343 |                 0.153 |            0.850 | 768.000 |
| mert_L1         |            0.386 |      0.237 |     0.350 |                 0.163 |            0.779 | 768.000 |
| mert_L2         |            0.405 |      0.252 |     0.405 |                 0.172 |            0.752 | 768.000 |
| mert_L3         |            0.454 |      0.271 |     0.380 |                 0.187 |            0.761 | 768.000 |
| mert_L4         |            0.480 |      0.285 |     0.402 |                 0.196 |            0.765 | 768.000 |
| mert_L5         |            0.499 |      0.298 |     0.402 |                 0.207 |            0.770 | 768.000 |
| mert_L6         |            0.509 |      0.307 |     0.427 |                 0.211 |            0.783 | 768.000 |
| mert_L7         |            0.491 |      0.303 |     0.398 |                 0.210 |            0.777 | 768.000 |
| mert_L8         |            0.479 |      0.307 |     0.420 |                 0.210 |            0.799 | 768.000 |
| mert_L9         |            0.457 |      0.290 |     0.402 |                 0.197 |            0.801 | 768.000 |
| mert_L10        |            0.418 |      0.265 |     0.375 |                 0.177 |            0.795 | 768.000 |
| mert_L11        |            0.412 |      0.256 |     0.375 |                 0.173 |            0.726 | 768.000 |
| mert_L12        |            0.351 |      0.243 |     0.361 |                 0.162 |            0.804 | 768.000 |
| mert_mean_all   |            0.456 |      0.279 |     0.416 |                 0.191 |            0.805 | 768.000 |

## Held-out head-to-head (MERT layer chosen on half A by subgenre_jaccard@10, scored on half B, n=220)

| name         |   probe_macro_f1 |   knn_p@10 |   knn_acc |   subgenre_jaccard@10 |   anisotropy_raw |   dim |
|:-------------|-----------------:|-----------:|----------:|----------------------:|-----------------:|------:|
| clap         |            0.501 |      0.393 |     0.564 |                 0.243 |            0.204 |   512 |
| mert_L8      |            0.455 |      0.255 |     0.377 |                 0.163 |            0.795 |   768 |
| clap+mert_L8 |            0.493 |      0.372 |     0.550 |                 0.232 |           -0.001 |  1280 |

Selected MERT view: **mert_L8** (use view name `mert:L8`).

## Cost (cpu, 4 threads, this dev container; `scripts/bench_embed.py`)

* clap: 0.44 s/track model + 0.17 s decode (3 x 10 s windows)
* mert: 2.12 s/track model + 0.16 s decode (3 x 10 s windows)
