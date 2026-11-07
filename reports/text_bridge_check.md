# CLAP text bridge check (SIMULATED listener)

30 loved tracks with audio hidden, 8 seeds; catalogue ranked by kNN similarity to the anchors.

|                          |   NDCG@50 mean |   std |
|:-------------------------|---------------:|------:|
| FMA free tags (raw)      |          0.274 | 0.026 |
| FMA free tags (centered) |          0.239 | 0.073 |
| genre words (raw)        |          0.378 | 0.051 |
| genre words (centered)   |          0.401 | 0.040 |
| real audio (oracle)      |          0.422 | 0.066 |
| random                   |          0.269 | 0.014 |

Reading: with noisy free tags (artist/place names) the bridge is at chance. With genre-style descriptions it is close to using the real audio. That condition is optimistic, since the simulated listener is defined by genre labels, but it supports using the bridge with Last.fm/MusicBrainz tags only, which is what `service.text_anchors` now does. Centering makes little difference.
